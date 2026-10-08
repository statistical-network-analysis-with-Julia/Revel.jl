# =============================================================================
# Interactions and statistic wrappers
# =============================================================================
#
# The literature lets a covariate moderate an endogenous effect in several ways,
# and they are NOT interchangeable:
#
#   product term     Interaction(a, b)              — this file
#   filtered         EventLayer(keep=…), MatchedDegree, matching_third
#   type split       EventLayer(types=…), split_by_type, BalanceEffect
#   attribute-weighted  TertiusEffect
#   stratified       fit_stratified                 — src/fit.jl
#   time-varying     fit_moving_window, GlobalEffect × effect
#   random slope / cross-level                       — not implemented
#
# The wrappers below only call the shared `compute` generic, so their parts may
# be any statistic that evaluates on an InteractionHistory;
# a REM statistic, which has only REM's interface, works in them through
# `REM.fit_rem` but not through `fit_revel`.

_part_name(s::AbstractStatistic) = name(s)

"""
    Interaction(stats...; name=nothing)

The product of two or more statistics: the explicit product-term interaction
(remstats `a:b`; remulate `interact()`). `Interaction(Inertia(), SendEffect(x))`
moderates an endogenous effect by a covariate and
`Interaction(OutdegreeSender(), IndegreeReceiver())` is an endogenous ×
endogenous term. The parts may be any statistic with the history interface
(`PShift`, a user's own, …); a REM.jl statistic has only REM's interface, so an interaction
holding one works in `REM.fit_rem` but not in [`fit_revel`](@ref).

The advice that recurs in the applied literature, though few papers state it:
include the main effects alongside the product, scale cumulative statistics
before interacting them (see [`Standardized`](@ref) and the `transform=:log1p`
keyword), and centre a covariate first (see [`Transformed`](@ref)) so the main
effects stay interpretable.

# Example
```julia
using Revel
female = [1.0, 0.0, 1.0]
h = build_history([Event(2, 1, 1.0), Event(2, 1, 2.0)])
recip_by_sender = Interaction(Reciprocation(), SendEffect(female; name="female"))
name(recip_by_sender)                        # "reciprocity:female"
compute(recip_by_sender, h, 1, 2, 3.0)       # 2.0 × 1.0
compute(recip_by_sender, h, 2, 1, 3.0)       # 0.0
```
"""
struct Interaction{S<:Tuple} <: AbstractRevelStatistic
    parts::S
    label::String
end

function Interaction(stats::AbstractStatistic...; name=nothing)
    length(stats) >= 2 || throw(ArgumentError(
        "an Interaction needs at least two statistics"))
    auto = join((_part_name(s) for s in stats), ":")
    return Interaction{typeof(stats)}(stats, _label(name, auto))
end

# The parts may be foreign statistics, whose methods want the history's own time
# type: convert once here (a fractional time on an integer clock is passed on
# as it is; Revel's statistics accept any real time)
_part_time(::Type{T}, t) where T = t isa T ? t :
    (T <: Integer && t isa Real && !isinteger(t)) ? t : convert(T, t)

function compute(stat::Interaction, history::InteractionHistory{T}, s::Int, r::Int,
                 t) where T
    tt = _part_time(T, t)
    return prod(map(p -> compute(p, history, s, r, tt), stat.parts))
end
compute(stat::Interaction, state::REM.EventNetworkState, s::Int, r::Int) =
    prod(map(p -> compute(p, state, s, r), stat.parts))

_uses_history(stat::Interaction) = any(REM.needs_history, stat.parts)
_interval_constant(stat::Interaction) = all(is_interval_constant, stat.parts)

"""
    Transformed(stat, f; name=nothing)

`f` applied to the value of another statistic — a scaling (`log1p`, `sqrt`), a
centring (`x -> x - 3.2`), a dichotomisation. It wraps any statistic of the
ecosystem, including those that have no `transform=` keyword of their own.

Centring before forming a product term is the use the review singles out as
undocumented in the REM literature: `Interaction(Inertia(),
Transformed(SendEffect(age), x -> x - 40))` leaves the inertia main effect
meaning "inertia for a 40-year-old sender".

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 2, 2.0), Event(1, 2, 3.0)])
compute(Transformed(Inertia(), log1p), h, 1, 2, 4.0)                        # log(4)
compute(Transformed(SendEffect([30.0, 50.0]), x -> x - 40), h, 2, 1, 4.0)   # 10.0
```
"""
struct Transformed{S<:AbstractStatistic, F} <: AbstractRevelStatistic
    stat::S
    f::F
    label::String
end

function Transformed(stat::AbstractStatistic, f; name=nothing)
    fn = _transform_fn(f)
    tl = _transform_label(fn)
    auto = (isempty(tl) || startswith(tl, "#")) ? "f(" * Revel.name(stat) * ")" :
           "$tl(" * Revel.name(stat) * ")"
    return Transformed{typeof(stat), typeof(fn)}(stat, fn, _label(name, auto))
end

compute(stat::Transformed, history::InteractionHistory{T}, s::Int, r::Int,
        t) where T =
    Float64(stat.f(compute(stat.stat, history, s, r, _part_time(T, t))))
compute(stat::Transformed, state::REM.EventNetworkState, s::Int, r::Int) =
    Float64(stat.f(compute(stat.stat, state, s, r)))

_uses_history(stat::Transformed) = REM.needs_history(stat.stat)
_interval_constant(stat::Transformed) = is_interval_constant(stat.stat)

mutable struct _StdCache
    source::WeakRef
    len::Int
    t::Float64
    mean::Float64
    sd::Float64
    values::Vector{Float64}
end

_StdCache() = _StdCache(WeakRef(nothing), -1, NaN, 0.0, 1.0, Float64[])

# A copied statistic starts with an empty cache (see `_fresh`)
Base.deepcopy_internal(::_StdCache, seen::IdDict) = _StdCache()

"""
    Standardized(stat, n_actors; directed=true, corrected=false, name=nothing)
    Standardized(stat, dyads; corrected=false, name=nothing)

`stat` z-scored at every event time: its value minus the mean over a fixed set
of dyads, divided by their standard deviation. The set is every dyad among
actors `1:n_actors` (unordered pairs with `directed=false`), or the explicit
vector `dyads` of `(sender, receiver)` — a two-mode risk set, say. A statistic
that is constant across the set standardises to `0`.

The set must be the model's risk set, and the fitters check it: `riskset=:full`
with the same `n_actors` and `directed`, or `riskset=dyads` with the same dyads.
Risk sets that change from event to event (`:active` is fixed, but `:sender`,
`:receiver` and function risk sets are not) are refused, because remstats'
`scaling = "std"` — the construction this reproduces — standardises over the
risk set of each event.

`corrected=false` divides by the population standard deviation (denominator
`D`, the number of dyads); `corrected=true` by the sample one (denominator
`D − 1`), which is what remstats computes. The two differ by the constant factor
`sqrt((D − 1)/D)`, so they rescale the coefficient and nothing else.

Cumulative statistics grow without bound over an event sequence, which makes
their coefficients hard to compare across effects and can set off a feedback
loop in simulation ("process explosion"); per-time-point standardisation is one
remedy in the literature (Vieira, Leenders & Mulder 2024), a `log1p` transform
another. It is **a different model, not a rescaling**: the mean cancels within a
risk set but the standard deviation `σ_t` does not, so a coefficient `θ` on the
standardised statistic is an effect of `θ/σ_t` per raw unit, which shrinks as
the history accumulates and `σ_t` grows. Fitting `Standardized(Inertia(), n)`
to data generated by a constant raw inertia effect therefore misfits, and the
score-process test detects it.

The moments are computed once per event time, from one extra evaluation of
every dyad of the set.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 2, 2.0)])
z = Standardized(Inertia(), 3)
compute(z, h, 1, 2, 3.0)    # ≈ 2.24 — the only dyad with a history
compute(z, h, 2, 3, 3.0)    # ≈ -0.45
two_mode = Standardized(Inertia(), two_mode_dyads(1:2, 3:4))
compute(two_mode, h, 1, 3, 3.0)   # 0.0 — no dyad of the set has a history
```
"""
struct Standardized{S<:AbstractStatistic} <: AbstractRevelStatistic
    stat::S
    n_actors::Int
    directed::Bool
    dyads::Union{Nothing, Vector{Tuple{Int,Int}}}
    corrected::Bool
    cache::_StdCache
    label::String
end

function Standardized(stat::AbstractStatistic, n_actors::Int; directed::Bool=true,
                      corrected::Bool=false, name=nothing)
    n_actors >= (corrected && !directed ? 3 : 2) || throw(ArgumentError(
        "Standardized needs at least two dyads in the risk set"))
    return Standardized{typeof(stat)}(stat, n_actors, directed, nothing, corrected,
                                      _StdCache(),
                                      _label(name, "std(" * Revel.name(stat) * ")"))
end

function Standardized(stat::AbstractStatistic, dyads::AbstractVector; corrected::Bool=false,
                      name=nothing)
    set = [(Int(s), Int(r)) for (s, r) in dyads]
    length(set) >= (corrected ? 3 : 2) || throw(ArgumentError(
        "Standardized needs at least two dyads in the risk set"))
    allunique(set) || throw(ArgumentError("the dyads of Standardized list a dyad twice"))
    return Standardized{typeof(stat)}(stat, 0, true, set, corrected, _StdCache(),
                                      _label(name, "std(" * Revel.name(stat) * ")"))
end

function _each_std_dyad(f, stat::Standardized)
    if stat.dyads === nothing
        n = stat.n_actors
        for i in 1:n, j in (stat.directed ? (1:n) : ((i + 1):n))
            i == j || f(i, j)
        end
    else
        for (i, j) in stat.dyads
            f(i, j)
        end
    end
    return nothing
end

function _standardize(stat::Standardized, source, len::Int, t::Float64, value, s::Int,
                      r::Int)
    c = stat.cache
    if c.source.value !== source || c.len != len || c.t != t
        vals = c.values
        empty!(vals)
        _each_std_dyad((i, j) -> push!(vals, value(i, j)), stat)
        μ = sum(vals) / length(vals)
        # two passes over the stored values: no cancellation for a near-constant statistic
        ss = sum(v -> (v - μ)^2, vals)
        c.source = WeakRef(source); c.len = len; c.t = t; c.mean = μ
        c.sd = sqrt(ss / (stat.corrected ? length(vals) - 1 : length(vals)))
    end
    return c.sd > 0 ? (value(s, r) - c.mean) / c.sd : 0.0
end

function compute(stat::Standardized, history::InteractionHistory{T}, s::Int, r::Int,
                 t) where T
    tt = _part_time(T, t)
    return _standardize(stat, history.events, length(history.events), _tfloat(tt),
                        (i, j) -> compute(stat.stat, history, i, j, tt), s, r)
end
compute(stat::Standardized, state::REM.EventNetworkState, s::Int, r::Int) =
    _standardize(stat, state, state.n_events, _tfloat(state.current_time),
                 (i, j) -> compute(stat.stat, state, i, j), s, r)

_uses_history(stat::Standardized) = REM.needs_history(stat.stat)
_interval_constant(stat::Standardized) = is_interval_constant(stat.stat)

# The statistics a wrapper is built from
_parts(stat::Interaction) = stat.parts
_parts(stat::Transformed) = (stat.stat,)
_parts(stat::Standardized) = (stat.stat,)
_parts(::Any) = ()

function _foreach_part(f, stat)
    f(stat)
    foreach(p -> _foreach_part(f, p), _parts(stat))
    return nothing
end

# A Standardized statistic must standardise over the model's risk set
function _check_standardized(statistics, riskset, n_actors::Int, directed::Bool)
    for top in statistics
        _foreach_part(top) do stat
            stat isa Standardized || return
            where_ = "$(name(stat)) standardises over "
            if riskset === :full
                stat.dyads === nothing && stat.n_actors == n_actors &&
                    stat.directed == directed && return
                throw(ArgumentError(where_ *
                    (stat.dyads === nothing ?
                     "the $(stat.directed ? "" : "un")directed dyads among 1:$(stat.n_actors)" :
                     "an explicit list of $(length(stat.dyads)) dyads") *
                    ", but the model's risk set is every $(directed ? "" : "un")directed " *
                    "dyad among 1:$n_actors. Build it as `Standardized(stat, $n_actors" *
                    (directed ? "" : "; directed=false") * ")`."))
            elseif riskset isa AbstractVector
                set = Set((Int(a), Int(b)) for (a, b) in riskset)
                stat.dyads !== nothing && Set(stat.dyads) == set && return
                throw(ArgumentError(where_ * "a different set of dyads than the " *
                    "model's risk set; build it as `Standardized(stat, riskset)` with " *
                    "the same dyads."))
            elseif riskset isa _CaseRestricted
                throw(ArgumentError(where_ * "a fixed set of dyads, but the `cases` " *
                    "predicate depends on the acting dyad, so each case's risk set is " *
                    "the part of it the predicate admits and changes from event to " *
                    "event. Standardized is available with riskset=:full or a vector " *
                    "of dyads and a `cases` selection that does not depend on the dyad."))
            else
                throw(ArgumentError(where_ * "a fixed set of dyads, but " *
                    "riskset=$(repr(riskset)) " *
                    (riskset === :active ? "is the set of dyads active in the sequence; " *
                     "pass that set explicitly, as a vector, to both" :
                     "changes from event to event") *
                    ". Standardized is available with riskset=:full or a vector of dyads."))
            end
        end
    end
    return nothing
end

"""
    split_by_type(constructor, types; kwargs...) -> Vector

One copy of an effect per past event type: `constructor(; types=ty, kwargs...)`
for each `ty` in `types`. This is the type-split interaction — signed inertia
and reciprocity in Brandes, Lerner & Snijders (2009), remstats
`consider_type = "separate"`, one attribute per event type in eventnet — the
oldest and best-supported way of conditioning an endogenous effect in the
literature.

It conditions on the type of the **past** events only. Whether the coefficient
also differs by the type of the event being explained is a second, independent
axis (remstats `consider_type = "interact"`), which needs a risk set of
dyad × type and is not implemented here; fit separate models per outcome type
with [`fit_stratified`](@ref) instead.

# Example
```julia
using Revel
effects = split_by_type(Reciprocation, [:praise, :criticism])
name.(effects)     # ["reciprocity[types=praise]", "reciprocity[types=criticism]"]
h = build_history([Event(2, 1, 1.0; eventtype=:praise)])
[compute(e, h, 1, 2, 2.0) for e in effects]    # [1.0, 0.0]
```
"""
split_by_type(constructor, types; kwargs...) =
    [constructor(; types=ty, kwargs...) for ty in (types isa Symbol ? (types,) : types)]
