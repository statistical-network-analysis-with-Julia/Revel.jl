# =============================================================================
# Interactions and statistic wrappers
# =============================================================================
#
# The review found eight ways in which the literature lets a covariate moderate
# an endogenous effect, three of them common, and stresses that they are NOT
# interchangeable:
#
#   product term     Interaction(a, b)              — this file
#   filtered         EventLayer(keep=…), MatchedDegree, matching_third
#   type split       EventLayer(types=…), split_by_type, BalanceEffect
#   attribute-weighted  TertiusEffect
#   stratified       fit_stratified                 — src/fit.jl
#   time-varying     fit_moving_window, GlobalEffect × effect
#   random slope / cross-level                       — not implemented
#
# The wrappers below work on ANY `AbstractStatistic` of the ecosystem — Revel's,
# Relevent's (`PShift`, `CovSnd`, …) and REM's — because they only call the
# shared `compute` generic.

_part_name(s::AbstractStatistic) = name(s)

"""
    Interaction(stats...; name=nothing)

The product of two or more statistics: the explicit product-term interaction
(remstats `a:b`; remulate `interact()`). The parts may be any statistics of the
ecosystem, so `Interaction(Inertia(), SendEffect(x))` moderates an endogenous
effect by a covariate and `Interaction(OutdegreeSender(), IndegreeReceiver())` is
an endogenous × endogenous term.

The literature's guidance (review §"Guidance is thin but convergent"): include
the main effects alongside the product, scale cumulative statistics before
interacting them (see [`Standardized`](@ref) and the `transform=:log1p`
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
# type: convert once here
function compute(stat::Interaction, history::InteractionHistory{T}, s::Int, r::Int,
                 t) where T
    tt = convert(T, t)::T
    return prod(map(p -> compute(p, history, s, r, tt), stat.parts))
end
compute(stat::Interaction, state::REM.EventNetworkState, s::Int, r::Int) =
    prod(map(p -> compute(p, state, s, r), stat.parts))

_uses_history(stat::Interaction) = any(REM.needs_history, stat.parts)
_interval_constant(stat::Interaction) = all(Relevent.is_interval_constant, stat.parts)

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
    Float64(stat.f(compute(stat.stat, history, s, r, convert(T, t)::T)))
compute(stat::Transformed, state::REM.EventNetworkState, s::Int, r::Int) =
    Float64(stat.f(compute(stat.stat, state, s, r)))

_uses_history(stat::Transformed) = REM.needs_history(stat.stat)
_interval_constant(stat::Transformed) = Relevent.is_interval_constant(stat.stat)

mutable struct _StdCache
    source::WeakRef
    len::Int
    t::Float64
    mean::Float64
    sd::Float64
end

"""
    Standardized(stat, n_actors; directed=true, corrected=false, name=nothing)

`stat` z-scored across the risk set at every event time: its value minus the
mean over all dyads among actors `1:n_actors`, divided by their standard
deviation. A statistic that is constant across the risk set standardises to `0`.

`corrected=false` divides by the population standard deviation (denominator
`D`, the number of dyads); `corrected=true` by the sample one (denominator
`D − 1`), which is what remstats' `scaling = "std"` computes. The two differ by
the constant factor `sqrt((D − 1)/D)`, so they rescale the coefficient and
nothing else.

Cumulative statistics grow without bound over an event sequence, which makes
their coefficients hard to compare across effects and can set off a feedback
loop in simulation ("process explosion"); per-time-point standardisation is one
of the two remedies in the literature (Vieira, Leenders & Mulder), the other
being a `log1p` transform.

`directed=false` standardises over unordered pairs. The mean and standard
deviation are computed once per event time, so the wrapper costs one extra pass
over the risk set.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 2, 2.0)])
z = Standardized(Inertia(), 3)
compute(z, h, 1, 2, 3.0)    # ≈ 2.24 — the only dyad with a history
compute(z, h, 2, 3, 3.0)    # ≈ -0.45
```
"""
struct Standardized{S<:AbstractStatistic} <: AbstractRevelStatistic
    stat::S
    n_actors::Int
    directed::Bool
    corrected::Bool
    cache::_StdCache
    label::String
end

function Standardized(stat::AbstractStatistic, n_actors::Int; directed::Bool=true,
                      corrected::Bool=false, name=nothing)
    n_actors >= (corrected && !directed ? 3 : 2) || throw(ArgumentError(
        "Standardized needs at least two dyads in the risk set"))
    return Standardized{typeof(stat)}(stat, n_actors, directed, corrected,
                                      _StdCache(WeakRef(nothing), -1, NaN, 0.0, 1.0),
                                      _label(name, "std(" * Revel.name(stat) * ")"))
end

# Mean and standard deviation of `value(i, j)` over the risk set
function _risk_set_moments(value, n::Int, directed::Bool, corrected::Bool)
    total = 0.0; count = 0
    for i in 1:n, j in (directed ? (1:n) : ((i + 1):n))
        i == j && continue
        total += value(i, j); count += 1
    end
    μ = total / count
    # a second pass on the deviations: no cancellation for a near-constant statistic
    ss = 0.0
    for i in 1:n, j in (directed ? (1:n) : ((i + 1):n))
        i == j && continue
        ss += (value(i, j) - μ)^2
    end
    return μ, sqrt(ss / (corrected ? count - 1 : count))
end

function _standardize(stat::Standardized, source, len::Int, t::Float64, value, s::Int,
                      r::Int)
    c = stat.cache
    if c.source.value !== source || c.len != len || c.t != t
        μ, σ = _risk_set_moments(value, stat.n_actors, stat.directed, stat.corrected)
        c.source = WeakRef(source); c.len = len; c.t = t; c.mean = μ; c.sd = σ
    end
    return c.sd > 0 ? (value(s, r) - c.mean) / c.sd : 0.0
end

function compute(stat::Standardized, history::InteractionHistory{T}, s::Int, r::Int,
                 t) where T
    tt = convert(T, t)::T
    return _standardize(stat, history.events, length(history.events), _tfloat(tt),
                        (i, j) -> compute(stat.stat, history, i, j, tt), s, r)
end
compute(stat::Standardized, state::REM.EventNetworkState, s::Int, r::Int) =
    _standardize(stat, state, state.n_events, _tfloat(state.current_time),
                 (i, j) -> compute(stat.stat, state, i, j), s, r)

_uses_history(stat::Standardized) = REM.needs_history(stat.stat)
_interval_constant(stat::Standardized) = Relevent.is_interval_constant(stat.stat)

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
    [constructor(; types=ty, kwargs...) for ty in types]
