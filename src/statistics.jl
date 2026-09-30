# =============================================================================
# The statistic protocol
# =============================================================================
#
# Every Revel statistic implements BOTH compute signatures of the ecosystem, on
# the shared `compute`/`name` generics (Networks.jl `src/statistics.jl`):
#
#   compute(stat, history::InteractionHistory, sender, receiver, time)
#       — Relevent.jl's full-risk-set estimators and Revel's own fitters;
#   compute(stat, state::REM.EventNetworkState, sender, receiver)
#       — REM.jl's interface, so the statistics work inside `REM.fit_rem`
#         (case-control sampling for large networks).
#
# Both reduce to one internal method, `_value(stat, events, sender, receiver, t)`,
# on the raw event vector, so the two interfaces cannot drift apart.

"""
    AbstractRevelStatistic <: REM.AbstractStatistic

Supertype of every statistic defined in Revel.jl. A subtype implements the
internal `_value(stat, events, sender, receiver, time)` and inherits both
`compute` signatures of the ecosystem — Relevent's
`compute(stat, history, sender, receiver, time)` and REM's
`compute(stat, state, sender, receiver)` — together with `name(stat)`.

# Example
```julia
using Revel, REM
Inertia() isa AbstractRevelStatistic          # true
SendEffect([1.0, 2.0]) isa AbstractStatistic  # true — usable in REM.fit_rem too
```
"""
abstract type AbstractRevelStatistic <: AbstractStatistic end

compute(stat::AbstractRevelStatistic, history::InteractionHistory, sender::Int,
        receiver::Int, current_time) =
    _value(stat, history.events, sender, receiver, _tfloat(current_time))

"""
    compute(stat::AbstractRevelStatistic, history::InteractionHistory, sender, receiver, time)
    compute(stat::AbstractRevelStatistic, state::REM.EventNetworkState, sender, receiver)

The value of a Revel statistic for the candidate event `sender → receiver`, read
off the events that happened before it. These are methods of the ecosystem's
shared `compute` generic (Networks.jl), in the two signatures its relational
event packages use: Relevent.jl's, on an `InteractionHistory` at an explicit
`time`, and REM.jl's, on an `EventNetworkState` at its `current_time`. The two
agree exactly.

The REM interface reads the state's event log, which carries no event types, so
a statistic restricted to `types=` works with the history interface only.

# Example
```julia
using Revel, REM
events = [Event(1, 2, 1.0), Event(1, 2, 2.0), Event(2, 1, 3.0)]
history = build_history(events)
compute(Inertia(), history, 1, 2, 4.0)          # 2.0
state = EventNetworkState{Float64}()
foreach(e -> update!(state, e), events)
state.current_time = 4.0
compute(Inertia(), state, 1, 2)                 # 2.0 — the same computation
```
"""
function compute(stat::AbstractRevelStatistic, state::REM.EventNetworkState,
                 sender::Int, receiver::Int)
    _uses_history(stat) && !state.keep_history && throw(ArgumentError(
        "$(name(stat)) reads the event log; construct the state with keep_history=true"))
    return _value(stat, state.event_history, sender, receiver,
                  _tfloat(state.current_time))
end

"""
    name(stat::AbstractRevelStatistic) -> String

The statistic's name: the label of its coefficient and of its column in an
[`event_design`](@ref). The automatic name spells out what distinguishes the
statistic from the default one — its scaling, transform, memory, event types —
and the `name=` keyword of every constructor overrides it. Two statistics in one
model must have different names.

# Example
```julia
using Revel
name(Inertia())                                       # "inertia"
name(Inertia(scaling=:prop, memory=HalfLife(30.0)))   # "inertia.prop[halflife=30.0]"
name(OTP(transform=:log1p, types=:email))             # "log1p(otp[types=email])"
name(Inertia(keep=(s, r, t, w, ty) -> w > 1, name="heavy_inertia"))   # "heavy_inertia"
```
"""
name(stat::AbstractRevelStatistic) = stat.label

# Does the statistic read the event history? (REM keeps its per-event log only
# when some statistic in the model says so.)
_uses_history(::AbstractRevelStatistic) = true
REM.needs_history(stat::AbstractRevelStatistic) = _uses_history(stat)

# Is the statistic constant between events at fixed history? The exact-time
# likelihood (`Relevent.fit_timing`) requires it; the default is `false`.
_interval_constant(::AbstractRevelStatistic) = false
Relevent.is_interval_constant(stat::AbstractRevelStatistic) = _interval_constant(stat)

Base.show(io::IO, stat::AbstractRevelStatistic) =
    print(io, nameof(typeof(stat)), "(", name(stat), ")")

# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------

_indicator(x) = x > 0 ? 1.0 : 0.0

const _TRANSFORMS = (identity=identity, none=identity, log1p=log1p, sqrt=sqrt,
                     indicator=_indicator, binary=_indicator)

_transform_fn(f::Function) = f
function _transform_fn(s::Symbol)
    haskey(_TRANSFORMS, s) || throw(ArgumentError(
        "unknown transform :$s; use one of $(join(keys(_TRANSFORMS), ", ")) or pass a function"))
    return _TRANSFORMS[s]
end

_transform_label(f) = f === identity ? "" : f === _indicator ? "indicator" :
                      string(nameof(f))

# "base" · ".prop" · "[layer]" wrapped in the transform's name
function _auto_name(base::AbstractString, suffix::AbstractString, transform)
    core = base * suffix
    t = _transform_label(transform)
    return isempty(t) ? core : "$t($core)"
end

_label(name::Nothing, auto::AbstractString) = String(auto)
_label(name, ::AbstractString) = String(name)

"""
    build_history(events) -> InteractionHistory

An `InteractionHistory` holding `events` (absorbed in the order given): the
history against which `compute(stat, history, sender, receiver, time)` evaluates
a statistic by hand.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0)])
compute(Inertia(), h, 1, 2, 4.0)         # 2.0 — two past 1 → 2 events
compute(Reciprocation(), h, 1, 2, 4.0)   # 1.0 — one past 2 → 1 event
```
"""
function build_history(events::AbstractVector{Event{T}}) where T
    h = InteractionHistory{T}()
    for e in events
        update_history!(h, e)
    end
    return h
end
