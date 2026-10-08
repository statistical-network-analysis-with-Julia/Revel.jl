# =============================================================================
# The interaction history
# =============================================================================
#
# The history interface of the statistic protocol is
# `compute(stat, history::InteractionHistory, sender, receiver, time)`: the
# full-risk-set likelihoods read every statistic off the history as it stood
# before an interval, and `each_risk_set` does the same for the design frame.
# Every Revel statistic reads only `history.events`; the layers index them
# lazily (src/layers.jl), so the history itself keeps nothing else.

"""
    InteractionHistory{T}()
    InteractionHistory()

The time-ordered events absorbed so far (`history.events::Vector{Event{T}}`),
against which `compute(stat, history, sender, receiver, time)` evaluates a
statistic. `T` is the type of the event times (`Float64` by default).
[`update_history!`](@ref) appends an event and [`build_history`](@ref) builds a
history from a vector of events.

The fitters replay one history in place (they empty `history.events` and absorb
the events again), and a statistic's layer notices the rewrite and rebuilds its
index; a history must otherwise only grow.

# Example
```julia
using Revel
h = InteractionHistory()
update_history!(h, Event(1, 2, 1.0))
update_history!(h, Event(2, 1, 2.0))
length(h.events)                         # 2
compute(Reciprocation(), h, 1, 2, 3.0)   # 1.0 — one past 2 → 1 event
```
"""
struct InteractionHistory{T}
    events::Vector{Event{T}}

    InteractionHistory{T}() where T = new{T}(Event{T}[])
end

InteractionHistory() = InteractionHistory{Float64}()

"""
    update_history!(history::InteractionHistory, event::Event) -> history

Append `event` to the history. Events must be absorbed in time order.

# Example
```julia
using Revel
h = InteractionHistory()
update_history!(h, Event(1, 2, 1.0))
compute(Inertia(), h, 1, 2, 2.0)         # 1.0
```
"""
function update_history!(history::InteractionHistory{T}, event::Event{T}) where T
    push!(history.events, event)
    return history
end

# Empty the history so that it can be replayed from the start (the streamed
# risk sets of the full-risk-set fitters)
function _reset_history!(history::InteractionHistory)
    empty!(history.events)
    return history
end
