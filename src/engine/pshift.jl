# =============================================================================
# Participation shifts (Gibson 2003; Butts 2008)
# =============================================================================

# The 13 participation shifts, named as in R relevent's rem.dyad
# ("PSAB-BA" ↦ :AB_BA). Grouped as in Gibson (2003):
#   turn receiving:  AB-BA, AB-B0, AB-BY
#   turn claiming:   A0-X0, A0-XA, A0-XY
#   turn usurping:   AB-X0, AB-XA, AB-XB, AB-XY
#   turn continuing: A0-AY, AB-A0, AB-AY
const _PSHIFT_TYPES = (:AB_BA, :AB_B0, :AB_BY,
                       :A0_X0, :A0_XA, :A0_XY,
                       :AB_X0, :AB_XA, :AB_XB, :AB_XY,
                       :A0_AY, :AB_A0, :AB_AY)

"""
    pshift_types() -> NTuple{13, Symbol}

The 13 Gibson (2003) participation-shift types accepted by [`PShift`](@ref),
in relevent's grouping (turn receiving, claiming, usurping, continuing).

# Example
```julia
using Revel
length(pshift_types())      # 13
first(pshift_types())       # :AB_BA
```
"""
pshift_types() = _PSHIFT_TYPES

"""
    PShift(shift::Symbol; name="PS…") <: AbstractStatistic
    PShift(shift::AbstractString; name="PS…") <: AbstractStatistic

Participation-shift indicator (Gibson 2003), matching R relevent's
`PSAB-BA`-family effects: with previous event A→B, the statistic is 1 for
a candidate event that realizes the shift, 0 otherwise (and 0 for the
first event, which has no previous event).

`shift` is one of [`pshift_types`](@ref) (e.g. `:AB_BA`) or the R name
(e.g. `"PSAB-BA"`). In shift names, `A`/`B` are the previous event's
sender/receiver, `X`/`Y` are any *other* actors, and `0` is the null
actor: an event "to the group" is encoded with `receiver == 0`, so shifts
involving `0` (e.g. `:AB_B0`, `:A0_X0`) can only be nonzero when such
group-directed events occur in the data — with strictly dyadic events
they are structurally zero, exactly as in `relevent::rem.dyad`.

A participation shift is constant between events, so it is admissible in the
interval-timing model. For the shift in which the same dyad repeats, see
[`PShiftABAB`](@ref); for undirected events, [`UndirectedPShift`](@ref).

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0)])
compute(PShift(:AB_BA), h, 2, 1, 2.0)      # 1.0 — turn receiving: B answers A
compute(PShift("PSAB-XY"), h, 3, 4, 2.0)   # 1.0 — an outsider addresses another
compute(PShift(:AB_BA), h, 3, 1, 2.0)      # 0.0
name(PShift(:AB_BA))                       # "PSAB-BA"
```
"""
struct PShift <: AbstractStatistic
    shift::Symbol
    stat_name::String

    function PShift(shift::Symbol; name::String="")
        shift in _PSHIFT_TYPES ||
            throw(ArgumentError("unknown participation shift :$shift; " *
                                "valid shifts: $(join(_PSHIFT_TYPES, ", "))"))
        new(shift, isempty(name) ? "PS" * replace(String(shift), "_" => "-") : name)
    end
end

PShift(shift::AbstractString; kwargs...) =
    PShift(Symbol(replace(replace(String(shift), r"^PS" => ""), "-" => "_")); kwargs...)

name(stat::PShift) = stat.stat_name

# Indicator that the candidate event i→j realizes `shift` after the
# previous event a→b (b == 0 means the previous event was group-directed).
function _pshift_value(shift::Symbol, a::Int, b::Int, i::Int, j::Int)
    i == j && return 0.0
    if b == 0
        # Previous event was A→0 (to the group)
        shift === :A0_X0 && return (i != a && j == 0) ? 1.0 : 0.0
        shift === :A0_XA && return (i != a && j == a) ? 1.0 : 0.0
        shift === :A0_XY && return (i != a && j != a && j != 0) ? 1.0 : 0.0
        shift === :A0_AY && return (i == a && j != a && j != 0) ? 1.0 : 0.0
        return 0.0
    else
        # Previous event was dyadic A→B
        new_i = i != a && i != b
        new_j = j != a && j != b && j != 0
        shift === :AB_BA && return (i == b && j == a) ? 1.0 : 0.0
        shift === :AB_B0 && return (i == b && j == 0) ? 1.0 : 0.0
        shift === :AB_BY && return (i == b && new_j) ? 1.0 : 0.0
        shift === :AB_A0 && return (i == a && j == 0) ? 1.0 : 0.0
        shift === :AB_AY && return (i == a && new_j) ? 1.0 : 0.0
        shift === :AB_X0 && return (new_i && j == 0) ? 1.0 : 0.0
        shift === :AB_XA && return (new_i && j == a) ? 1.0 : 0.0
        shift === :AB_XB && return (new_i && j == b) ? 1.0 : 0.0
        shift === :AB_XY && return (new_i && new_j) ? 1.0 : 0.0
        return 0.0
    end
end

function compute(stat::PShift, history::InteractionHistory{T},
                 sender::Int, receiver::Int, current_time::T) where T
    isempty(history.events) && return 0.0
    prev = history.events[end]
    return _pshift_value(stat.shift, prev.sender, prev.receiver, sender, receiver)
end

function compute(stat::PShift, state::REM.EventNetworkState,
                 sender::Int, receiver::Int)
    isempty(state.event_history) && return 0.0
    a, b, _, _ = state.event_history[end]
    return _pshift_value(stat.shift, a, b, sender, receiver)
end
