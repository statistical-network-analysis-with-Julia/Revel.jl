# =============================================================================
# Risk sets and designs
# =============================================================================
#
# One streaming pass over the event sequence yields, for every event, its risk
# set and the matrix of statistics on it, read off the strictly pre-event
# history. Everything downstream is a consumer of that pass: the design frame
# (`event_design`), the fitters that need a restricted risk set or a subset of
# cases, and every goodness-of-fit diagnostic. Only `compute`, `update_history!`
# and `InteractionHistory` — public API of Relevent.jl — are used, so the pass
# works for any statistic of the ecosystem.

const _DESIGN_TIES_SUPPORTED = (:error, :ordered, :breslow, :efron)
const _DESIGN_TIES_MODEL = "Revel's ordinal design (a likelihood over the ORDER of events)"
const _DESIGN_TIES_REASONS = Dict(
    :batch => "an ordinal likelihood has no exposure interval for a batch to " *
              "consume; holding the history fixed across the tied events IS the " *
              "Breslow correction, so pass `ties=:breslow` (or `:efron`) instead")

const _BOOKKEEPING = ("event_index", "sender", "receiver", "time", "is_event", "stratum",
                      "risk_set_size", "sampling_prob", "tie_weight")

"""
    RiskSetView

What [`each_risk_set`](@ref) hands its callback for one event:

- `index` — the event's position in the time-sorted sequence;
- `event` — the event itself;
- `dyads` — the risk set, a vector of `(sender, receiver)`;
- `X` — the `length(dyads) × p` matrix of statistics on it (a reused buffer:
  copy it to keep it beyond the callback);
- `case` — the row of the observed dyad;
- `tied` — under `ties=:efron`, the rows of every case tied with this one
  (empty otherwise);
- `tie_weight` — the Efron denominator weight `1 − (j−1)/d` those rows take
  (`1.0` otherwise).

# Example
```julia
using Revel
events = [Event(1, 2, 1.0), Event(2, 1, 2.0)]
sizes = Int[]
each_risk_set(events, [Inertia()], 3) do view
    push!(sizes, length(view.dyads))
end
sizes    # [6, 6] — three actors, six ordered dyads
```
"""
struct RiskSetView{T, M<:AbstractMatrix{Float64}}
    index::Int
    event::Event{T}
    dyads::Vector{Tuple{Int,Int}}
    X::M
    case::Int
    tied::Vector{Int}
    tie_weight::Float64
end

function _stat_names(statistics)
    names = String[name(s) for s in statistics]
    isempty(names) && throw(ArgumentError(
        "need at least one statistic (e.g. `[Inertia(), Reciprocation()]`)"))
    if !allunique(names)
        dup = first(n for n in names if count(==(n), names) > 1)
        throw(ArgumentError(
            "two statistics share the name $(repr(dup)). Give one of them a " *
            "`name=` so that coefficients and design columns can be told apart."))
    end
    clash = intersect(names, _BOOKKEEPING)
    isempty(clash) || throw(ArgumentError(
        "statistic name $(repr(first(clash))) collides with a bookkeeping column " *
        "of the design; rename it with `name=`"))
    return names
end

# Maximal runs of equal time in a time-sorted vector
function _tie_blocks(sorted::Vector{<:Event})
    blocks = UnitRange{Int}[]
    i = 1
    n = length(sorted)
    while i <= n
        j = i
        while j < n && sorted[j + 1].time == sorted[i].time
            j += 1
        end
        push!(blocks, i:j)
        i = j + 1
    end
    return blocks
end

_full_dyads(n::Int, directed::Bool) =
    directed ? [(s, r) for s in 1:n for r in 1:n if s != r] :
               [(s, r) for s in 1:n for r in (s + 1):n]

@inline _norm_dyad(s::Int, r::Int, directed::Bool) =
    directed || s <= r ? (s, r) : (r, s)

"""
    two_mode_dyads(senders, receivers) -> Vector{Tuple{Int,Int}}

The risk set of a two-mode event network: every `(sender, receiver)` pair with
the sender in one node set and the receiver in the other (users and articles,
actors and claims, authors and papers). Pass it as `riskset=` to
[`fit_revel`](@ref), [`event_design`](@ref) or [`each_risk_set`](@ref). The two
node sets must use disjoint actor IDs.

# Example
```julia
using Revel
dyads = two_mode_dyads(1:2, 11:13)
length(dyads)      # 6
first(dyads)       # (1, 11)
```
"""
function two_mode_dyads(senders, receivers)
    isempty(intersect(senders, receivers)) || throw(ArgumentError(
        "the two node sets of a two-mode risk set must use disjoint actor IDs"))
    return [(Int(s), Int(r)) for s in senders for r in receivers]
end

# The risk set of event `m`: a provider `(m, event) -> dyads` and, for a static
# risk set, the row lookup built once.
function _riskset_provider(riskset, n::Int, directed::Bool, sorted::Vector{<:Event})
    if riskset === :full
        dyads = _full_dyads(n, directed)
        return (m, ev) -> dyads, Dict(dy => k for (k, dy) in enumerate(dyads))
    elseif riskset === :active
        dyads = unique!([_norm_dyad(e.sender, e.receiver, directed) for e in sorted])
        return (m, ev) -> dyads, Dict(dy => k for (k, dy) in enumerate(dyads))
    elseif riskset === :sender || riskset === :receiver
        directed || throw(ArgumentError(
            "riskset=:$riskset (an actor-oriented choice set) needs directed events"))
        buf = Vector{Tuple{Int,Int}}(undef, n - 1)
        by_sender = riskset === :sender
        return function (m, ev)
            k = 0
            fixed = by_sender ? ev.sender : ev.receiver
            for a in 1:n
                a == fixed && continue
                k += 1
                buf[k] = by_sender ? (fixed, a) : (a, fixed)
            end
            return buf
        end, nothing
    elseif riskset isa AbstractVector
        dyads = [(Int(s), Int(r)) for (s, r) in riskset]
        allunique(dyads) || throw(ArgumentError("the risk set lists a dyad twice"))
        return (m, ev) -> dyads, Dict(dy => k for (k, dy) in enumerate(dyads))
    elseif riskset isa Function
        return (m, ev) -> riskset(m, ev)::Vector{Tuple{Int,Int}}, nothing
    end
    throw(ArgumentError(
        "riskset must be :full, :active, :sender, :receiver, a vector of " *
        "(sender, receiver) dyads, or a function (index, event) -> dyads; " *
        "got $(repr(riskset))"))
end

_row_of(lookup::Dict, dyads, dyad) = get(lookup, dyad, 0)
_row_of(::Nothing, dyads, dyad) = something(findfirst(==(dyad), dyads), 0)

function _case_mask(cases, sorted::Vector{<:Event})
    n = length(sorted)
    cases === nothing && return trues(n)
    if cases isa Function
        return BitVector(cases(e)::Bool for e in sorted)
    elseif cases isa AbstractVector{Bool}
        length(cases) == n || throw(ArgumentError(
            "`cases` has $(length(cases)) flags for $n events"))
        return BitVector(cases)
    elseif cases isa AbstractVector{<:Integer} || cases isa AbstractRange{<:Integer}
        mask = falses(n)
        for k in cases
            1 <= k <= n || throw(ArgumentError(
                "`cases` names event $k, but the sequence has $n events"))
            mask[k] = true
        end
        return mask
    end
    throw(ArgumentError(
        "`cases` must be `nothing`, a vector of indices into the time-sorted " *
        "events, a Bool mask, or a predicate `event -> Bool`"))
end

# Function barrier: `stats` is a tuple, so each compute call is statically
# dispatched
function _fill_rows!(X::Matrix{Float64}, stats::S, history, dyads, t) where S<:Tuple
    @inbounds for (d, (s, r)) in enumerate(dyads)
        vals = map(stat -> compute(stat, history, s, r, t), stats)
        for k in eachindex(vals)
            X[d, k] = vals[k]
        end
    end
    return X
end

function _reject_design_ties(sorted, blocks)
    tied = filter(b -> length(b) > 1, blocks)
    b = first(tied)
    throw(ArgumentError(
        "the event sequence contains tied timestamps: events $(first(b))–$(last(b)) " *
        "all occur at t = $(sorted[first(b)].time)" *
        (length(tied) > 1 ? "; $(length(tied)) timestamps carry ties in all" : "") *
        ". An ordinal likelihood is a likelihood over the ORDER of events, which a " *
        "tie leaves unobserved. Choose a policy: `ties=:efron` (the best " *
        "approximation), `ties=:breslow`, or `ties=:ordered` (sequence order, no " *
        "correction)."))
end

"""
    each_risk_set(f, events, statistics, n_actors; directed=true, riskset=:full,
                  ties=:error, cases=nothing) -> Symbol

Stream the risk sets of an event sequence: for each event (in time order) call
`f(view::RiskSetView)` with the risk set and the statistics of every dyad in
it, read off the history **before** the event. Returns the tie policy that
actually applied (`:none` when the data had no ties).

- `directed` — `false` treats events as undirected: the risk set holds unordered
  pairs `(i, j)` with `i < j`, and statistics should be built on symmetric
  layers.
- `riskset` — `:full` (every dyad among `1:n_actors`); `:active` (only dyads
  that occur somewhere in the sequence — remify's "active" risk set); `:sender`
  (the observed sender's `n−1` possible receivers: the receiver-choice step of
  an actor-oriented model, DyNAM-choice); `:receiver` (the observed receiver's
  possible senders); a vector of dyads (e.g. [`two_mode_dyads`](@ref)); or a
  function `(index, event) -> dyads` for a risk set that changes over time.
- `ties` — `:error` (default), `:ordered`, `:breslow` or `:efron`, the shared
  `Networks.TIE_POLICIES` vocabulary. Under the two corrections the history is
  frozen across a block of tied events.
- `cases` — which events are visited: `nothing` (all), indices into the
  time-sorted sequence, a `Bool` mask, or a predicate `event -> Bool`. Events
  that are not cases still enter the history, which is what stratified and
  moving-window fits need.

The pass costs `O(events × risk set × statistics)` and holds one design matrix
at a time.

# Example
```julia
using Revel
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0)]
each_risk_set(events, [Inertia(), Reciprocation()], 3) do view
    println(view.event, ": ", view.X[view.case, :])
end
# Event(1 → 2 @ 1.0): [0.0, 0.0]
# Event(2 → 1 @ 2.0): [0.0, 1.0]
# Event(1 → 2 @ 3.0): [1.0, 1.0]
```
"""
function each_risk_set(f::F, events::Vector{Event{T}}, statistics, n_actors::Int;
                       directed::Bool=true, riskset=:full, ties::Symbol=:error,
                       cases=nothing) where {F, T}
    check_tie_policy(ties, _DESIGN_TIES_SUPPORTED; model=_DESIGN_TIES_MODEL,
                     reasons=_DESIGN_TIES_REASONS)
    n_actors >= 2 || throw(ArgumentError("need at least two actors"))
    stats = Tuple(statistics)
    p = length(stats)
    p >= 1 || throw(ArgumentError("need at least one statistic"))

    sorted = sort(events; by=e -> e.time)
    blocks = _tie_blocks(sorted)
    has_ties = any(b -> length(b) > 1, blocks)
    ties === :error && has_ties && _reject_design_ties(sorted, blocks)
    freeze = ties === :breslow || ties === :efron
    is_case = _case_mask(cases, sorted)
    provider, lookup = _riskset_provider(riskset, n_actors, directed, sorted)

    history = InteractionHistory{T}()
    X = Matrix{Float64}(undef, 0, p)
    tied = Int[]
    no_tied = Int[]

    for block in blocks
        d = length(block)
        for (j, m) in enumerate(block)
            ev = sorted[m]
            if is_case[m]
                dyads = provider(m, ev)
                D = length(dyads)
                D >= 2 || throw(ArgumentError(
                    "the risk set of event $m ($ev) holds $D dyad$(D == 1 ? "" : "s"); " *
                    "a case needs at least one alternative"))
                size(X, 1) < D && (X = Matrix{Float64}(undef, D, p))
                _fill_rows!(X, stats, history, dyads, ev.time)
                case = _row_of(lookup, dyads, _norm_dyad(ev.sender, ev.receiver, directed))
                case > 0 || throw(ArgumentError(
                    "event $m ($ev) is not in its own risk set; declare the actor " *
                    "universe (`n_actors`) or the `riskset` so that every observed " *
                    "event is possible"))
                w = 1.0
                rows = no_tied
                if ties === :efron && d > 1
                    empty!(tied)
                    for k in block
                        e2 = sorted[k]
                        row = _row_of(lookup, dyads,
                                      _norm_dyad(e2.sender, e2.receiver, directed))
                        row > 0 || throw(ArgumentError(
                            "ties=:efron needs the tied events to share one risk " *
                            "set, but event $k is not in the risk set of event $m"))
                        push!(tied, row)
                    end
                    allunique(tied) || throw(ArgumentError(
                        "ties=:efron requires the events tied at one timestamp to " *
                        "be distinct dyads, but a dyad acts twice at t = $(ev.time). " *
                        "Use `ties=:breslow` or `ties=:ordered`."))
                    w = 1.0 - (j - 1) / d
                    rows = tied
                end
                f(RiskSetView(m, ev, dyads, view(X, 1:D, :), case, rows, w))
            end
            freeze || update_history!(history, ev)
        end
        if freeze
            for m in block
                update_history!(history, sorted[m])
            end
        end
    end
    return has_ties ? ties : :none
end

"""
    event_design(events, statistics, n_actors; directed=true, riskset=:full,
                 ties=:error, cases=nothing, n_controls=nothing,
                 rng=Random.default_rng()) -> DataFrame

The stratified design of a relational event model as a `DataFrame`: one row per
dyad in each event's risk set, with the statistics as columns (named by
`name(stat)`) and the bookkeeping columns `REM.fit_rem(::DataFrame, names)`
reads — `event_index`, `sender`, `receiver`, `time`, `is_event`, `stratum`,
`risk_set_size`, `sampling_prob`, `tie_weight`. The tie policy that applied
rides along as the `"tie_method"` metadata.

This is the frame to take to a different estimator (a GAM, a mixed model, a
penalised regression), to inspect for collinearity, or to fit with
`fit_rem(design, names)`. `directed`, `riskset`, `ties` and `cases` are as in
[`each_risk_set`](@ref).

`n_controls` keeps the case plus that many dyads drawn without replacement from
the rest of each risk set (nested case-control sampling; Vu, Pattison & Robins
2015; Lerner & Lomi 2020), with all randomness from `rng`. `nothing` keeps the
full risk set, which costs `events × risk set` rows.

# Example
```julia
using Revel
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0)]
design = event_design(events, [Inertia(), Reciprocation()], 3)
size(design, 1)                       # 18 — three events × six dyads
design[design.is_event, :reciprocity]  # [0.0, 1.0, 0.0]
```
"""
function event_design(events::Vector{Event{T}}, statistics, n_actors::Int;
                      directed::Bool=true, riskset=:full, ties::Symbol=:error,
                      cases=nothing, n_controls::Union{Nothing,Int}=nothing,
                      rng::AbstractRNG=Random.default_rng()) where T
    names = _stat_names(statistics)
    p = length(names)
    n_controls === nothing || n_controls >= 1 || throw(ArgumentError(
        "n_controls must be at least 1 (or `nothing` for the full risk set)"))

    event_index = Int[]; sender = Int[]; receiver = Int[]; time = T[]
    is_event = Bool[]; stratum = Int[]; rs_size = Int[]; prob = Float64[]
    tie_weight = Float64[]
    columns = [Float64[] for _ in 1:p]
    pool = Int[]
    n_strata = Ref(0)

    function push_row!(v, row, case_row, weight, sp)
        s, r = v.dyads[row]
        push!(event_index, v.index); push!(sender, s); push!(receiver, r)
        push!(time, v.event.time); push!(is_event, case_row)
        push!(stratum, n_strata[]); push!(rs_size, length(v.dyads))
        push!(prob, sp); push!(tie_weight, weight)
        for k in 1:p
            push!(columns[k], v.X[row, k])
        end
        return nothing
    end

    tie_applied = each_risk_set(events, statistics, n_actors; directed=directed,
                                riskset=riskset, ties=ties, cases=cases) do v
        n_strata[] += 1
        D = length(v.dyads)
        forced = isempty(v.tied) ? 1 : length(v.tied)
        eligible = D - forced
        keep = n_controls === nothing ? eligible : min(n_controls, eligible)
        sp = eligible == 0 ? 1.0 : keep / eligible

        push_row!(v, v.case, true, v.tie_weight, sp)
        # The other cases tied with this one stay in the Efron denominator
        for row in v.tied
            row == v.case || push_row!(v, row, false, v.tie_weight, sp)
        end
        if keep == eligible
            for row in 1:D
                (row == v.case || row in v.tied) && continue
                push_row!(v, row, false, 1.0, sp)
            end
        else
            empty!(pool)
            for row in 1:D
                (row == v.case || row in v.tied) && continue
                push!(pool, row)
            end
            shuffle!(rng, pool)
            for row in view(pool, 1:keep)
                push_row!(v, row, false, 1.0, sp)
            end
        end
    end

    df = DataFrame(event_index=event_index, sender=sender, receiver=receiver,
                   time=time, is_event=is_event, stratum=stratum,
                   risk_set_size=rs_size, sampling_prob=prob, tie_weight=tie_weight)
    for k in 1:p
        df[!, names[k]] = columns[k]
    end
    metadata!(df, "tie_method", string(tie_applied); style=:note)
    return df
end
