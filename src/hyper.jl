# =============================================================================
# Relational hyperevents
# =============================================================================
#
# A hyperevent is an interaction among a SET of actors — a meeting, a coauthored
# paper, a co-offence (undirected), or an e-mail to several recipients (directed:
# a sender set and a receiver set). Relational hyperevent models (RHEM; Lerner,
# Tranmer, Mowbray & Hâncean 2019; Lerner, Lomi, Mowbray, Rollings & Tranmer
# 2021; Lerner & Lomi 2023) put the event rate on the hyperedge instead of on
# the dyad, and replace dyadic statistics by hyperedge statistics built in two
# steps: aggregate past events into a hyperedge
# attribute — *activity* (events on exactly h) or *degree* (events on any
# superset of h) — then aggregate that attribute over the sub-hyperedges of a
# given order. The vocabulary is an algebra rather than a list:
#
#   {activity | degree | sender-specific degree | outcome-weighted degree}
#     × {subset order p or (p, q)} × {mean | sum | min | max | sd | absdiff}
#     × {memory kernel}
#
# and this file exposes those axes; the papers' named effects are thin
# constructors.
#
# The memory model is kept general by storing, for every indexed sub-hyperedge,
# the list of past events that contain it and evaluating Σ kernel(t − tₑ)·wₑ on
# read (walking backwards and stopping at the kernel's support). A subset order
# is indexed only once a statistic has asked for it.
#
# Like the rest of Revel this file hosts no optimizer and no likelihood kernel:
# `fit_rhem` builds the size-stratified case-control design and hands it to
# `REM.fit_rem(::DataFrame, names)`.

import Networks: is_directed

# -----------------------------------------------------------------------------
# The data structure
# -----------------------------------------------------------------------------

# A sorted copy of an actor collection, validated
function _actor_set(actors, what::AbstractString)
    set = sort!(vec(collect(Int, actors)))
    isempty(set) || first(set) >= 1 || throw(ArgumentError(
        "$what must be positive actor IDs, got $(first(set))"))
    allunique(set) || throw(ArgumentError(
        "$what list an actor twice: $(set). A hyperedge is a set of actors."))
    return set
end

"""
    HyperEvent(participants, time; eventtype=:event, weight=1.0)
    HyperEvent(senders, receivers, time; eventtype=:event, weight=1.0)

A relational hyperevent: an interaction among a set of actors at `time`.

The two-argument form is **undirected** — a meeting, a coauthored paper, a
co-offence: all participants are stored in `senders` and `receivers` is empty.
The three-argument form is **directed** — typically one sender and a set of
receivers (an e-mail, the multicast events of Perry & Wolfe 2013 and Lerner &
Lomi 2023), though several senders are allowed (Lerner et al. 2019).

Actor sets are stored sorted; an actor listed twice, a non-positive ID, a
directed event without senders, or an actor that is both sender and receiver
(the models are defined for loopless hypergraphs) is an `ArgumentError`.
`weight` carries an event weight or an outcome (team performance, citations)
that `weighted=true` statistics read; `eventtype` is what the `types=` filter of
a statistic matches.

# Example
```julia
using Revel
meeting = HyperEvent([3, 1, 2], 1.0)
meeting.senders, meeting.receivers        # ([1, 2, 3], Int64[])
mail = HyperEvent([1], [4, 2], 2.0; eventtype=:email)
is_directed(mail), participants(mail)     # (true, [1, 2, 4])
```
"""
struct HyperEvent{T}
    senders::Vector{Int}
    receivers::Vector{Int}
    time::T
    eventtype::Symbol
    weight::Float64
    function HyperEvent{T}(senders, receivers, time, eventtype::Symbol,
                           weight::Real) where T
        S = _actor_set(senders, "the senders of a HyperEvent")
        R = _actor_set(receivers, "the receivers of a HyperEvent")
        isempty(S) && throw(ArgumentError(
            isempty(R) ? "a HyperEvent needs at least one participant" :
            "a directed HyperEvent needs at least one sender"))
        both = intersect(S, R)
        isempty(both) || throw(ArgumentError(
            "actor$(length(both) == 1 ? "" : "s") $(join(both, ", ")) both send " *
            "and receive this hyperevent. Hyperevent statistics are defined for " *
            "loopless hypergraphs: drop the sender from its own receiver set."))
        isfinite(weight) || throw(ArgumentError(
            "the weight of a HyperEvent must be finite, got $weight"))
        return new{T}(S, R, time, eventtype, Float64(weight))
    end
end

HyperEvent(senders, receivers, time::T; eventtype::Symbol=:event,
           weight::Real=1.0) where T =
    HyperEvent{T}(senders, receivers, time, eventtype, weight)
HyperEvent(participants, time::T; eventtype::Symbol=:event, weight::Real=1.0) where T =
    HyperEvent{T}(participants, Int[], time, eventtype, weight)

Base.:(==)(a::HyperEvent, b::HyperEvent) =
    a.senders == b.senders && a.receivers == b.receivers && a.time == b.time &&
    a.eventtype === b.eventtype && a.weight == b.weight
Base.hash(e::HyperEvent, h::UInt) =
    hash(e.senders, hash(e.receivers, hash(e.time, hash(e.eventtype, hash(e.weight, h)))))

"""
    is_directed(event::HyperEvent) -> Bool

Whether a [`HyperEvent`](@ref) distinguishes senders from receivers. (A method
of the generic the ecosystem shares through Networks.jl.)

# Example
```julia
using Revel
is_directed(HyperEvent([1, 2, 3], 1.0))      # false — a meeting
is_directed(HyperEvent([1], [2, 3], 1.0))    # true — one sender, two receivers
```
"""
is_directed(e::HyperEvent) = !isempty(e.receivers)

"""
    participants(event::HyperEvent) -> Vector{Int}

Every actor taking part in a [`HyperEvent`](@ref), sorted: the senders and the
receivers of a directed hyperevent, the participant set of an undirected one.

# Example
```julia
using Revel
participants(HyperEvent([4], [2, 1], 1.0))    # [1, 2, 4]
participants(HyperEvent([3, 1], 1.0))         # [1, 3]
```
"""
participants(e::HyperEvent) =
    isempty(e.receivers) ? copy(e.senders) : sort!(vcat(e.senders, e.receivers))

_set_string(set) = "{" * join(set, ", ") * "}"

Base.show(io::IO, e::HyperEvent) =
    print(io, "HyperEvent(", _set_string(e.senders),
          isempty(e.receivers) ? "" : " → " * _set_string(e.receivers),
          " @ ", e.time, ")")

# What an index holds: the events containing a sub-hyperedge, by position
const _HyperIndex = Dict{Vector{Int}, Vector{Int}}

"""
    HyperHistory{T}()

The past of a hyperevent sequence: the absorbed [`HyperEvent`](@ref)s in time
order, plus the indices the statistics read. It is what
`compute(stat, history, senders, receivers, time)` evaluates a hyperevent
statistic against; [`build_hyper_history`](@ref) makes one from a vector of
events and [`update_hyper_history!`](@ref) appends to it.

For each sub-hyperedge an index lists the past events that contain it, so a
statistic reads `Σ kernel_weight(memory, t − tₑ)·wₑ` under *any* memory kernel
and any event-type filter. A subset order is indexed the first time a statistic
asks for it and kept up to date from then on; indexing order `p` costs
`binomial(size, p)` entries per event, which is the cost of high-order subset
repetition on large hyperedges.

A history holds either undirected or directed hyperevents, not both. It carries
scratch buffers: do not evaluate statistics on one history from several tasks.

# Example
```julia
using Revel
h = HyperHistory{Float64}()
update_hyper_history!(h, HyperEvent([1, 2, 3], 1.0))
length(h)                                            # 1
compute(SubsetRepetition(2), h, [1, 2], Int[], 2.0)  # 1.0
```
"""
mutable struct HyperHistory{T}
    events::Vector{HyperEvent{T}}
    times::Vector{Float64}
    weights::Vector{Float64}
    types::Vector{Symbol}
    directed::Union{Nothing, Bool}
    # (:exact, 0, 0)  events on exactly this (senders, receivers) hyperedge
    # (:union, 0, 0)  events on exactly this participant set
    # (:dir, p, q)    events containing p given senders and q given receivers
    # (:und, p, 0)    events whose participant set contains p given actors
    indices::Dict{Tuple{Symbol,Int,Int}, _HyperIndex}
    neighbours::Union{Nothing, Dict{Int, Set{Int}}}
    # scratch for reads …
    key::Vector{Int}
    pair::Vector{Int}
    idx1::Vector{Int}
    idx2::Vector{Int}
    members::Vector{Int}
    others::Vector{Int}
    vals::Vector{Float64}
    overlap::Dict{Int, Int}
    # … and for index maintenance, which may run in the middle of a read
    bkey::Vector{Int}
    bidx1::Vector{Int}
    bidx2::Vector{Int}
    bmembers::Vector{Int}
end

HyperHistory{T}() where T = HyperHistory{T}(
    HyperEvent{T}[], Float64[], Float64[], Symbol[], nothing,
    Dict{Tuple{Symbol,Int,Int}, _HyperIndex}(), nothing,
    Int[], [0, 0], Int[], Int[], Int[], Int[], Float64[], Dict{Int,Int}(),
    Int[], Int[], Int[], Int[])
HyperHistory() = HyperHistory{Float64}()

Base.length(h::HyperHistory) = length(h.events)

Base.show(io::IO, h::HyperHistory) =
    print(io, "HyperHistory(", length(h), " ",
          h.directed === nothing ? "" : h.directed ? "directed " : "undirected ",
          "hyperevent", length(h) == 1 ? "" : "s", ")")

# The participant set of (S, R), sorted; `S` itself when there are no receivers
function _members!(buf::Vector{Int}, S::AbstractVector{Int}, R::AbstractVector{Int})
    isempty(R) && return S
    empty!(buf)
    i = 1; j = 1
    @inbounds while i <= length(S) || j <= length(R)
        if j > length(R) || (i <= length(S) && S[i] <= R[j])
            (isempty(buf) || buf[end] != S[i]) && push!(buf, S[i])
            i += 1
        else
            (isempty(buf) || buf[end] != R[j]) && push!(buf, R[j])
            j += 1
        end
    end
    return buf
end

# Call `f()` once per `p`-subset of the sorted `set`, with the subset written to
# `key[offset+1 : offset+p]` (lexicographic order; `p = 0` is the empty subset)
function _each_subset(f::F, set::AbstractVector{Int}, p::Int, idx::Vector{Int},
                      key::Vector{Int}, offset::Int) where F
    n = length(set)
    p > n && return nothing
    p == 0 && (f(); return nothing)
    resize!(idx, p)
    @inbounds for k in 1:p
        idx[k] = k
    end
    @inbounds while true
        for k in 1:p
            key[offset + k] = set[idx[k]]
        end
        f()
        i = p
        while i >= 1 && idx[i] == n - p + i
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for k in (i + 1):p
            idx[k] = idx[k - 1] + 1
        end
    end
    return nothing
end

function _index_push!(index::_HyperIndex, key::Vector{Int}, e::Int)
    list = get(index, key, nothing)
    if list === nothing
        index[copy(key)] = [e]
    else
        push!(list, e)
    end
    return nothing
end

# Enter event number `e` into one index
function _index_event!(h::HyperHistory, index::_HyperIndex, spec::Tuple{Symbol,Int,Int},
                       ev::HyperEvent, e::Int)
    kind, p, q = spec
    key = h.bkey
    if kind === :exact
        empty!(key)
        append!(key, ev.senders); push!(key, 0); append!(key, ev.receivers)
        _index_push!(index, key, e)
    elseif kind === :union
        P = _members!(h.bmembers, ev.senders, ev.receivers)
        empty!(key); append!(key, P)
        _index_push!(index, key, e)
    elseif kind === :und
        P = _members!(h.bmembers, ev.senders, ev.receivers)
        resize!(key, p)
        _each_subset(P, p, h.bidx1, key, 0) do
            _index_push!(index, key, e)
        end
    else  # :dir
        resize!(key, p + q)
        _each_subset(ev.senders, p, h.bidx1, key, 0) do
            _each_subset(ev.receivers, q, h.bidx2, key, p) do
                _index_push!(index, key, e)
            end
        end
    end
    return nothing
end

# The index for `spec`, built from the whole past the first time it is asked for
function _index!(h::HyperHistory, spec::Tuple{Symbol,Int,Int})
    index = get(h.indices, spec, nothing)
    index === nothing || return index
    index = _HyperIndex()
    for (e, ev) in enumerate(h.events)
        _index_event!(h, index, spec, ev, e)
    end
    h.indices[spec] = index
    return index
end

# The dyadic projection's adjacency: for directed hyperevents every
# (sender, receiver) pair, for undirected ones every pair of participants
function _link_event!(nb::Dict{Int, Set{Int}}, ev::HyperEvent)
    if isempty(ev.receivers)
        for i in ev.senders, j in ev.senders
            i == j || push!(get!(() -> Set{Int}(), nb, i), j)
        end
    else
        for i in ev.senders, j in ev.receivers
            push!(get!(() -> Set{Int}(), nb, i), j)
            push!(get!(() -> Set{Int}(), nb, j), i)
        end
    end
    return nothing
end

function _neighbours!(h::HyperHistory)
    nb = h.neighbours
    nb === nothing || return nb
    nb = Dict{Int, Set{Int}}()
    for ev in h.events
        _link_event!(nb, ev)
    end
    h.neighbours = nb
    return nb
end

"""
    update_hyper_history!(history::HyperHistory, event::HyperEvent) -> history

Append `event` to the history and bring every index already built up to date.
Events must arrive in time order (an earlier time than the last absorbed event
is an `ArgumentError`), and a history cannot mix directed and undirected
hyperevents.

# Example
```julia
using Revel
h = HyperHistory{Float64}()
update_hyper_history!(h, HyperEvent([1, 2], 1.0))
update_hyper_history!(h, HyperEvent([1, 2, 3], 2.0))
compute(ExactRepetition(), h, [1, 2], Int[], 3.0)     # 1.0
compute(SubsetRepetition(2), h, [1, 2], Int[], 3.0)   # 2.0
```
"""
function update_hyper_history!(h::HyperHistory{T}, ev::HyperEvent{T}) where T
    t = _tfloat(ev.time)
    isempty(h.times) || t >= h.times[end] || throw(ArgumentError(
        "hyperevents must be absorbed in time order: got t = $t after " *
        "t = $(h.times[end]). Sort the events (build_hyper_history does)."))
    dir = is_directed(ev)
    h.directed === nothing || h.directed == dir || throw(ArgumentError(
        "a HyperHistory holds either directed or undirected hyperevents, but " *
        "$ev is $(dir ? "directed" : "undirected") and the history is not. " *
        "Model the two kinds of event in separate sequences."))
    h.directed = dir
    push!(h.events, ev); push!(h.times, t); push!(h.weights, ev.weight)
    push!(h.types, ev.eventtype)
    e = length(h.events)
    for (spec, index) in h.indices
        _index_event!(h, index, spec, ev, e)
    end
    h.neighbours === nothing || _link_event!(h.neighbours, ev)
    return h
end

"""
    build_hyper_history(events) -> HyperHistory

A [`HyperHistory`](@ref) holding `events`, absorbed in time order (the vector is
sorted by time first; events at equal times keep their order). It is the history
against which `compute(stat, history, senders, receivers, time)` evaluates a
hyperevent statistic by hand.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1, 2, 3], 1.0), HyperEvent([1, 2], 2.0)])
compute(SubsetRepetition(1), h, [1, 4], Int[], 3.0)    # (2 + 0)/2 = 1.0
```
"""
function build_hyper_history(events::AbstractVector{HyperEvent{T}}) where T
    h = HyperHistory{T}()
    for ev in sort(events; by=e -> e.time)
        update_hyper_history!(h, ev)
    end
    return h
end

# -----------------------------------------------------------------------------
# The statistic protocol
# -----------------------------------------------------------------------------

"""
    AbstractHyperStatistic <: REM.AbstractStatistic

Supertype of the hyperevent statistics. A subtype adds a method to the shared
`compute` generic with the hyperedge signature

    compute(stat, history::HyperHistory, senders, receivers, time)

where `senders` and `receivers` are the sorted actor sets of the **candidate**
hyperedge (`receivers` is empty for an undirected candidate) and the statistic
is read off the events in `history` that are not later than `time`.
`compute(stat, history, candidate::HyperEvent)` is the same call with the
candidate's sets and time. `name(stat)` is the column and coefficient name.

[`Interaction`](@ref) and [`Transformed`](@ref) accept hyperevent statistics, so
a product term or a rescaling is built exactly as for dyadic statistics.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1], [2, 3], 1.0)])
stat = ReceiverSetRepetition(2)
stat isa AbstractHyperStatistic                           # true
compute(stat, h, [4], [2, 3], 2.0)                        # 1.0
compute(stat, h, HyperEvent([4], [2, 3], 2.0))            # 1.0 — the same call
name(Interaction(HyperedgeSize(), stat))                  # "size:rec.subrep(2)"
```
"""
abstract type AbstractHyperStatistic <: AbstractStatistic end

name(stat::AbstractHyperStatistic) = stat.label

Base.show(io::IO, stat::AbstractHyperStatistic) =
    print(io, nameof(typeof(stat)), "(", name(stat), ")")

function _check_actor_set(set::AbstractVector{Int}, what::AbstractString)
    @inbounds for k in eachindex(set)
        ok = k == firstindex(set) ? set[k] >= 1 : set[k] > set[k - 1]
        ok || throw(ArgumentError(
            "the $what of a candidate hyperedge must be sorted, distinct, positive " *
            "actor IDs; got $(collect(set))"))
    end
    return nothing
end

function _check_candidate(S::AbstractVector{Int}, R::AbstractVector{Int})
    isempty(S) && throw(ArgumentError(
        "a candidate hyperedge needs at least one sender (an undirected " *
        "candidate passes its participants as `senders` and an empty `receivers`)"))
    _check_actor_set(S, "senders")
    _check_actor_set(R, "receivers")
    i = 1; j = 1
    @inbounds while i <= length(S) && j <= length(R)
        S[i] == R[j] && throw(ArgumentError(
            "actor $(S[i]) is both sender and receiver of the candidate " *
            "hyperedge; hyperevent statistics are defined for loopless hyperedges"))
        S[i] < R[j] ? (i += 1) : (j += 1)
    end
    return nothing
end

function compute(stat::AbstractHyperStatistic, history::HyperHistory,
                 senders::AbstractVector{Int}, receivers::AbstractVector{Int}, time)
    _check_candidate(senders, receivers)
    return Float64(stat.transform(
        _hvalue(stat, history, senders, receivers, _tfloat(time))))
end

compute(stat::AbstractStatistic, history::HyperHistory, candidate::HyperEvent) =
    compute(stat, history, candidate.senders, candidate.receivers, candidate.time)

# Product terms and rescalings of hyperevent statistics
compute(stat::Interaction, history::HyperHistory, senders::AbstractVector{Int},
        receivers::AbstractVector{Int}, time) =
    prod(map(p -> compute(p, history, senders, receivers, time), stat.parts))
compute(stat::Transformed, history::HyperHistory, senders::AbstractVector{Int},
        receivers::AbstractVector{Int}, time) =
    Float64(stat.f(compute(stat.stat, history, senders, receivers, time)))

_is_hyper(::AbstractHyperStatistic) = true
_is_hyper(stat::Interaction) = all(_is_hyper, stat.parts)
_is_hyper(stat::Transformed) = _is_hyper(stat.stat)
_is_hyper(::Any) = false

# A hyperevent statistic has no value on a dyad: say so instead of a MethodError
_not_dyadic(stat) = throw(ArgumentError(
    "$(name(stat)) is a hyperevent statistic: it is evaluated on a candidate " *
    "hyperedge against a HyperHistory. Fit it with `fit_rhem` (or build the " *
    "design with `hyper_design`), not with a dyadic fitter."))
compute(stat::AbstractHyperStatistic, ::InteractionHistory, ::Int, ::Int, time) =
    _not_dyadic(stat)
compute(stat::AbstractHyperStatistic, ::REM.EventNetworkState, ::Int, ::Int) =
    _not_dyadic(stat)

# How a statistic reads the past: the memory kernel, the event-type filter and
# whether event weights count
struct _HyperRead{M<:AbstractMemory}
    memory::M
    types::Union{Nothing, Vector{Symbol}}
    weighted::Bool
end

_hyper_read(memory::AbstractMemory, types, weighted::Bool) =
    _HyperRead{typeof(memory)}(memory, _types_arg(types), weighted)
_hyper_read(memory, types, weighted) = throw(ArgumentError(
    "`memory` must be a memory kernel (FullMemory(), HalfLife(h), Window(w), …), " *
    "got $(repr(memory))"))

function _suffix(rd::_HyperRead)
    parts = String[]
    m = _memory_label(rd.memory)
    isempty(m) || push!(parts, m)
    rd.types === nothing || push!(parts, "types=" * join(rd.types, "+"))
    rd.weighted && push!(parts, "weighted")
    return isempty(parts) ? "" : "[" * join(parts, ",") * "]"
end

# The weight past event `e` carries at age `age` (0 when the type filter drops it)
@inline function _event_weight(h::HyperHistory, rd::_HyperRead, e::Int, age::Float64)
    rd.types === nothing || (@inbounds h.types[e]) in rd.types || return 0.0
    w = kernel_weight(rd.memory, age)
    return rd.weighted ? w * @inbounds(h.weights[e]) : w
end

# Σ kernel(t − tₑ)·wₑ over the listed events. The list is in time order, so the
# walk runs backwards and stops at the kernel's support; events later than `t`
# have not happened yet.
function _read_list(h::HyperHistory, list::Vector{Int}, rd::_HyperRead, t::Float64)
    if rd.memory isa FullMemory && rd.types === nothing && !rd.weighted
        # every listed event up to `t` counts 1: bisect for how many there are,
        # so a read costs O(log events) however long the list
        lo, hi = 0, length(list)
        @inbounds while lo < hi
            mid = (lo + hi + 1) >>> 1
            h.times[list[mid]] <= t ? (lo = mid) : (hi = mid - 1)
        end
        return Float64(lo)
    end
    total = 0.0
    sup = _support(rd.memory)
    @inbounds for k in length(list):-1:1
        e = list[k]
        age = t - h.times[e]
        age < 0 && continue
        age > sup && break
        total += _event_weight(h, rd, e, age)
    end
    return total
end

@inline function _degree(h::HyperHistory, index::_HyperIndex, key::Vector{Int},
                         rd::_HyperRead, t::Float64)
    list = get(index, key, nothing)
    return list === nothing ? 0.0 : _read_list(h, list, rd, t)
end

# -----------------------------------------------------------------------------
# Aggregation over sub-hyperedges, pairs or participants
# -----------------------------------------------------------------------------

function _pairwise(v::Vector{Float64}, differ::Bool)
    total = 0.0
    n = length(v)
    @inbounds for a in 1:n, b in (a + 1):n
        total += differ ? Float64(v[a] != v[b]) : abs(v[a] - v[b])
    end
    return total
end

# An empty collection (a candidate too small to have a sub-hyperedge of the
# order asked for) aggregates to 0 under every rule
function _aggregate(v::Vector{Float64}, how::Symbol)
    n = length(v)
    n == 0 && return 0.0
    how === :mean && return sum(v) / n
    how === :sum && return sum(v)
    how === :min && return minimum(v)
    how === :max && return maximum(v)
    if how === :sd || how === :samplesd
        n == 1 && return 0.0
        μ = sum(v) / n
        ss = 0.0
        for x in v
            ss += (x - μ)^2
        end
        return sqrt(ss / (how === :sd ? n : n - 1))
    end
    if how === :homogeneity
        # Lerner et al. (2021, pp. 228–229), for a binary covariate: the larger
        # group minus the smaller, over the size; for an odd size rescaled so
        # that the most even split is 0 and a single group is 1
        k = count(==(1.0), v)
        d = abs(n - 2k) / n
        return isodd(n) ? (n == 1 ? 1.0 : (d - 1 / n) / (1 - 1 / n)) : d
    end
    n == 1 && return 0.0
    pairs = n * (n - 1) / 2
    how === :absdiff && return _pairwise(v, false) / pairs
    how === :catdiff && return _pairwise(v, true) / pairs
    return -_pairwise(v, false)             # :assortativity
end

const _SUBSET_AGGREGATES = (:mean, :sum, :min, :max, :sd, :samplesd, :absdiff,
                            :assortativity)

function _check_aggregate(aggregate::Symbol, allowed, what::AbstractString)
    aggregate in (:gw, :gwsr, :geometric) && throw(ArgumentError(
        "$what: geometrically weighted subset repetition (Fabbrucci Barbagli, " *
        "Lerner, Amati & De Stefano 2026; eventnet `*_GW_SUB_REP_STAT`) is not " *
        "implemented. It is not an aggregation rule but a different weighting of " *
        "past events by overlap size; use low subset orders, or `aggregate=:mean`."))
    aggregate in allowed || throw(ArgumentError(
        "$what: aggregate must be one of $(allowed), got :$aggregate"))
    return aggregate
end

_agg_label(aggregate::Symbol) = aggregate === :mean ? "" : ".$(aggregate)"

# -----------------------------------------------------------------------------
# Hyperedge size
# -----------------------------------------------------------------------------

const _ENDPOINTS = (:all, :senders, :receivers)

function _check_endpoint(endpoint::Symbol, what::AbstractString)
    endpoint in _ENDPOINTS || throw(ArgumentError(
        "$what: endpoint must be :all, :senders or :receivers, got :$endpoint"))
    return endpoint
end

"""
    HyperedgeSize(; endpoint=:all, transform=identity, name=nothing)

The number of actors in the candidate hyperedge (Lerner, Tranmer, Mowbray &
Hâncean 2019, who call it "one of the few hyperedge statistics that have no
related statistic in dyadic REM"; eventnet `UHE_SIZE_STAT`/`DHE_SIZE_STAT`).
`endpoint=:senders` counts the senders, `:receivers` the receivers, `:all` both.

**It cannot be estimated on its own here.** [`hyper_design`](@ref) and
[`fit_rhem`](@ref) compare each observed hyperevent with alternatives of the
*same* size (the size-stratified risk set of Lerner & Lomi 2023), so a size-only
statistic is constant within every stratum and its coefficient is not
identified — the fit would report a singular information matrix. Use it as a
moderator, in an [`Interaction`](@ref) with a statistic that does vary within
the stratum (does familiarity matter less in large meetings?).

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1, 2, 3], 1.0)])
compute(HyperedgeSize(), h, [1, 2, 3], Int[], 2.0)                        # 3.0
compute(HyperedgeSize(endpoint=:receivers), h, [1], [2, 3, 4], 2.0)       # 3.0
by_size = Interaction(HyperedgeSize(), SubsetRepetition(2))
compute(by_size, h, [1, 2, 3], Int[], 2.0)                                # 3 × 1 = 3.0
```
"""
struct HyperedgeSize{F} <: AbstractHyperStatistic
    endpoint::Symbol
    transform::F
    label::String
end

function HyperedgeSize(; endpoint::Symbol=:all, transform=identity, name=nothing)
    _check_endpoint(endpoint, "HyperedgeSize")
    f = _transform_fn(transform)
    base = endpoint === :all ? "size" : "size.$(endpoint)"
    return HyperedgeSize{typeof(f)}(endpoint, f, _label(name, _auto_name(base, "", f)))
end

_hvalue(stat::HyperedgeSize, ::HyperHistory, S, R, ::Float64) =
    Float64(stat.endpoint === :senders ? length(S) :
            stat.endpoint === :receivers ? length(R) : length(S) + length(R))

# -----------------------------------------------------------------------------
# Subset repetition on the participant set
# -----------------------------------------------------------------------------

function _check_order(order::Integer, what::AbstractString)
    order >= 1 || throw(ArgumentError("$what: the subset order must be at least 1, got $order"))
    return Int(order)
end

"""
    SubsetRepetition(order; aggregate=:mean, memory=FullMemory(), weighted=false,
                     types=nothing, transform=identity, name=nothing)

Subset repetition of order `p = order` for undirected hyperevents: the hyperedge
degree

    deg(h′; t) = Σ_{e: tₑ ≤ t} w(t − tₑ) · χ(h′ ⊆ hₑ)

— the (memory-weighted) number of past events whose participants include all of
`h′` — aggregated over every `p`-subset `h′` of the candidate's participants.
Order 1 is individual activity (preferential attachment), order 2 dyadic
familiarity, order 3 triadic familiarity; orders of three and above express
dependence no dyadic model can (Lerner et al. 2019).

The papers differ in the aggregation, which changes the number, so it is a
keyword:

- `:mean` (default) — `(1/C(|h|, p)) Σ deg(h′)`: "sub-repetition of order p" in
  Lerner, Tranmer, Mowbray & Hâncean (2019), "subset repetition" in Lerner,
  Lomi, Mowbray, Rollings & Tranmer (2021) and Lerner & Hâncean (2023);
  eventnet `UHE_SUB_REPETITION_STAT` with `AVERAGE`.
- `:sum` — the unnormalised `subrep⁽ᵏ⁾` of Lerner, Hâncean & Perc (2025),
  preferential attachment of order k.
- `:min`, `:max`, `:sd` (population standard deviation) — the other aggregators
  the 2019 paper allows; `:samplesd` — the sample standard deviation, which with
  `p = 1` and `weighted=true` is the *prior success disparity* of Lerner &
  Hâncean (2023, p. 17); `:absdiff` — the mean absolute difference over pairs
  of subsets (eventnet `ABSDIFF`).
- `:assortativity` — `−Σ |deg(h′) − deg(h″)|` over unordered pairs of
  `p`-subsets: the degree assortativity of order k of Lerner, Hâncean & Perc
  (2025), zero unless the candidate has more than `p` participants.

A candidate with fewer than `p` participants has no `p`-subset and scores 0.
`memory` is any Revel memory kernel (the papers use [`HalfLife`](@ref));
`weighted=true` adds each past event's `weight` instead of counting it; `types`
restricts the past to some event types. On directed hyperevents the statistic
reads the participant sets (senders and receivers together) — see
[`DirectedSubsetRepetition`](@ref) for the directed families.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1, 2, 3], 1.0), HyperEvent([1, 2], 2.0),
                         HyperEvent([2, 3, 4], 3.0)])
# pairs of {1,2,3}: deg{1,2} = 2, deg{1,3} = 1, deg{2,3} = 2
compute(SubsetRepetition(2), h, [1, 2, 3], Int[], 4.0)                    # 5/3
compute(SubsetRepetition(2; aggregate=:sum), h, [1, 2, 3], Int[], 4.0)    # 5.0
compute(SubsetRepetition(1), h, [1, 2, 3], Int[], 4.0)                    # (2 + 3 + 2)/3
compute(SubsetRepetition(2; memory=Window(1.5)), h, [1, 2, 3], Int[], 4.0)   # 1/3
```
"""
struct SubsetRepetition{M, F} <: AbstractHyperStatistic
    order::Int
    aggregate::Symbol
    read::_HyperRead{M}
    transform::F
    label::String
end

function SubsetRepetition(order::Integer; aggregate::Symbol=:mean,
                          memory=FullMemory(), weighted::Bool=false, types=nothing,
                          transform=identity, name=nothing,
                          base::Union{Nothing,AbstractString}=nothing)
    p = _check_order(order, "SubsetRepetition")
    _check_aggregate(aggregate, _SUBSET_AGGREGATES, "SubsetRepetition")
    rd = _hyper_read(memory, types, weighted)
    f = _transform_fn(transform)
    b = something(base, "subrep($p)") * _agg_label(aggregate)
    return SubsetRepetition{typeof(rd.memory), typeof(f)}(
        p, aggregate, rd, f, _label(name, _auto_name(b, _suffix(rd), f)))
end

# The degrees of every `p`-subset of the participant set `P`, left in `h.vals`
function _subset_degrees!(h::HyperHistory, P::AbstractVector{Int}, p::Int,
                          rd::_HyperRead, t::Float64)
    vals = empty!(h.vals)
    length(P) < p && return vals
    index = _index!(h, (:und, p, 0))
    key = resize!(h.key, p)
    _each_subset(P, p, h.idx1, key, 0) do
        push!(vals, _degree(h, index, key, rd, t))
    end
    return vals
end

_hvalue(stat::SubsetRepetition, h::HyperHistory, S, R, t::Float64) =
    _aggregate(_subset_degrees!(h, _members!(h.members, S, R), stat.order, stat.read, t),
               stat.aggregate)

"""
    ExactRepetition(; direction=:out, memory=FullMemory(), weighted=false,
                    types=nothing, transform=identity, name=nothing)

Exact repetition: the hyperedge activity

    activity(h; t) = Σ_{e: tₑ ≤ t} w(t − tₑ) · χ(hₑ = h)

— the (memory-weighted) number of past events on *exactly* the candidate
hyperedge, with no one missing and no one added ("repetition" in Lerner,
Tranmer, Mowbray & Hâncean 2019; eventnet `UHE_REPETITION_STAT` and
`DHE_REPETITION_STAT`). For a directed candidate both the sender set and the
receiver set must match — the "mailing list" effect of Lerner & Lomi (2023),
`Σ w·1(iₘ = i ∧ Jₘ = J)`.

`direction` matters for directed hyperevents only (eventnet's `direction`
argument):

- `:out` — exact repetition;
- `:in` — exact reciprocation, past events on the reversed hyperedge, the former
  receivers sending to the former senders (the "reciprocation" of Lerner et al.
  2019);
- `:sym` — past events on the same participant set in any role, which is
  [`UnorderedRepetition`](@ref).

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1, 2, 3], 1.0), HyperEvent([1, 2], 2.0),
                         HyperEvent([1, 2], 3.0)])
compute(ExactRepetition(), h, [1, 2], Int[], 4.0)       # 2.0 — not the {1,2,3} meeting
compute(ExactRepetition(), h, [1, 3], Int[], 4.0)       # 0.0
mails = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([2, 3], [1], 2.0)])
compute(ExactRepetition(), mails, [1], [2, 3], 3.0)                 # 1.0
compute(ExactRepetition(direction=:in), mails, [1], [2, 3], 3.0)    # 1.0
```
"""
struct ExactRepetition{M, F} <: AbstractHyperStatistic
    direction::Symbol
    read::_HyperRead{M}
    transform::F
    label::String
end

function _check_direction(direction::Symbol, what::AbstractString)
    direction in (:out, :in, :sym) || throw(ArgumentError(
        "$what: direction must be :out, :in or :sym, got :$direction"))
    return direction
end

function ExactRepetition(; direction::Symbol=:out, memory=FullMemory(),
                         weighted::Bool=false, types=nothing, transform=identity,
                         name=nothing)
    _check_direction(direction, "ExactRepetition")
    rd = _hyper_read(memory, types, weighted)
    f = _transform_fn(transform)
    base = direction === :out ? "exact.rep" : direction === :in ? "exact.recip" :
           "unordered.rep"
    return ExactRepetition{typeof(rd.memory), typeof(f)}(
        direction, rd, f, _label(name, _auto_name(base, _suffix(rd), f)))
end

function _hvalue(stat::ExactRepetition, h::HyperHistory, S, R, t::Float64)
    key = empty!(h.key)
    if stat.direction === :sym
        append!(key, _members!(h.members, S, R))
        return _degree(h, _index!(h, (:union, 0, 0)), key, stat.read, t)
    end
    # The reverse of an undirected hyperedge is the hyperedge itself
    if stat.direction === :out || isempty(R)
        append!(key, S); push!(key, 0); append!(key, R)
    else
        append!(key, R); push!(key, 0); append!(key, S)
    end
    return _degree(h, _index!(h, (:exact, 0, 0)), key, stat.read, t)
end

"""
    UnorderedRepetition(; memory=FullMemory(), weighted=false, types=nothing,
                        transform=identity, name=nothing)

Unordered repetition (Lerner & Lomi 2023): `Σ w·1({iₘ} ∪ Jₘ = {i} ∪ J)`, the
past events among exactly the candidate's participants whoever sent them —
interaction within a stable group with turn-taking among senders, the
"reply to all" pattern. It is `ExactRepetition(direction=:sym)` (eventnet
`DHE_REPETITION_STAT` with direction `SYM`).

The 2019 preprint operationalises "reply to all" differently, as switch
reciprocation of order 1; that statistic is not implemented.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([2], [1, 3], 2.0)])
compute(UnorderedRepetition(), h, [3], [1, 2], 3.0)    # 2.0 — both were among {1,2,3}
compute(ExactRepetition(), h, [3], [1, 2], 3.0)        # 0.0 — 3 has not sent before
```
"""
UnorderedRepetition(; kwargs...) = ExactRepetition(; direction=:sym, kwargs...)

"""
    SharedPriorEvents(order; memory=FullMemory(), weighted=false, types=nothing,
                      transform=identity, name=nothing)

Shared prior events of order `p` (Lerner, Tranmer, Mowbray & Hâncean 2019):

    Σ_{e: tₑ ≤ t} w(t − tₑ) · χ(|hₑ ∩ h| ≥ p)

— the past events that share at least `p` participants with the candidate. It is
the alternative to [`SubsetRepetition`](@ref) that counts each past event once,
where the sum of degrees over `p`-subsets counts it `C(|hₑ ∩ h|, p)` times.
Participant sets are read without roles on directed hyperevents.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1, 2, 3], 1.0), HyperEvent([1, 2], 2.0),
                         HyperEvent([3, 4], 3.0)])
compute(SharedPriorEvents(2), h, [1, 2, 3], Int[], 4.0)   # 2.0 — the first two events
compute(SubsetRepetition(2; aggregate=:sum), h, [1, 2, 3], Int[], 4.0)   # 4.0 = 3 + 1
```
"""
struct SharedPriorEvents{M, F} <: AbstractHyperStatistic
    order::Int
    read::_HyperRead{M}
    transform::F
    label::String
end

function SharedPriorEvents(order::Integer; memory=FullMemory(), weighted::Bool=false,
                           types=nothing, transform=identity, name=nothing)
    p = _check_order(order, "SharedPriorEvents")
    rd = _hyper_read(memory, types, weighted)
    f = _transform_fn(transform)
    return SharedPriorEvents{typeof(rd.memory), typeof(f)}(
        p, rd, f, _label(name, _auto_name("shared.events($p)", _suffix(rd), f)))
end

function _hvalue(stat::SharedPriorEvents, h::HyperHistory, S, R, t::Float64)
    P = _members!(h.members, S, R)
    length(P) < stat.order && return 0.0
    index = _index!(h, (:und, 1, 0))
    overlap = empty!(h.overlap)
    sup = _support(stat.read.memory)
    key = resize!(h.key, 1)
    for a in P
        key[1] = a
        list = get(index, key, nothing)
        list === nothing && continue
        @inbounds for k in length(list):-1:1
            e = list[k]
            age = t - h.times[e]
            age < 0 && continue
            age > sup && break
            overlap[e] = get(overlap, e, 0) + 1
        end
    end
    total = 0.0
    for (e, shared) in overlap
        shared >= stat.order && (total += _event_weight(h, stat.read, e, t - h.times[e]))
    end
    return total
end

"""
    PriorSuccess(order; memory=FullMemory(), types=nothing, transform=identity,
                 name=nothing)

Prior success of order `p` (Lerner & Hâncean 2023): with the event `weight`
holding the outcome `yₑ` of a past hyperevent (the impact of a paper, the
performance of a team),

    Σ_{h′} performance(h′) / Σ_{h′} deg(h′),    performance(h′) = Σ_e w(t − tₑ)·yₑ·χ(h′ ⊆ hₑ)

over the `p`-subsets `h′` of the candidate's participants: the average outcome
of the past events its members (`p = 1`), pairs (`p = 2`) or triads (`p = 3`)
took part in — "prior shared success" for `p ≥ 2`. It is 0 when no `p`-subset
has a past event.

This is a ratio of an outcome-weighted to an unweighted subset repetition; the
outcome-weighted hyperedge degree alone is
`SubsetRepetition(p; weighted=true, aggregate=:sum)`.

The paper's *prior success disparity* — the sample standard deviation of the
members' summed past performance (Lerner & Hâncean 2023, p. 17) — is
`SubsetRepetition(1; weighted=true, aggregate=:samplesd)`.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1, 2], 1.0; weight=10.0),
                         HyperEvent([2, 3], 2.0; weight=4.0)])
# actors 1, 2, 3: outcomes 10, 10 + 4, 4 over 1, 2, 1 past events
compute(PriorSuccess(1), h, [1, 2, 3], Int[], 3.0)     # 28/4 = 7.0
compute(PriorSuccess(2), h, [1, 2, 3], Int[], 3.0)     # (10 + 0 + 4)/2 = 7.0
compute(PriorSuccess(2), h, [1, 3], Int[], 3.0)        # 0.0 — never together
```
"""
struct PriorSuccess{M, F} <: AbstractHyperStatistic
    order::Int
    outcome::_HyperRead{M}
    count::_HyperRead{M}
    transform::F
    label::String
end

function PriorSuccess(order::Integer; memory=FullMemory(), types=nothing,
                      transform=identity, name=nothing)
    p = _check_order(order, "PriorSuccess")
    outcome = _hyper_read(memory, types, true)
    count = _hyper_read(memory, types, false)
    f = _transform_fn(transform)
    return PriorSuccess{typeof(count.memory), typeof(f)}(
        p, outcome, count, f,
        _label(name, _auto_name("prior.success($p)", _suffix(count), f)))
end

function _hvalue(stat::PriorSuccess, h::HyperHistory, S, R, t::Float64)
    P = _members!(h.members, S, R)
    events = sum(_subset_degrees!(h, P, stat.order, stat.count, t); init=0.0)
    events > 0 || return 0.0
    return sum(_subset_degrees!(h, P, stat.order, stat.outcome, t); init=0.0) / events
end

# -----------------------------------------------------------------------------
# Directed sub-hyperedges
# -----------------------------------------------------------------------------

"""
    DirectedSubsetRepetition(p, q; direction=:out, aggregate=:mean,
                             memory=FullMemory(), weighted=false, types=nothing,
                             transform=identity, name=nothing)

Subset repetition of order `(p, q)` for directed hyperevents (Lerner, Tranmer,
Mowbray & Hâncean 2019; eventnet `DHE_SUB_REPETITION_STAT`): the directed
hyperedge degree

    deg(a′, b′; t) = Σ_{e: tₑ ≤ t} w(t − tₑ) · χ(a′ ⊆ Iₑ ∧ b′ ⊆ Jₑ)

— the past events that had all of `a′` among their senders `Iₑ` and all of `b′`
among their receivers `Jₑ` — aggregated over the sub-hyperedges `(a′, b′)` of
the candidate `(a, b)` that `direction` selects:

- `:out` — **subset repetition**: `a′ ⊆ a` with `|a′| = p` and `b′ ⊆ b` with
  `|b′| = q`. `(p, 0)`: the same `p` actors co-initiate again, toward anyone;
  `(0, q)`: the same `q` actors co-receive again, from anyone.
- `:in` — **subset reciprocation**: sub-hyperedges of the *reversed* hyperedge,
  `p` of the candidate's receivers as past senders and `q` of its senders as
  past receivers.
- `:sym` — undirected subset repetition: roles are ignored, and the statistic is
  [`SubsetRepetition`](@ref) of order `p + q` on the participant sets. (eventnet
  documents `SYM` as "undirected subset repetition" without saying how a
  `(p, q)` order maps onto it; this is the reading implemented.)

`aggregate` is as in [`SubsetRepetition`](@ref): `:mean` (default, the 2019 and
2023 papers), `:sum`, `:min`, `:max`, `:sd`, `:absdiff`, `:assortativity`. A
candidate with fewer than `p` senders or `q` receivers (as the direction
requires) scores 0.

The named effects of the papers (Lerner, Tranmer, Mowbray & Hâncean 2019;
Lerner & Lomi 2023) are thin constructors:

| constructor | order, direction |
|---|---|
| [`HyperSenderActivity`](@ref) | `(1, 0)`, `:out` |
| [`HyperReceiverPopularity`](@ref) | `(0, 1)`, `:out` |
| [`ReceiverSetRepetition`](@ref)`(p)` | `(0, p)`, `:out` |
| [`SenderReceiverSetRepetition`](@ref)`(p)` | `(1, p)`, `:out` |
| [`HyperReciprocation`](@ref) | `(1, 1)`, `:in` |
| [`OutInPopularity`](@ref) | `(1, 0)`, `:in` |

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([1], [2, 4], 2.0),
                         HyperEvent([2], [1], 3.0)])
# (1,1) sub-hyperedges of 1 → {2,3}: deg(1→2) = 2, deg(1→3) = 1
compute(DirectedSubsetRepetition(1, 1), h, [1], [2, 3], 4.0)                  # 1.5
# reversed: deg(2→1) = 1, deg(3→1) = 0
compute(DirectedSubsetRepetition(1, 1; direction=:in), h, [1], [2, 3], 4.0)   # 0.5
compute(DirectedSubsetRepetition(0, 2), h, [5], [2, 3], 4.0)                  # 1.0
```
"""
struct DirectedSubsetRepetition{M, F} <: AbstractHyperStatistic
    p::Int
    q::Int
    direction::Symbol
    aggregate::Symbol
    read::_HyperRead{M}
    transform::F
    label::String
end

function DirectedSubsetRepetition(p::Integer, q::Integer; direction::Symbol=:out,
                                  aggregate::Symbol=:mean, memory=FullMemory(),
                                  weighted::Bool=false, types=nothing,
                                  transform=identity, name=nothing,
                                  base::Union{Nothing,AbstractString}=nothing)
    (p >= 0 && q >= 0 && p + q >= 1) || throw(ArgumentError(
        "DirectedSubsetRepetition: the order (p, q) needs p ≥ 0, q ≥ 0 and " *
        "p + q ≥ 1, got ($p, $q)"))
    _check_direction(direction, "DirectedSubsetRepetition")
    _check_aggregate(aggregate, _SUBSET_AGGREGATES, "DirectedSubsetRepetition")
    rd = _hyper_read(memory, types, weighted)
    f = _transform_fn(transform)
    auto = direction === :out ? "subrep($p,$q)" : direction === :in ? "subrecip($p,$q)" :
           "subrep.sym($p,$q)"
    b = something(base, auto) * _agg_label(aggregate)
    return DirectedSubsetRepetition{typeof(rd.memory), typeof(f)}(
        Int(p), Int(q), direction, aggregate, rd, f,
        _label(name, _auto_name(b, _suffix(rd), f)))
end

# The degrees of the (p, q) sub-hyperedges with `p` actors of `A` as past
# senders and `q` actors of `B` as past receivers, left in `h.vals`
function _directed_degrees!(h::HyperHistory, A::AbstractVector{Int},
                            B::AbstractVector{Int}, p::Int, q::Int, rd::_HyperRead,
                            t::Float64)
    vals = empty!(h.vals)
    (length(A) < p || length(B) < q) && return vals
    index = _index!(h, (:dir, p, q))
    key = resize!(h.key, p + q)
    idx2 = h.idx2
    _each_subset(A, p, h.idx1, key, 0) do
        _each_subset(B, q, idx2, key, p) do
            push!(vals, _degree(h, index, key, rd, t))
        end
    end
    return vals
end

function _hvalue(stat::DirectedSubsetRepetition, h::HyperHistory, S, R, t::Float64)
    vals = stat.direction === :out ? _directed_degrees!(h, S, R, stat.p, stat.q, stat.read, t) :
           stat.direction === :in  ? _directed_degrees!(h, R, S, stat.p, stat.q, stat.read, t) :
           _subset_degrees!(h, _members!(h.members, S, R), stat.p + stat.q, stat.read, t)
    return _aggregate(vals, stat.aggregate)
end

"""
    ReceiverSetRepetition(order; aggregate=:mean, memory=FullMemory(),
                          weighted=false, types=nothing, transform=identity,
                          name=nothing)

Partial receiver-set repetition of order `p` (Lerner & Lomi 2023,
`rec_sub_rep⁽ᵖ⁾`): the average, over the `p`-subsets `J′` of the candidate's
receivers, of the hyperedge in-degree `Σ w·1(J′ ⊆ Jₘ)` — how often those `p`
actors were addressed together, by anyone. Order 1 is the average in-degree of
the receivers (popularity); orders of two and above are not sums of dyadic
covariates. It is `DirectedSubsetRepetition(0, p)`.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([4], [2, 3, 5], 2.0)])
compute(ReceiverSetRepetition(2), h, [6], [2, 3], 3.0)    # 2.0 — {2,3} co-received twice
compute(ReceiverSetRepetition(1), h, [6], [2, 5], 3.0)    # (2 + 1)/2 = 1.5
```
"""
ReceiverSetRepetition(order::Integer; kwargs...) =
    DirectedSubsetRepetition(0, _check_order(order, "ReceiverSetRepetition");
                             base="rec.subrep($order)", kwargs...)

"""
    SenderReceiverSetRepetition(order; aggregate=:mean, memory=FullMemory(),
                                weighted=false, types=nothing, transform=identity,
                                name=nothing)

Sender-specific partial receiver-set repetition of order `p` (Lerner & Lomi
2023, `send_rec_sub_rep⁽ᵖ⁾`): the average, over the `p`-subsets `J′` of the
candidate's receivers, of `Σ w·1(i = iₘ ∧ J′ ⊆ Jₘ)` — how often *this sender*
addressed those `p` actors together. Order 1 is the hyperevent form of inertia;
orders of two and above capture sender-specific clusterings of receivers. It is
`DirectedSubsetRepetition(1, p)`, which also averages over the senders when a
candidate has several.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([4], [2, 3], 2.0)])
compute(SenderReceiverSetRepetition(2), h, [1], [2, 3], 3.0)    # 1.0 — once by sender 1
compute(ReceiverSetRepetition(2), h, [1], [2, 3], 3.0)          # 2.0 — twice by anyone
```
"""
SenderReceiverSetRepetition(order::Integer; kwargs...) =
    DirectedSubsetRepetition(1, _check_order(order, "SenderReceiverSetRepetition");
                             base="send.rec.subrep($order)", kwargs...)

"""
    HyperSenderActivity(; aggregate=:mean, memory=FullMemory(), weighted=false,
                        types=nothing, transform=identity, name=nothing)

Sender activity: the average over the candidate's senders of the number of past
hyperevents each has sent — subset repetition of order `(1, 0)` (Lerner, Tranmer,
Mowbray & Hâncean 2019). It is constant within a choice set that holds the
sender fixed (`sampler=:receivers` in [`hyper_design`](@ref)), where the
sender-stratified baseline of Lerner & Lomi (2023) absorbs it.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([1], [4], 2.0)])
compute(HyperSenderActivity(), h, [1], [5, 6], 3.0)    # 2.0
compute(HyperSenderActivity(), h, [2], [5, 6], 3.0)    # 0.0
```
"""
HyperSenderActivity(; kwargs...) =
    DirectedSubsetRepetition(1, 0; base="sender.activity", kwargs...)

"""
    HyperReceiverPopularity(; aggregate=:mean, memory=FullMemory(), weighted=false,
                            types=nothing, transform=identity, name=nothing)

Receiver popularity: the average over the candidate's receivers of the number of
past hyperevents each has received — subset repetition of order `(0, 1)`, and
partial receiver-set repetition of order 1 in Lerner & Lomi (2023). Identical to
`ReceiverSetRepetition(1)` apart from its name.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([4], [2], 2.0)])
compute(HyperReceiverPopularity(), h, [5], [2, 3], 3.0)    # (2 + 1)/2 = 1.5
```
"""
HyperReceiverPopularity(; kwargs...) =
    DirectedSubsetRepetition(0, 1; base="receiver.popularity", kwargs...)

"""
    HyperReciprocation(; aggregate=:mean, memory=FullMemory(), weighted=false,
                       types=nothing, transform=identity, name=nothing)

Reciprocation for directed hyperevents (Lerner & Lomi 2023):
`Σ_{j ∈ J} hy_deg(j, {i}) / |J|` — the average, over the candidate's receivers
`j`, of the past events `j` sent that had the candidate's sender among their
receivers. It is subset reciprocation of order `(1, 1)`,
`DirectedSubsetRepetition(1, 1; direction=:in)`, and a dyadic effect in Lerner
& Lomi's classification (a sum of dyadic covariates over the receivers).

The 2019 preprint's "reciprocation" is the stricter *exact* reciprocation — the
whole receiver set replying to the whole sender set — which is
`ExactRepetition(direction=:in)`.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([2], [1, 4], 1.0), HyperEvent([3], [5], 2.0)])
compute(HyperReciprocation(), h, [1], [2, 3], 3.0)    # (1 + 0)/2 = 0.5
```
"""
HyperReciprocation(; kwargs...) =
    DirectedSubsetRepetition(1, 1; direction=:in, base="reciprocation", kwargs...)

"""
    OutInPopularity(; aggregate=:mean, memory=FullMemory(), weighted=false,
                    types=nothing, transform=identity, name=nothing)

Out-in popularity (Lerner & Lomi 2023): `Σ_{j ∈ J} deg⁽ᵒᵘᵗ⁾(j) / |J|`, the
average number of past hyperevents the candidate's receivers have *sent* — do
active senders attract messages, from anyone? It is subset reciprocation of
order `(1, 0)`, `DirectedSubsetRepetition(1, 0; direction=:in)`.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([2], [1, 4], 1.0), HyperEvent([2], [5], 2.0)])
compute(OutInPopularity(), h, [1], [2, 3], 3.0)    # (2 + 0)/2 = 1.0
```
"""
OutInPopularity(; kwargs...) =
    DirectedSubsetRepetition(1, 0; direction=:in, base="out.in.popularity", kwargs...)

"""
    InteractionAmongReceivers(order=1; aggregate=:mean, memory=FullMemory(),
                              weighted=false, types=nothing, transform=identity,
                              name=nothing)

Past interaction among the receivers, of order `p` (Lerner & Lomi 2023,
`interact_rec⁽ᵖ⁾`):

    Σ_{j ∈ J} Σ_{J′ ⊆ J∖{j}, |J′| = p} hy_deg(j, J′) / (|J| · C(|J| − 1, p)),
    hy_deg(j, J′) = Σ w·1(j = iₘ ∧ J′ ⊆ Jₘ)

— the candidate addresses a past sender `j` together with `p` of that sender's
past receivers (in citation data: cite a paper and some of its references). It
is not a sum of dyadic covariates for any `p ≥ 1`. `aggregate=:sum` drops the
normalisation. A receiver set of fewer than `p + 1` actors scores 0.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([2], [3, 4], 1.0)])
# j = 2: deg(2→3) = 1; j = 3: deg(3→2) = 0
compute(InteractionAmongReceivers(1), h, [1], [2, 3], 2.0)       # (1 + 0)/2 = 0.5
# order 2 over {2,3,4}: only j = 2 with J′ = {3,4} has a past event
compute(InteractionAmongReceivers(2), h, [1], [2, 3, 4], 2.0)    # 1/3
```
"""
struct InteractionAmongReceivers{M, F} <: AbstractHyperStatistic
    order::Int
    aggregate::Symbol
    read::_HyperRead{M}
    transform::F
    label::String
end

function InteractionAmongReceivers(order::Integer=1; aggregate::Symbol=:mean,
                                   memory=FullMemory(), weighted::Bool=false,
                                   types=nothing, transform=identity, name=nothing)
    p = _check_order(order, "InteractionAmongReceivers")
    _check_aggregate(aggregate, (:mean, :sum, :min, :max), "InteractionAmongReceivers")
    rd = _hyper_read(memory, types, weighted)
    f = _transform_fn(transform)
    b = "interact.rec($p)" * _agg_label(aggregate)
    return InteractionAmongReceivers{typeof(rd.memory), typeof(f)}(
        p, aggregate, rd, f, _label(name, _auto_name(b, _suffix(rd), f)))
end

function _hvalue(stat::InteractionAmongReceivers, h::HyperHistory, S, R, t::Float64)
    p = stat.order
    vals = empty!(h.vals)
    length(R) < p + 1 && return 0.0
    index = _index!(h, (:dir, 1, p))
    key = resize!(h.key, p + 1)
    others = h.others
    for j in R
        empty!(others)
        for r in R
            r == j || push!(others, r)
        end
        key[1] = j
        _each_subset(others, p, h.idx1, key, 1) do
            push!(vals, _degree(h, index, key, stat.read, t))
        end
    end
    return _aggregate(vals, stat.aggregate)
end

# -----------------------------------------------------------------------------
# Closure
# -----------------------------------------------------------------------------

const _CLOSURE_KINDS = (transitive=(:out, :in), cyclic=(:in, :out),
                        shared_senders=(:in, :in), shared_receivers=(:out, :out))

"""
    HyperClosure(; dir1=:out, dir2=:in, combine=:min, parallel=:sum,
                 aggregate=:mean, normalize=:none, n_actors=nothing,
                 memory=FullMemory(), weighted=false, types=nothing,
                 transform=identity, name=nothing)
    HyperClosure(kind::Symbol; kwargs...)

Triadic closure for hyperevents (eventnet `UHE_CLOSURE_STAT`/`DHE_CLOSURE_STAT`):
the value of the two-paths `i – k – j` through third actors `k`, aggregated over
the pairs of the candidate hyperedge. It is built on the **dyadic projection**
of the past hyperevents,

    W(i, j; t) = Σ_{e: tₑ ≤ t} w(t − tₑ) · χ(i ∈ Iₑ ∧ j ∈ Jₑ)      (directed)
    W(i, j; t) = Σ_{e: tₑ ≤ t} w(t − tₑ) · χ({i, j} ⊆ hₑ)         (undirected)

— the hyperedge degree of the pair. For each pair of the candidate — a sender
`i` and a receiver `j` of a directed candidate, an unordered pair of participants
of an undirected one — and each third actor `k ∉ {i, j}` (`k` may be another
member of the candidate), the two legs are combined by `combine` (`:min`, the
papers' choice and the default here; or `:product`, eventnet's default), the
parallel paths through different `k` by `parallel` (`:sum` or `:max`), and the
pairs by `aggregate` (`:mean`, `:sum`, `:min`, `:max`).

With the defaults this is the closure of Lerner, Lomi, Mowbray, Rollings &
Tranmer (2021) for undirected meetings — the sum over pairs `{u, v} ⊆ h` and
third actors `w` of `min[deg({u, w}), deg({v, w})]`, divided by the number of
pairs — and, on directed hyperevents, the triadic effects of Lerner & Lomi
(2023), `Σ_{j ∈ J, a ≠ i, j} min{·, ·} / |J|`. `aggregate=:sum` gives the
unnormalised triadic closure of Lerner, Hâncean & Perc (2025).

The legs (directed hyperevents only; the projection of undirected hyperevents is
symmetric and the directions are ignored): `dir1` reads the sender's leg —
`:out` is `W(i, k)`, `:in` is `W(k, i)`, `:sym` their sum; `dir2` reads the
receiver's leg — `:in` is `W(k, j)`, `:out` is `W(j, k)`, `:sym` their sum. The
four variants of Lerner et al. (2019) and Lerner & Lomi (2023) are

| `kind` | `dir1`, `dir2` | legs | also called |
|---|---|---|---|
| `:transitive` | `:out`, `:in` | `W(i,k)`, `W(k,j)` | transitive closure |
| `:cyclic` | `:in`, `:out` | `W(k,i)`, `W(j,k)` | cyclic closure |
| `:shared_senders` | `:in`, `:in` | `W(k,i)`, `W(k,j)` | incoming balance (2023), sender balance, sibling |
| `:shared_receivers` | `:out`, `:out` | `W(i,k)`, `W(j,k)` | outgoing balance (2023), receiver balance, cosibling |

and `HyperClosure(kind)` sets the two directions.

`normalize` records a drift between the papers: the 2019 preprint divides
closure by the number of possible third actors, the papers from 2021 onward do
not. `:none` (default) follows the later papers; `:thirds` also divides by
`n_actors − 2` and needs `n_actors=`. (The preprint does not say whether its
count excludes the other members of the candidate; `n_actors − 2`, every actor
but the pair, is the reading implemented. With a fixed actor set it rescales the
coefficient by a constant.)

Only closure of order `(1, 1, 1)` is implemented — single actors at the three
corners — not the general order `(p, q, l)` of the 2019 preprint. The optional
node attribute on the third actor, the four-cycle and the neighbour statistics
of eventnet are not implemented either.

# Example
```julia
using Revel
h = build_hyper_history([HyperEvent([1, 3], 1.0), HyperEvent([1, 3], 2.0),
                         HyperEvent([2, 3, 4], 3.0)])
# pair {1,2}: through 3, min(W13, W23) = min(2, 1) = 1; through 4, min(0, 1) = 0
compute(HyperClosure(), h, [1, 2], Int[], 4.0)                       # 1.0
compute(HyperClosure(combine=:product), h, [1, 2], Int[], 4.0)       # 2.0
compute(HyperClosure(normalize=:thirds, n_actors=4), h, [1, 2], Int[], 4.0)   # 1/2
mails = build_hyper_history([HyperEvent([1], [3], 1.0), HyperEvent([3], [2, 4], 2.0)])
compute(HyperClosure(:transitive), mails, [1], [2, 4], 3.0)          # (1 + 1)/2 = 1.0
compute(HyperClosure(:cyclic), mails, [1], [2, 4], 3.0)              # 0.0
```
"""
struct HyperClosure{M, F} <: AbstractHyperStatistic
    dir1::Symbol
    dir2::Symbol
    combine::Symbol
    parallel::Symbol
    aggregate::Symbol
    thirds::Float64          # the divisor (1 when normalize = :none)
    read::_HyperRead{M}
    transform::F
    label::String
end

function HyperClosure(; dir1::Symbol=:out, dir2::Symbol=:in, combine::Symbol=:min,
                      parallel::Symbol=:sum, aggregate::Symbol=:mean,
                      normalize::Symbol=:none, n_actors::Union{Nothing,Integer}=nothing,
                      memory=FullMemory(), weighted::Bool=false, types=nothing,
                      transform=identity, name=nothing,
                      base::Union{Nothing,AbstractString}=nothing)
    dir1 in (:out, :in, :sym) || throw(ArgumentError(
        "HyperClosure: dir1 must be :out, :in or :sym, got :$dir1"))
    dir2 in (:out, :in, :sym) || throw(ArgumentError(
        "HyperClosure: dir2 must be :out, :in or :sym, got :$dir2"))
    combine in (:min, :product) || throw(ArgumentError(
        "HyperClosure: combine must be :min or :product, got :$combine"))
    parallel in (:sum, :max) || throw(ArgumentError(
        "HyperClosure: parallel must be :sum or :max, got :$parallel"))
    _check_aggregate(aggregate, (:mean, :sum, :min, :max), "HyperClosure")
    normalize in (:none, :thirds) || throw(ArgumentError(
        "HyperClosure: normalize must be :none (the papers from 2021 onward) or " *
        ":thirds (divide by the number of possible third actors, Lerner et al. " *
        "2019), got :$normalize"))
    thirds = 1.0
    if normalize === :thirds
        n_actors === nothing && throw(ArgumentError(
            "HyperClosure: normalize=:thirds divides by the number of possible " *
            "third actors, n_actors − 2; pass `n_actors=`"))
        n_actors >= 3 || throw(ArgumentError(
            "HyperClosure: normalize=:thirds needs at least three actors, got $n_actors"))
        thirds = Float64(n_actors - 2)
    else
        n_actors === nothing || throw(ArgumentError(
            "HyperClosure: `n_actors` is used by normalize=:thirds only"))
    end
    rd = _hyper_read(memory, types, weighted)
    f = _transform_fn(transform)
    b = something(base, "closure($(dir1),$(dir2))") *
        (combine === :min ? "" : ".product") * (parallel === :sum ? "" : ".max") *
        _agg_label(aggregate) * (normalize === :none ? "" : ".thirds")
    return HyperClosure{typeof(rd.memory), typeof(f)}(
        dir1, dir2, combine, parallel, aggregate, thirds, rd, f,
        _label(name, _auto_name(b, _suffix(rd), f)))
end

function HyperClosure(kind::Symbol; kwargs...)
    haskey(_CLOSURE_KINDS, kind) || throw(ArgumentError(
        "HyperClosure: kind must be one of $(keys(_CLOSURE_KINDS)), got :$kind"))
    dir1, dir2 = _CLOSURE_KINDS[kind]
    return HyperClosure(; dir1=dir1, dir2=dir2, base="closure.$(kind)", kwargs...)
end

# W(i, j) of the dyadic projection
@inline function _pair_weight(h::HyperHistory, index::_HyperIndex, rd::_HyperRead,
                              i::Int, j::Int, t::Float64)
    key = h.pair
    @inbounds if h.directed === true || i < j
        key[1] = i; key[2] = j
    else
        key[1] = j; key[2] = i
    end
    return _degree(h, index, key, rd, t)
end

# The leg between `a` (an actor of the candidate) and the third actor `k`, seen
# from `a`: :out is a → k, :in is k → a
@inline function _closure_leg(h::HyperHistory, index::_HyperIndex, rd::_HyperRead,
                              dir::Symbol, a::Int, k::Int, t::Float64)
    h.directed === true || return _pair_weight(h, index, rd, a, k, t)
    return dir === :out ? _pair_weight(h, index, rd, a, k, t) :
           dir === :in  ? _pair_weight(h, index, rd, k, a, t) :
           _pair_weight(h, index, rd, a, k, t) + _pair_weight(h, index, rd, k, a, t)
end

function _closure_pair(stat::HyperClosure, h::HyperHistory, index::_HyperIndex,
                       nb::Dict{Int, Set{Int}}, i::Int, j::Int, t::Float64)
    thirds = get(nb, i, nothing)
    thirds === nothing && return 0.0
    total = 0.0
    # A third actor with no weight on the first leg contributes nothing under
    # either `combine`, so the neighbours of `i` are all that is needed
    for k in thirds
        (k == i || k == j) && continue
        a = _closure_leg(h, index, stat.read, stat.dir1, i, k, t)
        a > 0 || continue
        b = _closure_leg(h, index, stat.read, stat.dir2, j, k, t)
        b > 0 || continue
        v = stat.combine === :min ? min(a, b) : a * b
        total = stat.parallel === :sum ? total + v : max(total, v)
    end
    return total
end

function _hvalue(stat::HyperClosure, h::HyperHistory, S, R, t::Float64)
    h.directed === nothing && return 0.0
    index = _index!(h, h.directed ? (:dir, 1, 1) : (:und, 2, 0))
    nb = _neighbours!(h)
    vals = empty!(h.vals)
    if isempty(R)
        for a in eachindex(S), b in (a + 1):lastindex(S)
            push!(vals, _closure_pair(stat, h, index, nb, S[a], S[b], t))
        end
    else
        for i in S, j in R
            push!(vals, _closure_pair(stat, h, index, nb, i, j, t))
        end
    end
    return _aggregate(vals, stat.aggregate) / stat.thirds
end

# -----------------------------------------------------------------------------
# Covariates on a hyperedge
# -----------------------------------------------------------------------------

const _COVARIATE_AGGREGATES = (:mean, :sum, :min, :max, :sd, :samplesd, :absdiff, :catdiff,
                               :homogeneity)

"""
    HyperCovariate(x; endpoint=:all, aggregate=:mean, transform=identity,
                   name=nothing)

A summary of an actor covariate over the candidate hyperedge (eventnet
`UHE_NODE_STAT`/`DHE_NODE_STAT` and its aggregation functions). `x` is a
[`Covariate`](@ref) (static or time-varying) or a vector indexed by actor ID.

- `endpoint` — whose values are summarised: `:all` participants, the `:senders`
  or the `:receivers` (an undirected hyperedge has `:all` only).
- `aggregate` — `:mean`, `:sum`, `:min`, `:max`, `:sd` (population standard
  deviation; eventnet `SDEV`), `:samplesd` (`SAMPLESDEV`), `:absdiff` (the mean
  absolute difference over pairs), `:catdiff` (the share of pairs with
  different values, for a categorical covariate — the only aggregate a
  categorical covariate accepts) or `:homogeneity` (for a 0/1 covariate: the
  covariate homogeneity of Lerner et al. 2021 — the larger group minus the
  smaller, over the size, rescaled for an odd size so that the most even split
  scores 0 and a single group 1).

Pairs for `:absdiff` and `:catdiff` are the unordered pairs within the chosen
endpoint — except with `endpoint=:all` on a directed hyperedge, where, as in
eventnet, they are the (sender, receiver) pairs: that is how sender–receiver
homophily is tested. A set with fewer than two actors has no pair and scores 0.

The covariate effects of the papers are special cases:

| effect | arguments |
|---|---|
| covariate average (Lerner et al. 2021) | `aggregate=:mean` |
| covariate homogeneity of a binary covariate (Lerner et al. 2021) | `aggregate=:homogeneity` |
| covariate dispersion within the hyperedge | `aggregate=:absdiff`, `:sd` or `:samplesd` |
| receiver-set average (Lerner & Lomi 2023) | `endpoint=:receivers, aggregate=:mean` |
| sender–receiver heterophily (2023), mean `abs(zᵢ − zⱼ)` | `endpoint=:all, aggregate=:absdiff` on directed hyperedges |
| receiver-set heterophily (2023), mean pairwise difference within `J` | `endpoint=:receivers, aggregate=:absdiff` |

A summary over the senders alone is constant within a choice set that holds the
senders fixed (`sampler=:receivers`), and any summary that depends only on the
size of the hyperedge is never identified.

# Example
```julia
using Revel
age = [30.0, 40.0, 50.0, 20.0]
h = HyperHistory{Float64}()
compute(HyperCovariate(age), h, [1, 2, 3], Int[], 0.0)                        # 40.0
compute(HyperCovariate(age; aggregate=:absdiff), h, [1, 2, 3], Int[], 0.0)    # (10 + 20 + 10)/3
# directed: sender 1, receivers {2, 4}
compute(HyperCovariate(age; endpoint=:receivers), h, [1], [2, 4], 0.0)        # 30.0
compute(HyperCovariate(age; aggregate=:absdiff), h, [1], [2, 4], 0.0)         # (10 + 10)/2
dept = Covariate([:a, :a, :b, :b]; name="dept")
compute(HyperCovariate(dept; aggregate=:catdiff), h, [1, 2, 3], Int[], 0.0)   # 2/3
female = [1.0, 0.0, 1.0, 1.0]
compute(HyperCovariate(female; aggregate=:homogeneity), h, [1, 2, 3, 4], Int[], 0.0)   # (3 − 1)/4 = 0.5
compute(HyperCovariate(female; aggregate=:homogeneity), h, [1, 3, 4], Int[], 0.0)      # 1.0 — one group
```
"""
struct HyperCovariate{F} <: AbstractHyperStatistic
    x::Covariate
    endpoint::Symbol
    aggregate::Symbol
    transform::F
    label::String
end

function HyperCovariate(x; endpoint::Symbol=:all, aggregate::Symbol=:mean,
                        transform=identity, name=nothing)
    _check_endpoint(endpoint, "HyperCovariate")
    aggregate in _COVARIATE_AGGREGATES || throw(ArgumentError(
        "HyperCovariate: aggregate must be one of $(_COVARIATE_AGGREGATES), " *
        "got :$aggregate"))
    c = _covariate(x)
    aggregate === :catdiff || _require_numeric(c, "HyperCovariate(aggregate=:$aggregate)")
    aggregate === :homogeneity && !all(v -> v == 0 || v == 1, c.values) && throw(ArgumentError(
        "HyperCovariate(aggregate=:homogeneity) is defined for a binary (0/1) " *
        "covariate; :$(c.label) takes other values"))
    f = _transform_fn(transform)
    base = "$(aggregate).$(c.label)" * (endpoint === :all ? "" : ".$(endpoint)")
    return HyperCovariate{typeof(f)}(c, endpoint, aggregate, f,
                                     _label(name, _auto_name(base, "", f)))
end

function _hvalue(stat::HyperCovariate, h::HyperHistory, S, R, t::Float64)
    x = stat.x
    differ = stat.aggregate === :catdiff
    if stat.endpoint === :all && !isempty(R) && (differ || stat.aggregate === :absdiff)
        total = 0.0
        for i in S, j in R
            a = covariate_value(x, i, t)
            b = covariate_value(x, j, t)
            total += differ ? Float64(a != b) : abs(a - b)
        end
        return total / (length(S) * length(R))
    end
    stat.endpoint === :receivers && isempty(R) && throw(ArgumentError(
        "$(name(stat)): endpoint=:receivers needs a directed hyperedge, but the " *
        "candidate $(_set_string(S)) has no receivers"))
    vals = empty!(h.vals)
    if stat.endpoint !== :receivers
        for i in S
            push!(vals, covariate_value(x, i, t))
        end
    end
    if stat.endpoint !== :senders
        for j in R
            push!(vals, covariate_value(x, j, t))
        end
    end
    return _aggregate(vals, stat.aggregate)
end

# -----------------------------------------------------------------------------
# Non-event sampling: the size-stratified case-control design
# -----------------------------------------------------------------------------

const _HYPER_TIES_SUPPORTED = (:error, :ordered, :breslow)
const _HYPER_TIES_MODEL = "the hyperevent case-control design (a conditional " *
                          "likelihood over the ORDER of hyperevents)"
const _HYPER_TIES_REASONS = Dict(
    :efron => "the Efron correction keeps every case tied with the one being " *
              "explained in its denominator, which requires the tied events to " *
              "share one risk set; here each hyperevent is compared with " *
              "sampled alternatives of its own size, so tied events of " *
              "different sizes have different risk sets. Pass `ties=:breslow`",
    :batch => "an ordinal likelihood has no exposure interval for a batch to " *
              "consume; holding the history fixed across the tied events IS the " *
              "Breslow correction, so pass `ties=:breslow` instead")

const _HYPER_SAMPLERS = (:auto, :uniform, :receivers)
const _HYPER_BOOKKEEPING = ("senders", "receivers")

# binomial(n, k), saturating at typemax(Int) instead of overflowing
function _binomial_sat(n::Int, k::Int)
    (k < 0 || k > n) && return 0
    k = min(k, n - k)
    r = Int128(1)
    for i in 1:k
        r = (r * (n - k + i)) ÷ i          # exact: r·(n−k+i)/i = C(n−k+i, i)
        r > typemax(Int) && return typemax(Int)
    end
    return Int(r)
end

function _mul_sat(a::Int, b::Int)
    r, overflow = Base.Checked.mul_with_overflow(a, b)
    return overflow ? typemax(Int) : r
end

# The number of hyperedges with `p` senders and `q` receivers (disjoint) among
# `n` actors; `fixed` holds the senders fixed and counts the receiver sets
_n_hyperedges(n::Int, p::Int, q::Int, fixed::Bool) =
    fixed ? _binomial_sat(n - p, q) :
    q == 0 ? _binomial_sat(n, p) : _mul_sat(_binomial_sat(n, p), _binomial_sat(n - p, q))

const _Hyperedge = Tuple{Vector{Int}, Vector{Int}}

# Every hyperedge with `p` senders and `q` receivers among the sorted `pool`
# (or, with the senders `fixed`, every receiver set of size `q` outside them)
function _all_hyperedges(pool::Vector{Int}, p::Int, q::Int,
                         fixed::Union{Nothing, Vector{Int}}=nothing)
    out = _Hyperedge[]
    idx1 = Int[]; idx2 = Int[]
    rest = Int[]
    each_receiver_set = function (S::Vector{Int})
        if q == 0
            push!(out, (S, Int[]))
            return nothing
        end
        empty!(rest)
        for a in pool
            insorted(a, S) || push!(rest, a)
        end
        R = Vector{Int}(undef, q)
        _each_subset(rest, q, idx2, R, 0) do
            push!(out, (S, copy(R)))
        end
        return nothing
    end
    if fixed !== nothing
        each_receiver_set(fixed)
    else
        S = Vector{Int}(undef, p)
        _each_subset(pool, p, idx1, S, 0) do
            each_receiver_set(copy(S))
        end
    end
    return out
end

# A uniform draw of `k` distinct actors: a partial Fisher–Yates shuffle, which
# is uniform from any starting arrangement of `work`
function _shuffle_head!(rng::AbstractRNG, work::Vector{Int}, k::Int)
    n = length(work)
    @inbounds for i in 1:k
        j = rand(rng, i:n)
        work[i], work[j] = work[j], work[i]
    end
    return work
end

# `keep` distinct hyperedges of size (p, q), none equal to an `exclude`d one,
# uniformly without replacement (rejection on a uniform proposal)
function _sample_hyperedges!(rng::AbstractRNG, out::Vector{_Hyperedge},
                             seen::Set{_Hyperedge}, work::Vector{Int}, p::Int, q::Int,
                             keep::Int, fixed::Union{Nothing, Vector{Int}})
    while length(out) < keep
        if fixed === nothing
            _shuffle_head!(rng, work, p + q)
            candidate = (sort!(work[1:p]), sort!(work[(p + 1):(p + q)]))
        else
            _shuffle_head!(rng, work, q)
            candidate = (fixed, sort!(work[1:q]))
        end
        candidate in seen && continue
        push!(seen, candidate)
        push!(out, candidate)
    end
    return out
end

# `:auto` is the sender-stratified design for directed hyperevents (Lerner &
# Lomi 2023) and the uniform one for undirected hyperevents, which have no senders
_resolve_sampler(sampler::Symbol, sorted) =
    sampler !== :auto ? sampler :
    (!isempty(sorted) && is_directed(first(sorted))) ? :receivers : :uniform

function _hyper_pool(actors, n_actors::Int)
    n_actors >= 2 || throw(ArgumentError("need at least two actors"))
    actors === nothing && return collect(1:n_actors)
    actors isa Tuple && throw(ArgumentError(
        "two actor sets were passed as `actors`, which describes a two-mode " *
        "hyperevent network (authors publishing a paper that cites papers; Lerner, " *
        "Hâncean & Lomi 2025). Two-mode hyperevents are not implemented: senders " *
        "and receivers are drawn from one set of actors at risk."))
    pool = _actor_set(actors, "the actors at risk")
    length(pool) >= 2 || throw(ArgumentError("need at least two actors at risk"))
    last(pool) <= n_actors || throw(ArgumentError(
        "the actors at risk include $(last(pool)), beyond n_actors = $n_actors"))
    return pool
end

function _hyper_stat_names(statistics)
    names = _stat_names(statistics)
    for stat in statistics
        _is_hyper(stat) || throw(ArgumentError(
            "$(name(stat)) is not a hyperevent statistic: it is defined on a dyad " *
            "(sender, receiver), not on a hyperedge. Hyperevent models take " *
            "subtypes of AbstractHyperStatistic (SubsetRepetition, HyperClosure, " *
            "HyperCovariate, …) and Interaction/Transformed wrappers of them."))
    end
    clash = intersect(names, _HYPER_BOOKKEEPING)
    isempty(clash) || throw(ArgumentError(
        "statistic name $(repr(first(clash))) collides with a bookkeeping column " *
        "of the design; rename it with `name=`"))
    return names
end

# Maximal runs of equal time in a time-sorted vector of hyperevents
function _hyper_tie_blocks(sorted::Vector{<:HyperEvent})
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

# Function barrier: `stats` is a tuple, so each compute call is statically
# dispatched
function _push_hyper_row!(columns::Vector{Vector{Float64}}, stats::S, history,
                          senders::Vector{Int}, receivers::Vector{Int}, t) where S<:Tuple
    vals = map(stat -> compute(stat, history, senders, receivers, t), stats)
    for k in eachindex(vals)
        push!(columns[k], vals[k])
    end
    return nothing
end

"""
    hyper_design(events, statistics, n_actors; n_controls=20,
                 rng=Random.default_rng(), sampler=:auto, ties=:error,
                 actors=nothing) -> DataFrame

The case-control design of a relational hyperevent model (Lerner & Lomi 2023;
eventnet): for each observed hyperevent, one row for the event and one for each
of `n_controls` alternative hyperedges **of the same size** — the same number of
senders and of receivers, sender and receiver sets disjoint — drawn uniformly
without replacement from the hyperedges the actors at risk could have formed,
the observed one excluded. The statistics of every row are read off the history
strictly before the event.

The number of hyperedges of a given size is astronomically large for all but
the smallest networks, so the risk set cannot be enumerated; conditioning on the
observed size and sampling non-events is what makes the model estimable. When
the alternatives of that size number `n_controls` or fewer they are **all**
included, and the conditional likelihood is then exact for the size-stratified
risk set.

The frame has the bookkeeping columns `REM.fit_rem(::DataFrame, names)` reads —
`event_index` (position in the time-sorted sequence), `is_event`, `stratum`,
`risk_set_size` (the number of possible hyperedges of that size, saturating at
`typemax(Int)`), `sampling_prob` (the share of the alternatives that was kept),
`tie_weight` — plus `time`, the list columns `senders` and `receivers`
describing each row's hyperedge, and one column per statistic, named by
`name(stat)`. The tie policy that applied rides along as the `"tie_method"`
metadata.

- `sampler` — `:receivers` keeps the observed senders and draws receiver sets
  only: the design of Lerner & Lomi (2023), whose baseline is stratified by
  sender and receiver-set size. Under it a statistic of the senders alone
  ([`HyperSenderActivity`](@ref), a sender covariate) is constant within every
  stratum and not identified. `:uniform` draws whole hyperedges (senders and
  receivers); for directed hyperevents it assumes every sender set is equally
  likely to act, and is biased when senders differ in how often they act
  unless the model accounts for it. `:auto` (the default) is `:receivers` for
  directed hyperevents and `:uniform` for undirected ones.
- `ties` — `:error` (default), `:ordered` (sequence order, no correction) or
  `:breslow` (the history is frozen across a block of tied events), with the
  meaning they have in [`each_risk_set`](@ref). `:efron` and `:batch` are
  refused with the reason.
- `actors` — the actor IDs at risk (default `1:n_actors`), or a function
  `(index, event) -> actor IDs` when the actors at risk change over the sequence
  (actors joining or leaving). Every observed
  participant must be among them.

A statistic that depends on the size of the hyperedge alone
([`HyperedgeSize`](@ref)) is constant within every stratum; include it only
inside an [`Interaction`](@ref).

# Example
```julia
using Revel, Random
events = [HyperEvent([1, 2], 1.0), HyperEvent([1, 2, 3], 2.0), HyperEvent([1, 2], 3.0)]
design = hyper_design(events, [SubsetRepetition(2)], 5; n_controls=3, rng=Xoshiro(1))
size(design, 1)                          # 12 — three events × (1 case + 3 controls)
design.risk_set_size[design.is_event]    # [10, 10, 10] — C(5,2), C(5,3), C(5,2)
design[design.is_event, "subrep(2)"]     # [0.0, 1/3, 2.0]
```
"""
function hyper_design(events::AbstractVector{HyperEvent{T}}, statistics, n_actors::Int;
                      n_controls::Int=20, rng::AbstractRNG=Random.default_rng(),
                      sampler::Symbol=:auto, ties::Symbol=:error,
                      actors=nothing) where T
    check_tie_policy(ties, _HYPER_TIES_SUPPORTED; model=_HYPER_TIES_MODEL,
                     reasons=_HYPER_TIES_REASONS)
    sampler in _HYPER_SAMPLERS || throw(ArgumentError(
        "sampler must be :auto, :uniform (draw whole hyperedges of the observed " *
        "size) or :receivers (keep the observed senders, draw receiver sets), got " *
        ":$sampler"))
    n_controls >= 1 || throw(ArgumentError("n_controls must be at least 1"))
    isempty(events) && throw(ArgumentError("no hyperevents to build a design from"))
    names = _hyper_stat_names(statistics)
    stats = Tuple(statistics)
    p_stats = length(names)
    # `actors` may change from event to event: a function (index, event) -> actors
    dynamic = actors isa Function
    pool = dynamic ? Int[] : _hyper_pool(actors, n_actors)
    n = length(pool)

    sorted = sort(events; by=e -> e.time)
    sampler = _resolve_sampler(sampler, sorted)
    blocks = _hyper_tie_blocks(sorted)
    has_ties = any(b -> length(b) > 1, blocks)
    if ties === :error && has_ties
        b = first(filter(b -> length(b) > 1, blocks))
        throw(ArgumentError(
            "the hyperevent sequence contains tied timestamps: events " *
            "$(first(b))–$(last(b)) all occur at t = $(sorted[first(b)].time). The " *
            "conditional likelihood is a likelihood over the ORDER of events, which " *
            "a tie leaves unobserved. Choose a policy: `ties=:breslow` (the " *
            "history is held fixed across the tied events) or `ties=:ordered` " *
            "(sequence order, no correction)."))
    end
    freeze = ties === :breslow

    event_index = Int[]; time = T[]; is_event = Bool[]; stratum = Int[]
    rs_size = Int[]; prob = Float64[]; tie_weight = Float64[]
    senders = Vector{Int}[]; receivers = Vector{Int}[]
    columns = [Float64[] for _ in 1:p_stats]

    history = HyperHistory{T}()
    work = copy(pool)
    controls = _Hyperedge[]
    seen = Set{_Hyperedge}()

    function push_row!(m::Int, ev::HyperEvent, S::Vector{Int}, R::Vector{Int},
                       case::Bool, N::Int, sp::Float64)
        push!(event_index, m); push!(time, ev.time); push!(is_event, case)
        push!(stratum, m); push!(rs_size, N); push!(prob, sp); push!(tie_weight, 1.0)
        push!(senders, copy(S)); push!(receivers, copy(R))
        _push_hyper_row!(columns, stats, history, S, R, ev.time)
        return nothing
    end

    for block in blocks
        for m in block
            ev = sorted[m]
            if dynamic
                pool = _hyper_pool(actors(m, ev), n_actors)
                n = length(pool)
                empty!(work); append!(work, pool)
            end
            p, q = length(ev.senders), length(ev.receivers)
            for set in (ev.senders, ev.receivers), a in set
                insorted(a, pool) || throw(ArgumentError(
                    "event $m ($ev) involves actor $a, who is not among the actors " *
                    "at risk; declare the actor universe (`n_actors`, `actors`) so " *
                    "that every observed hyperevent is possible"))
            end
            fixed = sampler === :receivers
            fixed && q == 0 && throw(ArgumentError(
                "sampler=:receivers draws alternative receiver sets, but event $m " *
                "($ev) is undirected; use sampler=:uniform"))
            N = _n_hyperedges(n, p, q, fixed)
            N >= 2 || throw(ArgumentError(
                "event $m ($ev) is the only possible hyperedge of its size among " *
                "$n actors at risk; a case needs at least one alternative"))
            eligible = N - 1
            keep = min(n_controls, eligible)
            sp = keep / eligible

            push_row!(m, ev, ev.senders, ev.receivers, true, N, sp)
            observed = (ev.senders, ev.receivers)
            if keep == eligible
                for (S, R) in _all_hyperedges(pool, p, q, fixed ? ev.senders : nothing)
                    (S == ev.senders && R == ev.receivers) && continue
                    push_row!(m, ev, S, R, false, N, sp)
                end
            else
                empty!(controls); empty!(seen); push!(seen, observed)
                if fixed
                    empty!(work)
                    for a in pool
                        insorted(a, ev.senders) || push!(work, a)
                    end
                end
                _sample_hyperedges!(rng, controls, seen, work, p, q, keep,
                                    fixed ? ev.senders : nothing)
                for (S, R) in controls
                    push_row!(m, ev, S, R, false, N, sp)
                end
                fixed && (empty!(work); append!(work, pool))
            end
            freeze || update_hyper_history!(history, ev)
        end
        if freeze
            for m in block
                update_hyper_history!(history, sorted[m])
            end
        end
    end

    df = DataFrame(event_index=event_index, time=time, is_event=is_event,
                   stratum=stratum, risk_set_size=rs_size, sampling_prob=prob,
                   tie_weight=tie_weight, senders=senders, receivers=receivers)
    for k in 1:p_stats
        df[!, names[k]] = columns[k]
    end
    metadata!(df, "tie_method", string(has_ties ? ties : :none); style=:note)
    return df
end

# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------

"""
    HyperFit

A fitted relational hyperevent model: the `REM.REMResult` of the conditional
logit on the case-control design (`fit.fit`) together with the time-sorted
hyperevents, the statistics, `n_actors`, `n_controls`, the `sampler` and the tie
policy that produced it.

It answers the StatsAPI verbs (`coef`, `stderror`, `vcov`, `confint`,
`loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `coeftable`, `coefnames`) and the
ecosystem's result-metadata protocol (`Networks.fit_metadata`) by forwarding
them to the underlying fit. Goodness of fit is not implemented for hyperevent
fits: `gof(fit)` throws an `ArgumentError`.

# Example
```julia
using Revel, Random
stats = [SubsetRepetition(2; transform=:log1p)]
events = simulate_hyperevents(stats, [1.0], 6, 80; sizes=[2, 3], rng=Xoshiro(1))
fit = fit_rhem(events, stats, 6; rng=Xoshiro(2))
fit isa HyperFit                  # true
coefnames(fit)                    # ["log1p(subrep(2))"]
nobs(fit), fit.n_controls         # (80, 20)
```
"""
struct HyperFit{F, T}
    fit::F
    events::Vector{HyperEvent{T}}
    statistics::Vector{AbstractStatistic}
    n_actors::Int
    n_controls::Int
    sampler::Symbol
    ties::Symbol
end

Base.show(io::IO, fit::HyperFit) =
    print(io, "HyperFit(", length(fit.events), " hyperevents, ", length(fit.statistics),
          " statistic", length(fit.statistics) == 1 ? "" : "s", ")")

function Base.show(io::IO, ::MIME"text/plain", fit::HyperFit)
    directed = !isempty(fit.events) && is_directed(first(fit.events))
    println(io, "Revel relational hyperevent model")
    println(io, "  events:    ", length(fit.events), directed ? ", directed" : ", undirected")
    println(io, "  actors:    ", fit.n_actors)
    println(io, "  risk set:  hyperedges of the observed size",
            fit.sampler === :receivers ? " with the observed senders" : "",
            " (up to ", fit.n_controls, " sampled controls)")
    println(io, "  estimator: REM.fit_rem on the hyperevent design")
    println(io)
    # REM's printout speaks of dyads and of a full risk set that a hyperevent
    # model cannot enumerate; say it in hyperedges
    text = sprint(show, fit.fit)
    text = replace(text, r"(Risk-set size:[^\n]*?)dyads" => s"\1hyperedges")
    text = replace(text, r"refit with a\s+larger `n_controls` or the full risk set, or measure the draw-to-draw\s+spread with `control_draw_cov`\." =>
                   "refit with a larger\n`n_controls`, or with another `rng`, and compare.")
    print(io, text)
end

# REM's list speaks of dyads and suggests remedies that do not exist for
# hyperedges; replace its sampling entry with the hyperevent one
function approximations(fit::HyperFit)
    return map(approximations(fit.fit)) do note
        startswith(note, "case-control sampling of the risk set") || return note
        "case-control sampling of hyperedges: each stratum holds the observed " *
        "hyperedge and up to $(fit.n_controls) others of the same size " *
        (fit.sampler === :receivers ? "with the same senders " : "") *
        "drawn uniformly, so the partial likelihood approximates the one over " *
        "every hyperedge of that size; if the model is misspecified the " *
        "estimates depend on the draw — refit with a larger `n_controls`, or " *
        "with another `rng`, and compare"
    end
end

"""
    fit_rhem(events, statistics, n_actors; n_controls=20,
             rng=Random.default_rng(), sampler=:auto, ties=:error,
             actors=nothing, se=:hessian, maxiter=100, tol=1e-8) -> HyperFit

Fit a relational hyperevent model (RHEM; Lerner, Tranmer, Mowbray & Hâncean
2019; Lerner & Lomi 2023): the rate of a hyperevent on the hyperedge `h` is
`λ(h; t) = λ₀(t, |h|) · exp(θ′x(h; t))`, with a baseline stratified by the size
of the hyperedge, and `θ` is estimated from the conditional probability that the
observed hyperedge — rather than one of the alternatives of its size — is the
one on which the event occurred. [`rhem`](@ref) is an alias.

The function builds the size-stratified case-control design with
[`hyper_design`](@ref) (`n_controls`, `rng`, `sampler`, `ties` and `actors` are
its keywords) and fits it with `REM.fit_rem(design, names)`, the ecosystem's
conditional logit — `se` is `:hessian` or `:sandwich`, and non-convergence,
collinearity and separation are reported as that function reports them.

Sampling non-events leaves the estimator consistent; the estimates vary from
one draw of controls to the next, so fix `rng` for reproducibility and raise
`n_controls` to reduce that variation — the default 20 keeps small examples
fast, and the papers use about 100. Statistic names must be unique.

# Example
```julia
using Revel, Random
truth = [SubsetRepetition(1; transform=:log1p), SubsetRepetition(2; transform=:log1p)]
events = simulate_hyperevents(truth, [0.5, 1.0], 7, 200; sizes=[2, 3], rng=Xoshiro(3))
fit = fit_rhem(events, truth, 7; n_controls=40, rng=Xoshiro(4))
round.(coef(fit); digits=1)      # close to [0.5, 1.0]
stderror(fit)                    # from the conditional-logit information
```
"""
function fit_rhem(events::AbstractVector{HyperEvent{T}},
                  statistics::AbstractVector{<:AbstractStatistic}, n_actors::Int;
                  n_controls::Int=20, rng::AbstractRNG=Random.default_rng(),
                  sampler::Symbol=:auto, ties::Symbol=:error, actors=nothing,
                  se::Symbol=:hessian, maxiter::Int=100, tol::Float64=1e-8) where T
    isempty(events) && throw(ArgumentError("no hyperevents to fit"))
    names = _hyper_stat_names(statistics)
    stats = collect(AbstractStatistic, statistics)
    sorted = sort(collect(HyperEvent{T}, events); by=e -> e.time)
    sampler = _resolve_sampler(sampler, sorted)
    design = hyper_design(sorted, stats, n_actors; n_controls=n_controls, rng=rng,
                          sampler=sampler, ties=ties, actors=actors)
    inner = REM.fit_rem(design, names; maxiter=maxiter, tol=tol, se=se)
    return HyperFit{typeof(inner), T}(inner, sorted, stats, n_actors, n_controls,
                                      sampler, ties)
end

"""
    rhem(events, statistics, n_actors; kwargs...) -> HyperFit

Alias for [`fit_rhem`](@ref) (the ecosystem's convention: every model answers to
a `fit_<model>` name and a short one).

# Example
```julia
using Revel, Random
stats = [SubsetRepetition(2; transform=:log1p)]
events = simulate_hyperevents(stats, [1.0], 6, 80; sizes=[2, 3], rng=Xoshiro(5))
rhem === fit_rhem                                  # true
coef(rhem(events, stats, 6; rng=Xoshiro(6)))[1] > 0    # true
```
"""
const rhem = fit_rhem

# StatsAPI and the result-metadata protocol: forwarded to the underlying fit
coef(fit::HyperFit) = coef(fit.fit)
stderror(fit::HyperFit) = stderror(fit.fit)
vcov(fit::HyperFit) = vcov(fit.fit)
confint(fit::HyperFit; kwargs...) = confint(fit.fit; kwargs...)
loglikelihood(fit::HyperFit) = loglikelihood(fit.fit)
nobs(fit::HyperFit) = nobs(fit.fit)
dof(fit::HyperFit) = dof(fit.fit)
aic(fit::HyperFit) = aic(fit.fit)
bic(fit::HyperFit) = bic(fit.fit)
coeftable(fit::HyperFit) = coeftable(fit.fit)

"""
    coefnames(fit::HyperFit) -> Vector{String}

The coefficient names in `coef(fit)` order: the names of the statistics.

# Example
```julia
using Revel, Random
stats = [SubsetRepetition(1; transform=:log1p), ExactRepetition(transform=:log1p)]
events = simulate_hyperevents(stats, [0.5, 0.5], 6, 60; sizes=[2, 3], rng=Xoshiro(7))
coefnames(fit_rhem(events, stats, 6; rng=Xoshiro(8)))
# ["log1p(subrep(1))", "log1p(exact.rep)"]
```
"""
coefnames(fit::HyperFit) = [name(s) for s in fit.statistics]

estimand(fit::HyperFit) = estimand(fit.fit)
objective(fit::HyperFit) = objective(fit.fit)
is_exact(fit::HyperFit) = is_exact(fit.fit)
se_method(fit::HyperFit) = se_method(fit.fit)
missing_method(fit::HyperFit) = missing_method(fit.fit)
tie_method(fit::HyperFit) = tie_method(fit.fit)

_no_hyper(what) = throw(ArgumentError(
    "$what is defined for dyadic relational event fits (`RevelFit`); it is not " *
    "implemented for relational hyperevent fits. Simulate from the fitted model " *
    "with `simulate_hyperevents(fit.statistics, coef(fit), fit.n_actors, n; " *
    "sizes=…)` and compare summaries of your own."))
event_diagnostics(::HyperFit; kwargs...) = _no_hyper("event_diagnostics")
prediction_summary(::HyperFit; kwargs...) = _no_hyper("prediction_summary")
score_process_test(::HyperFit; kwargs...) = _no_hyper("score_process_test")
score_test(::HyperFit, candidates) = _no_hyper("score_test")
statistic_collinearity(::HyperFit) = _no_hyper("statistic_collinearity(fit)")

gof(::HyperFit; kwargs...) = throw(ArgumentError(
    "goodness of fit is not implemented for relational hyperevent fits: the " *
    "auxiliary statistics and diagnostics of `gof`, `event_diagnostics` and " *
    "`prediction_summary` are defined on dyads. Simulate from the fitted model " *
    "with `simulate_hyperevents(fit.statistics, coef(fit), fit.n_actors, n; " *
    "sizes=…)` and compare summaries of your own."))

# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

function _hyper_sizes(sizes, directed::Union{Nothing, Bool}, n::Int)
    (sizes isa AbstractVector && !isempty(sizes)) || throw(ArgumentError(
        "`sizes` must be a non-empty vector of hyperedge sizes, or of " *
        "(senders, receivers) size pairs for directed hyperevents"))
    paired = all(s -> s isa Tuple{Integer, Integer}, sizes)
    paired || all(s -> s isa Integer, sizes) || throw(ArgumentError(
        "`sizes` must hold either integers or (senders, receivers) pairs, got " *
        "$(repr(sizes))"))
    paired && directed === false && throw(ArgumentError(
        "`sizes` holds (senders, receivers) pairs, which describe directed " *
        "hyperevents, but directed=false"))
    dir = something(directed, paired)
    out = Tuple{Int,Int}[]
    for s in sizes
        p, q = paired ? (Int(s[1]), Int(s[2])) : dir ? (1, Int(s)) : (Int(s), 0)
        (p >= 1 && (dir ? q >= 1 : q == 0)) || throw(ArgumentError(
            "a $(dir ? "directed hyperevent needs at least one sender and one " *
            "receiver" : "hyperevent needs at least one participant"), got size $s"))
        p + q <= n || throw(ArgumentError(
            "a hyperedge of size $s does not fit among $n actors"))
        push!(out, (p, q))
    end
    return out
end

"""
    simulate_hyperevents(statistics, coefficients, n_actors, n_events; sizes,
                         rng=Random.default_rng(), directed=nothing,
                         candidates=1000, actors=nothing)
        -> Vector{HyperEvent{Float64}}

Simulate a hyperevent sequence from a relational hyperevent model. At each step
the size of the next hyperedge is drawn uniformly from `sizes`, and then the
hyperedge itself is chosen among the hyperedges of that size with probability
proportional to `exp(θ′x)`, the statistics `x` being read off the hyperevents
simulated so far. Events occur at times `1, 2, …, n_events`.

- `sizes` — a vector of sizes (undirected hyperevents), or of
  `(senders, receivers)` size pairs (directed). With `directed=true` a plain
  integer `k` means one sender and `k` receivers.
- `candidates` — the choice is among **all** hyperedges of the drawn size when
  there are at most `candidates` of them. Otherwise it is among `candidates`
  hyperedges of that size sampled uniformly without replacement: an
  approximation, since a hyperedge outside the sampled set cannot be chosen at
  that step. Keep the network small, or raise `candidates`, when the simulation
  has to be exact (a recovery study).
- `actors` — the actor IDs at risk (default `1:n_actors`).

All randomness comes from `rng`. Cumulative statistics with positive
coefficients feed back on themselves; scale them (`transform=:log1p`) or use a
decaying memory.

# Example
```julia
using Revel, Random
stats = [SubsetRepetition(2; transform=:log1p)]
meetings = simulate_hyperevents(stats, [1.0], 6, 50; sizes=[2, 3, 3], rng=Xoshiro(1))
length(meetings), meetings[end].time                  # (50, 50.0)
mails = simulate_hyperevents([ReceiverSetRepetition(1; transform=:log1p)], [0.5], 6, 20;
                             sizes=[(1, 2), (1, 3)], rng=Xoshiro(1))
all(is_directed, mails)                               # true
```
"""
function simulate_hyperevents(statistics, coefficients::AbstractVector{<:Real},
                              n_actors::Int, n_events::Int; sizes,
                              rng::AbstractRNG=Random.default_rng(),
                              directed::Union{Nothing, Bool}=nothing,
                              candidates::Int=1000, actors=nothing)
    _hyper_stat_names(statistics)
    stats = Tuple(statistics)
    length(coefficients) == length(stats) || throw(ArgumentError(
        "$(length(coefficients)) coefficients for $(length(stats)) statistics"))
    n_events >= 0 || throw(ArgumentError("n_events must be non-negative"))
    candidates >= 2 || throw(ArgumentError("candidates must be at least 2"))
    θ = collect(Float64, coefficients)
    pool = _hyper_pool(actors, n_actors)
    shapes = _hyper_sizes(sizes, directed, length(pool))

    enumerated = Dict{Tuple{Int,Int}, Vector{_Hyperedge}}()
    sampled = _Hyperedge[]
    seen = Set{_Hyperedge}()
    work = copy(pool)
    η = Float64[]
    history = HyperHistory{Float64}()
    out = HyperEvent{Float64}[]
    sizehint!(out, n_events)

    for m in 1:n_events
        p, q = shapes[rand(rng, 1:length(shapes))]
        choice = if _n_hyperedges(length(pool), p, q, false) <= candidates
            get!(() -> _all_hyperedges(pool, p, q), enumerated, (p, q))
        else
            empty!(sampled); empty!(seen)
            _sample_hyperedges!(rng, sampled, seen, work, p, q, candidates, nothing)
        end
        t = Float64(m)
        resize!(η, length(choice))
        _hyper_predictor!(η, stats, θ, history, choice, t)
        d, _ = _sample_softmax(rng, η, length(choice))
        S, R = choice[d]
        ev = HyperEvent(S, R, t)
        update_hyper_history!(history, ev)
        push!(out, ev)
    end
    return out
end

function _hyper_predictor!(η::Vector{Float64}, stats::S, θ::Vector{Float64}, history,
                           choice::Vector{_Hyperedge}, t::Float64) where S<:Tuple
    @inbounds for (d, (senders, receivers)) in enumerate(choice)
        vals = map(stat -> compute(stat, history, senders, receivers, t), stats)
        acc = 0.0
        for k in eachindex(vals)
            acc += θ[k] * vals[k]
        end
        η[d] = acc
    end
    return η
end
