# =============================================================================
# Event layers: the remembered network an endogenous effect reads
# =============================================================================
#
# The review's main structural finding is that the REM literature contains four
# configurations (the dyad in either direction, node degree, the two-path, the
# three-path) crossed with a few measurement choices: how past events are
# weighted, which events feed the statistic, and whether event weights count.
# An `EventLayer` IS that set of measurement choices. It turns an event history
# into a weighted network w_t(i, j), and every endogenous effect is a function
# of one or more layers. The same configuration on a different layer is a
# different named effect in the literature (inertia vs. windowed inertia vs.
# typed inertia vs. a Brandes-style signed weight), but it is one piece of code
# here.
#
# A layer is synced lazily against the event vector it is evaluated on: events
# appended since the last evaluation are absorbed exactly once through a cursor
# (so a fit costs O(events), not O(events²), for the accumulating kernels), and
# the state is rebuilt if the vector was reset or rewritten — the same protocol
# as Relevent.jl's `_AccumCache`. Histories must be append-only or replayed from
# the start; the signatures of the first and the last absorbed event (sender,
# receiver, time, weight, type) are what detects a rewrite.
#
# Storage is sparse in the dyads: everything kept per dyad lives in vectors
# indexed by a "slot", one per dyad that has ever had an event, so memory grows
# with the number of distinct dyads, not with the square of the largest actor
# ID. A dyad finds its slot through an `Int32` matrix while actor IDs stay below
# `_DENSE_MAX` and through a hash map beyond. In the dense range the weights
# (and the half-life clocks) are also mirrored in `n × n` matrices, because the
# two-path statistics read them in their inner loop: at most 16 MiB for the slot
# index and 32 MiB per mirror.

const _NOSIG = (0, 0, 0.0, 0.0, :none)
const _DENSE_MAX = 2048

mutable struct _LayerState
    source::WeakRef
    cursor::Int                          # events examined so far
    sig::Tuple{Int,Int,Float64,Float64,Symbol}   # the last examined event
    sig1::Tuple{Int,Int,Float64,Float64,Symbol}  # the first one
    n::Int                               # actor capacity of the per-actor vectors
    dense::Bool                          # dyad → slot through `slotmat` or `slotdict`
    slotmat::Matrix{Int32}
    slotdict::Dict{Int,Int32}
    decays::Bool                         # a half-life layer: keep the clock mirror
    Wm::Matrix{Float64}                  # dense range only: W by (i, j)
    Lm::Matrix{Float64}                  # dense range, half-life only: L by (i, j)
    now::Float64                         # clock of the current evaluation
    synced_t::Float64                    # evaluation time of the last full check
    synced_len::Int                      # length of the history at that check
    synced_last::Tuple{Int,Int,Float64,Float64,Symbol}   # and its last event
    snap_cursor::Int                     # cursor the snapshot was taken at
    snap_now::Float64                    # clock the snapshot was taken at
    count::Float64                       # events currently carrying weight
    mass::Float64                        # their total weight
    massL::Float64
    # per dyad with a history, by slot
    sl_i::Vector{Int}
    sl_j::Vector{Int}
    W::Vector{Float64}                   # dyad weight
    L::Vector{Float64}                   # last-update clock (half-life only)
    first_t::Vector{Float64}             # clock of the dyad's first event
    last_t::Vector{Float64}              # clock of the dyad's last event
    last_i::Vector{Int}                  # index of the dyad's last event
    # per actor
    out::Vector{Float64}
    outL::Vector{Float64}
    inn::Vector{Float64}
    innL::Vector{Float64}
    last_out_t::Vector{Float64}          # clock the actor last sent (NaN: never)
    last_in_t::Vector{Float64}           # clock the actor last received (NaN: never)
    out_nb::Vector{Vector{Int}}          # actors ever sent to
    in_nb::Vector{Vector{Int}}           # actors ever received from
    # accepted events, kept for the non-accumulating kernels only
    ev_s::Vector{Int}
    ev_r::Vector{Int}
    ev_t::Vector{Float64}
    ev_w::Vector{Float64}
    ev_slot::Vector{Int}                 # slot of s → r
    ev_rslot::Vector{Int}                # slot of r → s on a symmetric layer
    touched::Vector{Int}                 # slots written by the snapshot
end

_LayerState(source, decays::Bool) = _LayerState(
    WeakRef(source), 0, _NOSIG, _NOSIG, 0, true, zeros(Int32, 0, 0), Dict{Int,Int32}(),
    decays, zeros(0, 0), zeros(0, 0),
    0.0, NaN, -1, _NOSIG, -1, NaN, 0.0, 0.0, 0.0,
    Int[], Int[], Float64[], Float64[], Float64[], Float64[], Int[],
    Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
    Vector{Int}[], Vector{Int}[],
    Int[], Int[], Float64[], Float64[], Int[], Int[], Int[])

"""
    EventLayer(; memory=FullMemory(), types=nothing, weighted=false, keep=nothing,
               clock=:time, symmetric=false)

The remembered network an endogenous effect reads: a rule turning the event
history into dyad weights `w_t(i, j)`.

- `memory` — how a past event's weight depends on its age: any
  [`AbstractMemory`](@ref) kernel.
- `types` — which **past** event types feed the layer (a `Symbol` or a collection
  of them); `nothing` takes every event. This is the type-split form of an
  interaction (signed events in Brandes, Lerner & Snijders 2009; remstats
  `consider_type = "separate"`; one attribute per type in eventnet).
- `weighted` — `true` adds each event's `weight`, `false` counts events.
- `keep` — a predicate `(sender, receiver, time, weight, eventtype) -> Bool`
  selecting the past events that count: the attribute-filtered form of an
  interaction (the `rem` package's `eventfiltervar`; Malang, Brandenberger &
  Leifeld 2019).
- `clock` — `:time` measures age in clock units; `:order` measures it in events
  (the age of the most recent event is 1), for order-only data and for half-lives
  stated in events.
- `symmetric` — `true` records each event in both directions, for undirected
  interaction (remstats' undirected statistics; goldfish coordination ties).

Every effect constructor takes these keywords directly and builds a private
layer; pass one shared `layer=` instead to let several effects read a single
index of the history.

A layer holds a mutable cache of the histories it has read. The package's entry
points (`fit_revel`, `each_risk_set`, `simulate_events`, the diagnostics, …)
work on private copies of the statistics they are given, so one specification
can be fitted from several tasks at once; calling `compute` by hand on one
statistic from several tasks concurrently is not safe. The copies also merge
layers with identical keywords, so `[Inertia(), Reciprocation(), OTP()]` index
the history once.

# Example
```julia
using Revel
recent = EventLayer(memory=HalfLife(30.0))
stats = [Inertia(layer=recent), Reciprocation(layer=recent), OTP(layer=recent)]
h = InteractionHistory()
update_history!(h, Event(1, 2, 0.0))
compute(stats[1], h, 1, 2, 30.0)     # 0.5 — one event, one half-life old
compute(stats[2], h, 2, 1, 30.0)     # 0.5 — the same event, seen from 2 → 1
```
"""
struct EventLayer{M<:AbstractMemory, K}
    memory::M
    types::Union{Nothing, Vector{Symbol}}
    weighted::Bool
    keep::K
    clock::Symbol
    symmetric::Bool
    rate::Float64        # half-life decay rate (0 otherwise)
    norm::Float64        # half-life normalisation (1 otherwise)
    states::Vector{_LayerState}
end

_types_arg(::Nothing) = nothing
_types_arg(t::Symbol) = [t]
_types_arg(t) = collect(Symbol, t)

function EventLayer(; memory::AbstractMemory=FullMemory(), types=nothing,
                    weighted::Bool=false, keep=nothing, clock::Symbol=:time,
                    symmetric::Bool=false)
    clock in (:time, :order) || throw(ArgumentError(
        "clock must be :time (age in clock units) or :order (age in events), got :$clock"))
    rate = memory isa HalfLife && isfinite(memory.halflife) ? log(2) / memory.halflife : 0.0
    norm = memory isa HalfLife && memory.normalized ? log(2) / memory.halflife : 1.0
    return EventLayer{typeof(memory), typeof(keep)}(
        memory, _types_arg(types), weighted, keep, clock, symmetric, rate, norm,
        _LayerState[])
end

function Base.show(io::IO, layer::EventLayer)
    label = _layer_label(layer)
    print(io, "EventLayer(", isempty(label) ? "full memory, all events" : label, ")")
end

# What distinguishes this layer from the default one, for statistic names
function _layer_label(layer::EventLayer)
    parts = String[]
    m = _memory_label(layer.memory)
    isempty(m) || push!(parts, m)
    layer.types === nothing || push!(parts, "types=" * join(layer.types, "+"))
    layer.weighted && push!(parts, "weighted")
    layer.keep === nothing || push!(parts, "filtered")
    layer.clock === :order && push!(parts, "clock=order")
    layer.symmetric && push!(parts, "symmetric")
    return join(parts, ",")
end

_suffix(layer::EventLayer) = (l = _layer_label(layer); isempty(l) ? "" : "[$l]")

# `layer=` wins; otherwise the layer keywords build a private one
function _resolve_layer(layer; kwargs...)
    layer === nothing && return EventLayer(; kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "pass either `layer=` or the layer keywords ($(join(keys(kwargs), ", "))), " *
        "not both: a shared layer already fixes them"))
    layer isa EventLayer || throw(ArgumentError("`layer` must be an EventLayer"))
    return layer
end

# -----------------------------------------------------------------------------
# Reading events: Relevent's history holds `Event{T}`, REM's state holds
# `(sender, receiver, time, weight)` tuples (no event type).
# -----------------------------------------------------------------------------

_tfloat(t::Real) = Float64(t)
_tfloat(t) = throw(ArgumentError(
    "Revel statistics need a numeric clock, got a time of type $(typeof(t)). " *
    "Convert calendar times to numbers (e.g. seconds since the first event) " *
    "before building the events."))

@inline _fields(e::Event) = (e.sender, e.receiver, _tfloat(e.time), e.weight, e.eventtype)
@inline _fields(e::Tuple) = (e[1], e[2], _tfloat(e[3]), Float64(e[4]), :event)
@inline _sig(e) = _fields(e)

function _state_for(layer::EventLayer, events::AbstractVector)
    states = layer.states
    # the common case: the history evaluated last time
    @inbounds !isempty(states) && states[1].source.value === events && return states[1]
    i = 1
    while i <= length(states)
        source = states[i].source.value
        if source === nothing
            deleteat!(states, i)             # its history was garbage-collected
        elseif source === events
            return states[i]
        else
            i += 1
        end
    end
    eltype(events) <: Event || _check_untyped_log(layer)
    st = _LayerState(events, layer.memory isa HalfLife)
    # the newest history first: it is the one the next call will ask for
    pushfirst!(states, st)
    return st
end

function _check_untyped_log(layer::EventLayer)
    fix = "Type-conditioned Revel statistics work with the full-risk-set fitters " *
          "(`fit_revel`, `Relevent.fit_obpm`), whose history keeps the `Event`s."
    layer.types === nothing || throw(ArgumentError(
        "this statistic is restricted to event types $(layer.types), but the " *
        "history it is evaluated on (`REM.EventNetworkState.event_history`) does " *
        "not carry event types. " * fix))
    layer.keep === nothing || throw(ArgumentError(
        "this statistic filters past events with `keep=`, whose predicate receives " *
        "the event type, but the history it is evaluated on " *
        "(`REM.EventNetworkState.event_history`) does not carry event types. " * fix))
    return nothing
end

# Entry points evaluate private copies of the statistics they are given (see
# `_fresh`): a copied layer starts with no cache, and layers with identical
# keywords become one, so a model whose effects share a measurement indexes the
# history once. The interning table lives in deepcopy's own `IdDict`, under a
# private key, so it spans exactly one copy.
const _INTERN_KEY = Ref(:revel_layers)

function Base.deepcopy_internal(layer::EventLayer{M,K}, seen::IdDict) where {M,K}
    haskey(seen, layer) && return seen[layer]::EventLayer{M,K}
    table = get!(() -> Dict{Any,Any}(), seen, _INTERN_KEY)::Dict{Any,Any}
    key = (layer.memory, layer.types, layer.weighted, layer.keep, layer.clock,
           layer.symmetric)
    copy_ = get!(table, key) do
        EventLayer{M,K}(layer.memory, layer.types === nothing ? nothing : copy(layer.types),
                        layer.weighted, layer.keep, layer.clock, layer.symmetric,
                        layer.rate, layer.norm, _LayerState[])
    end::EventLayer{M,K}
    seen[layer] = copy_
    return copy_
end

# A private, cache-free copy of a model specification
_fresh(stats) = deepcopy(stats)

function _reset!(st::_LayerState)
    st.cursor = 0
    st.sig = _NOSIG
    st.sig1 = _NOSIG
    st.snap_cursor = -1
    st.snap_now = NaN
    st.count = 0.0
    st.mass = 0.0
    st.massL = 0.0
    if st.dense
        @inbounds for k in eachindex(st.sl_i)
            i, j = st.sl_i[k], st.sl_j[k]
            st.slotmat[i, j] = 0
            st.Wm[i, j] = 0.0
            st.decays && (st.Lm[i, j] = 0.0)
        end
    else
        empty!(st.slotdict)
    end
    empty!(st.sl_i); empty!(st.sl_j); empty!(st.W); empty!(st.L)
    empty!(st.first_t); empty!(st.last_t); empty!(st.last_i)
    fill!(st.out, 0.0); fill!(st.outL, 0.0)
    fill!(st.inn, 0.0); fill!(st.innL, 0.0)
    fill!(st.last_out_t, NaN); fill!(st.last_in_t, NaN)
    foreach(empty!, st.out_nb); foreach(empty!, st.in_nb)
    empty!(st.ev_s); empty!(st.ev_r); empty!(st.ev_t); empty!(st.ev_w)
    empty!(st.ev_slot); empty!(st.ev_rslot)
    empty!(st.touched)
    return st
end

function _grow_vector!(v::Vector{T}, n::Int, fillvalue::T) where T
    m = length(v)
    resize!(v, n)
    @inbounds for i in (m + 1):n
        v[i] = fillvalue
    end
    return v
end

@inline _dkey(i::Int, j::Int) = (i << 32) | j

# Make room for actor `needed`: the per-actor vectors grow geometrically; the
# dyad → slot matrix grows with them up to `_DENSE_MAX` actors and is replaced
# by a hash map beyond.
function _grow!(st::_LayerState, needed::Int)
    needed < 2^31 || throw(ArgumentError("actor ID $needed is too large (at most 2^31 − 1)"))
    # geometric growth, by a quarter in the dense range (the index and the
    # mirrors are quadratic in it) and by half beyond
    n = max(needed, st.n + max(st.n ÷ (st.dense ? 4 : 2), 8))
    if st.dense
        if needed <= _DENSE_MAX
            n = min(n, _DENSE_MAX)
            M = zeros(Int32, n, n)
            Wm = zeros(n, n)
            Lm = st.decays ? zeros(n, n) : zeros(0, 0)
            @inbounds for k in eachindex(st.sl_i)
                i, j = st.sl_i[k], st.sl_j[k]
                M[i, j] = k
                Wm[i, j] = st.W[k]
                st.decays && (Lm[i, j] = st.L[k])
            end
            st.slotmat = M; st.Wm = Wm; st.Lm = Lm
        else
            sizehint!(st.slotdict, length(st.sl_i))
            @inbounds for k in eachindex(st.sl_i)
                st.slotdict[_dkey(st.sl_i[k], st.sl_j[k])] = k
            end
            st.slotmat = zeros(Int32, 0, 0)
            st.Wm = zeros(0, 0); st.Lm = zeros(0, 0)
            st.dense = false
        end
    end
    old = st.n
    _grow_vector!(st.out, n, 0.0); _grow_vector!(st.outL, n, 0.0)
    _grow_vector!(st.inn, n, 0.0); _grow_vector!(st.innL, n, 0.0)
    _grow_vector!(st.last_out_t, n, NaN); _grow_vector!(st.last_in_t, n, NaN)
    for _ in (old + 1):n
        push!(st.out_nb, Int[]); push!(st.in_nb, Int[])
    end
    st.n = n
    return st
end

# The slot of dyad i → j, 0 when it has no history
@inline function _slot(st::_LayerState, i::Int, j::Int)
    if st.dense
        (1 <= i <= st.n && 1 <= j <= st.n) || return 0
        return Int(@inbounds st.slotmat[i, j])
    end
    (1 <= i && 1 <= j) || return 0
    return Int(get(st.slotdict, _dkey(i, j), zero(Int32)))
end

function _new_slot!(st::_LayerState, i::Int, j::Int, tk::Float64)
    push!(st.sl_i, i); push!(st.sl_j, j)
    push!(st.W, 0.0); push!(st.L, 0.0)
    push!(st.first_t, tk); push!(st.last_t, NaN); push!(st.last_i, 0)
    sl = length(st.W)
    if st.dense
        @inbounds st.slotmat[i, j] = sl
    else
        st.slotdict[_dkey(i, j)] = sl
    end
    push!(st.out_nb[i], j)
    push!(st.in_nb[j], i)
    return sl
end

# A half-life accumulator after one more event of weight `x` at clock `tk`. An
# empty accumulator is not decayed: its "last update" clock is a placeholder,
# and on a far-negative clock `0 * exp(λ·|tk|)` would be `0 * Inf = NaN`.
@inline _decay_add(v::Float64, since::Float64, λ::Float64, tk::Float64, x::Float64) =
    v == 0.0 ? x : v * exp(-λ * (tk - since)) + x

# One directed dyad of one accepted event. `x` is the event's weight, `tk` its
# clock, `k` its index in the history. Returns the dyad's slot.
function _add_dyad!(layer::EventLayer, st::_LayerState, s::Int, r::Int, x::Float64,
                    tk::Float64, k::Int)
    sl = _slot(st, s, r)
    sl == 0 && (sl = _new_slot!(st, s, r, tk))
    @inbounds begin
        st.last_t[sl] = tk
        st.last_i[sl] = k
        st.last_out_t[s] = tk
        st.last_in_t[r] = tk
        if layer.memory isa FullMemory
            st.W[sl] += x
            st.out[s] += x
            st.inn[r] += x
        elseif layer.memory isa HalfLife
            λ = layer.rate
            st.W[sl] = _decay_add(st.W[sl], st.L[sl], λ, tk, x)
            st.L[sl] = tk
            st.out[s] = _decay_add(st.out[s], st.outL[s], λ, tk, x)
            st.outL[s] = tk
            st.inn[r] = _decay_add(st.inn[r], st.innL[r], λ, tk, x)
            st.innL[r] = tk
        end
        if st.dense
            st.Wm[s, r] = st.W[sl]
            st.decays && (st.Lm[s, r] = st.L[sl])
        end
    end
    return sl
end

function _absorb!(layer::EventLayer, st::_LayerState, e, k::Int)
    s, r, te, w, ty = _fields(e)
    # A non-positive ID is relevent's null actor (an event "to the group"):
    # it has no dyad, so it feeds no layer
    (s >= 1 && r >= 1) || return nothing
    layer.types === nothing || ty in layer.types || return nothing
    layer.keep === nothing || layer.keep(s, r, te, w, ty)::Bool || return nothing

    tk = layer.clock === :order ? Float64(k) : te
    x = layer.weighted ? w : 1.0
    max(s, r) > st.n && _grow!(st, max(s, r))

    if layer.memory isa FullMemory
        st.count += 1.0
        st.mass += x
    elseif layer.memory isa HalfLife
        st.count += 1.0
        st.mass = _decay_add(st.mass, st.massL, layer.rate, tk, x)
        st.massL = tk
    end
    sl = _add_dyad!(layer, st, s, r, x, tk, k)
    rsl = layer.symmetric && s != r ? _add_dyad!(layer, st, r, s, x, tk, k) : 0
    if !_accumulates(layer.memory)
        push!(st.ev_s, s); push!(st.ev_r, r); push!(st.ev_t, tk); push!(st.ev_w, x)
        push!(st.ev_slot, sl); push!(st.ev_rslot, rsl)
    end
    return nothing
end

# Re-read the retained events at clock `now` (the non-accumulating kernels).
# Walks the history backwards and stops at the kernel's support, so a window
# costs the events inside it, not the whole history.
function _snapshot!(layer::EventLayer, st::_LayerState, now::Float64)
    W = st.W
    dense = st.dense
    @inbounds for sl in st.touched
        W[sl] = 0.0
        dense && (st.Wm[st.sl_i[sl], st.sl_j[sl]] = 0.0)
    end
    empty!(st.touched)
    fill!(st.out, 0.0); fill!(st.inn, 0.0)
    count = 0.0; mass = 0.0
    sup = _support(layer.memory)
    @inbounds for k in length(st.ev_t):-1:1
        age = now - st.ev_t[k]
        age > sup && break
        wgt = kernel_weight(layer.memory, age)
        wgt == 0.0 && continue
        x = wgt * st.ev_w[k]
        s, r = st.ev_s[k], st.ev_r[k]
        sl = st.ev_slot[k]
        W[sl] += x; push!(st.touched, sl)
        dense && (st.Wm[s, r] = W[sl])
        st.out[s] += x; st.inn[r] += x
        rsl = st.ev_rslot[k]
        if rsl != 0
            W[rsl] += x; push!(st.touched, rsl)
            dense && (st.Wm[r, s] = W[rsl])
            st.out[r] += x; st.inn[s] += x
        end
        count += 1.0; mass += x
    end
    st.count = count
    st.mass = mass
    st.snap_cursor = st.cursor
    st.snap_now = now
    return st
end

@inline _event_time(e) = _fields(e)[3]

# How many events of the time-ordered `events` happened at or before `t`: the
# past a statistic evaluated at `t` may read. Inside the fitters that is every
# event of the history; a history passed by hand may run past `t`.
function _n_before(events::AbstractVector, t::Float64)
    isnan(t) && throw(ArgumentError("the evaluation time is NaN"))
    n = length(events)
    (n == 0 || _event_time(@inbounds events[n]) <= t) && return n
    lo, hi = 0, n                     # invariant: events[1:lo] are <= t, events[hi] > t
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        _event_time(@inbounds events[mid]) <= t ? (lo = mid) : (hi = mid)
    end
    # the first event left out must be genuinely later, not a NaN that compares
    # false with everything (the events kept are checked as they are absorbed)
    _check_order(events, lo + 1)
    return lo
end

function _check_order(events::AbstractVector, k::Int)
    tk = _event_time(@inbounds events[k])
    isfinite(tk) || throw(ArgumentError(
        "event $k of the history has time $tk; event times must be finite"))
    k > 1 && tk < _event_time(@inbounds events[k - 1]) && throw(ArgumentError(
        "the history is not in time order: event $k (t = $tk) comes after an " *
        "event at t = $(_event_time(@inbounds events[k - 1])). Sort the events " *
        "by time before building the history."))
    return nothing
end

# Bring the layer up to date with the events of `events` that happened at or
# before the evaluation clock `t`
function _sync!(layer::EventLayer, events::AbstractVector, t::Float64)
    st = _state_for(layer, events)
    # Evaluated again at the same time on an unchanged history — every dyad of
    # one risk set: nothing to absorb, nothing to re-check
    len = length(events)
    if t == st.synced_t && len == st.synced_len &&
       (len == 0 || isequal(_sig(@inbounds events[len]), st.synced_last))
        return st
    end
    n = _n_before(events, t)
    if st.cursor > n || (st.cursor > 0 &&
                         (!isequal(_sig(@inbounds events[st.cursor]), st.sig) ||
                          !isequal(_sig(@inbounds events[1]), st.sig1)))
        _reset!(st)
    end
    if st.cursor < n
        for k in (st.cursor + 1):n
            _check_order(events, k)
            _absorb!(layer, st, @inbounds(events[k]), k)
        end
        st.cursor == 0 && (st.sig1 = _sig(@inbounds events[1]))
        st.cursor = n
        st.sig = _sig(@inbounds events[n])
    end
    now = layer.clock === :order ? Float64(n + 1) : t
    st.now = now
    if !_accumulates(layer.memory) && (st.snap_cursor != n || st.snap_now != now)
        _snapshot!(layer, st, now)
    end
    st.synced_t = t
    st.synced_len = len
    len > 0 && (st.synced_last = _sig(@inbounds events[len]))
    return st
end

# -----------------------------------------------------------------------------
# Reads (all O(1); a dyad or actor beyond the storage has no history)
# -----------------------------------------------------------------------------

@inline function _w(layer::EventLayer{HalfLife}, st::_LayerState, i::Int, j::Int)
    Wm = st.Wm
    m = size(Wm, 1)                          # 0 outside the dense range
    if 1 <= i <= m && 1 <= j <= m
        @inbounds v = Wm[i, j]
        v == 0.0 && return 0.0
        return @inbounds v * exp(-layer.rate * (st.now - st.Lm[i, j])) * layer.norm
    end
    st.dense && return 0.0
    sl = _slot(st, i, j)
    sl == 0 && return 0.0
    @inbounds v = st.W[sl]
    v == 0.0 && return 0.0
    return @inbounds v * exp(-layer.rate * (st.now - st.L[sl])) * layer.norm
end
@inline function _w(::EventLayer, st::_LayerState, i::Int, j::Int)
    Wm = st.Wm
    m = size(Wm, 1)                          # 0 outside the dense range
    (1 <= i <= m && 1 <= j <= m) && return @inbounds Wm[i, j]
    st.dense && return 0.0
    sl = _slot(st, i, j)
    return sl == 0 ? 0.0 : @inbounds(st.W[sl])
end

@inline function _outdeg(layer::EventLayer{HalfLife}, st::_LayerState, i::Int)
    1 <= i <= st.n || return 0.0
    @inbounds v = st.out[i]
    v == 0.0 && return 0.0
    return @inbounds v * exp(-layer.rate * (st.now - st.outL[i])) * layer.norm
end
@inline _outdeg(::EventLayer, st::_LayerState, i::Int) =
    1 <= i <= st.n ? @inbounds(st.out[i]) : 0.0

@inline function _indeg(layer::EventLayer{HalfLife}, st::_LayerState, i::Int)
    1 <= i <= st.n || return 0.0
    @inbounds v = st.inn[i]
    v == 0.0 && return 0.0
    return @inbounds v * exp(-layer.rate * (st.now - st.innL[i])) * layer.norm
end
@inline _indeg(::EventLayer, st::_LayerState, i::Int) =
    1 <= i <= st.n ? @inbounds(st.inn[i]) : 0.0

@inline _mass(layer::EventLayer{HalfLife}, st::_LayerState) =
    st.mass == 0.0 ? 0.0 : st.mass * exp(-layer.rate * (st.now - st.massL)) * layer.norm
@inline _mass(::EventLayer, st::_LayerState) = st.mass

const _NO_NEIGHBORS = Int[]
@inline _out_nb(st::_LayerState, i::Int) = 1 <= i <= st.n ? @inbounds(st.out_nb[i]) : _NO_NEIGHBORS
@inline _in_nb(st::_LayerState, i::Int) = 1 <= i <= st.n ? @inbounds(st.in_nb[i]) : _NO_NEIGHBORS

# Distinct partners currently carrying weight (the "degree" rather than the
# "intensity" of Vu, Lomi, Mascia & Pallotti 2017)
function _out_partners(layer::EventLayer, st::_LayerState, i::Int)
    c = 0.0
    for k in _out_nb(st, i)
        _w(layer, st, i, k) > 0 && (c += 1.0)
    end
    return c
end
function _in_partners(layer::EventLayer, st::_LayerState, i::Int)
    c = 0.0
    for k in _in_nb(st, i)
        _w(layer, st, k, i) > 0 && (c += 1.0)
    end
    return c
end

@inline _last_t(st::_LayerState, i::Int, j::Int) =
    (sl = _slot(st, i, j); sl == 0 ? NaN : @inbounds(st.last_t[sl]))
@inline _first_t(st::_LayerState, i::Int, j::Int) =
    (sl = _slot(st, i, j); sl == 0 ? NaN : @inbounds(st.first_t[sl]))
@inline _last_i(st::_LayerState, i::Int, j::Int) =
    (sl = _slot(st, i, j); sl == 0 ? 0 : @inbounds(st.last_i[sl]))
