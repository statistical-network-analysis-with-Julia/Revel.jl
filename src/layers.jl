# =============================================================================
# Event layers: the remembered network an endogenous effect reads
# =============================================================================
#
# The review's main structural finding is that the REM literature contains about
# five configurations (the dyad, the reversed dyad, node degree, the two-path,
# the three-path) crossed with a few measurement choices: how past events are
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
# as Relevent.jl's `_AccumCache`.

mutable struct _LayerState
    source::WeakRef
    cursor::Int                          # events examined so far
    sig::Tuple{Int,Int,Float64}          # signature of the last examined event
    n::Int                               # actor capacity of the dense storage
    now::Float64                         # clock of the current evaluation
    snap_cursor::Int                     # cursor the snapshot was taken at
    snap_now::Float64                    # clock the snapshot was taken at
    count::Float64                       # events currently carrying weight
    mass::Float64                        # their total weight
    massL::Float64
    W::Matrix{Float64}                   # dyad weights
    L::Matrix{Float64}                   # last-update clock (half-life only)
    out::Vector{Float64}
    outL::Vector{Float64}
    inn::Vector{Float64}
    innL::Vector{Float64}
    first_t::Matrix{Float64}             # clock of the dyad's first event (NaN: none)
    last_t::Matrix{Float64}              # clock of the dyad's last event (NaN: none)
    last_i::Matrix{Int}                  # index of the dyad's last event (0: none)
    last_out_t::Vector{Float64}          # clock the actor last sent (NaN: never)
    last_in_t::Vector{Float64}           # clock the actor last received (NaN: never)
    out_nb::Vector{Vector{Int}}          # actors ever sent to
    in_nb::Vector{Vector{Int}}           # actors ever received from
    ev_s::Vector{Int}                    # accepted events, kept for the
    ev_r::Vector{Int}                    # non-accumulating kernels only
    ev_t::Vector{Float64}
    ev_w::Vector{Float64}
    touched::Vector{Int}                 # linear indices written by the snapshot
end

_LayerState(source) = _LayerState(
    WeakRef(source), 0, (0, 0, 0.0), 0, 0.0, -1, NaN, 0.0, 0.0, 0.0,
    zeros(0, 0), zeros(0, 0), Float64[], Float64[], Float64[], Float64[],
    fill(NaN, 0, 0), fill(NaN, 0, 0), zeros(Int, 0, 0), Float64[], Float64[],
    Vector{Int}[], Vector{Int}[], Int[], Int[], Float64[], Float64[], Int[])

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

A layer is mutable cache: do not share one `EventLayer`, or one statistic built
on it, between tasks that fit concurrently.

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
@inline _sig(e) = ((s, r, t, _, _) = _fields(e); (s, r, t))

function _state_for(layer::EventLayer, events::AbstractVector)
    states = layer.states
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
    layer.types === nothing || eltype(events) <: Event || throw(ArgumentError(
        "this statistic is restricted to event types $(layer.types), but the " *
        "history it is evaluated on (`REM.EventNetworkState.event_history`) does " *
        "not carry event types. Type-conditioned Revel statistics work with the " *
        "full-risk-set fitters (`fit_revel`, `Relevent.fit_obpm`), whose history " *
        "keeps the `Event`s."))
    st = _LayerState(events)
    push!(states, st)
    return st
end

function _reset!(st::_LayerState)
    st.cursor = 0
    st.sig = (0, 0, 0.0)
    st.snap_cursor = -1
    st.snap_now = NaN
    st.count = 0.0
    st.mass = 0.0
    st.massL = 0.0
    fill!(st.W, 0.0); fill!(st.L, 0.0)
    fill!(st.out, 0.0); fill!(st.outL, 0.0)
    fill!(st.inn, 0.0); fill!(st.innL, 0.0)
    fill!(st.first_t, NaN); fill!(st.last_t, NaN); fill!(st.last_i, 0)
    fill!(st.last_out_t, NaN); fill!(st.last_in_t, NaN)
    foreach(empty!, st.out_nb); foreach(empty!, st.in_nb)
    empty!(st.ev_s); empty!(st.ev_r); empty!(st.ev_t); empty!(st.ev_w)
    empty!(st.touched)
    return st
end

function _grow_matrix(A::Matrix{T}, n::Int, fillvalue::T) where T
    B = fill(fillvalue, n, n)
    m = size(A, 1)
    @inbounds for j in 1:m, i in 1:m
        B[i, j] = A[i, j]
    end
    return B
end

function _grow_vector!(v::Vector{T}, n::Int, fillvalue::T) where T
    m = length(v)
    resize!(v, n)
    @inbounds for i in (m + 1):n
        v[i] = fillvalue
    end
    return v
end

# Dense storage grows geometrically to the largest actor ID seen
function _grow!(st::_LayerState, needed::Int)
    n = max(needed, 2 * st.n, 8)
    old = st.n
    # The snapshot's touched list holds linear indices into the OLD matrix
    touched = [(CartesianIndices((old, old))[k][1], CartesianIndices((old, old))[k][2])
               for k in st.touched]
    st.W = _grow_matrix(st.W, n, 0.0)
    st.L = _grow_matrix(st.L, n, 0.0)
    st.first_t = _grow_matrix(st.first_t, n, NaN)
    st.last_t = _grow_matrix(st.last_t, n, NaN)
    st.last_i = _grow_matrix(st.last_i, n, 0)
    _grow_vector!(st.out, n, 0.0); _grow_vector!(st.outL, n, 0.0)
    _grow_vector!(st.inn, n, 0.0); _grow_vector!(st.innL, n, 0.0)
    _grow_vector!(st.last_out_t, n, NaN); _grow_vector!(st.last_in_t, n, NaN)
    for _ in (old + 1):n
        push!(st.out_nb, Int[]); push!(st.in_nb, Int[])
    end
    st.touched = [LinearIndices((n, n))[i, j] for (i, j) in touched]
    st.n = n
    return st
end

# One directed dyad of one accepted event. `x` is the event's weight, `tk` its
# clock, `k` its index in the history.
function _add_dyad!(layer::EventLayer, st::_LayerState, s::Int, r::Int, x::Float64,
                    tk::Float64, k::Int)
    @inbounds begin
        if st.last_i[s, r] == 0
            st.first_t[s, r] = tk
            push!(st.out_nb[s], r)
            push!(st.in_nb[r], s)
        end
        st.last_t[s, r] = tk
        st.last_i[s, r] = k
        st.last_out_t[s] = tk
        st.last_in_t[r] = tk
        if layer.memory isa FullMemory
            st.W[s, r] += x
            st.out[s] += x
            st.inn[r] += x
        elseif layer.memory isa HalfLife
            λ = layer.rate
            st.W[s, r] = st.W[s, r] * exp(-λ * (tk - st.L[s, r])) + x
            st.L[s, r] = tk
            st.out[s] = st.out[s] * exp(-λ * (tk - st.outL[s])) + x
            st.outL[s] = tk
            st.inn[r] = st.inn[r] * exp(-λ * (tk - st.innL[r])) + x
            st.innL[r] = tk
        end
    end
    return nothing
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
        st.mass = st.mass * exp(-layer.rate * (tk - st.massL)) + x
        st.massL = tk
    else
        push!(st.ev_s, s); push!(st.ev_r, r); push!(st.ev_t, tk); push!(st.ev_w, x)
    end
    _add_dyad!(layer, st, s, r, x, tk, k)
    layer.symmetric && s != r && _add_dyad!(layer, st, r, s, x, tk, k)
    return nothing
end

# Re-read the retained events at clock `now` (the non-accumulating kernels).
# Walks the history backwards and stops at the kernel's support, so a window
# costs the events inside it, not the whole history.
function _snapshot!(layer::EventLayer, st::_LayerState, now::Float64)
    W = st.W
    @inbounds for idx in st.touched
        W[idx] = 0.0
    end
    empty!(st.touched)
    fill!(st.out, 0.0); fill!(st.inn, 0.0)
    count = 0.0; mass = 0.0
    sup = _support(layer.memory)
    lin = LinearIndices(W)
    @inbounds for k in length(st.ev_t):-1:1
        age = now - st.ev_t[k]
        age > sup && break
        wgt = kernel_weight(layer.memory, age)
        wgt == 0.0 && continue
        x = wgt * st.ev_w[k]
        s, r = st.ev_s[k], st.ev_r[k]
        W[s, r] += x; push!(st.touched, lin[s, r])
        st.out[s] += x; st.inn[r] += x
        if layer.symmetric && s != r
            W[r, s] += x; push!(st.touched, lin[r, s])
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

# Bring the layer up to date with `events` and the evaluation clock `t`
function _sync!(layer::EventLayer, events::AbstractVector, t::Float64)
    st = _state_for(layer, events)
    n = length(events)
    if st.cursor > n || (st.cursor > 0 && _sig(@inbounds events[st.cursor]) != st.sig)
        _reset!(st)
    end
    if st.cursor < n
        for k in (st.cursor + 1):n
            _absorb!(layer, st, @inbounds(events[k]), k)
        end
        st.cursor = n
        st.sig = _sig(@inbounds events[n])
    end
    now = layer.clock === :order ? Float64(n + 1) : t
    st.now = now
    if !_accumulates(layer.memory) && (st.snap_cursor != n || st.snap_now != now)
        _snapshot!(layer, st, now)
    end
    return st
end

# -----------------------------------------------------------------------------
# Reads (all O(1); an actor beyond the storage has no history)
# -----------------------------------------------------------------------------

@inline function _w(layer::EventLayer{HalfLife}, st::_LayerState, i::Int, j::Int)
    (1 <= i <= st.n && 1 <= j <= st.n) || return 0.0
    @inbounds v = st.W[i, j]
    v == 0.0 && return 0.0
    return @inbounds v * exp(-layer.rate * (st.now - st.L[i, j])) * layer.norm
end
@inline function _w(::EventLayer, st::_LayerState, i::Int, j::Int)
    (1 <= i <= st.n && 1 <= j <= st.n) || return 0.0
    return @inbounds st.W[i, j]
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
    (1 <= i <= st.n && 1 <= j <= st.n) ? @inbounds(st.last_t[i, j]) : NaN
@inline _first_t(st::_LayerState, i::Int, j::Int) =
    (1 <= i <= st.n && 1 <= j <= st.n) ? @inbounds(st.first_t[i, j]) : NaN
@inline _last_i(st::_LayerState, i::Int, j::Int) =
    (1 <= i <= st.n && 1 <= j <= st.n) ? @inbounds(st.last_i[i, j]) : 0
