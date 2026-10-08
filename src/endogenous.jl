# =============================================================================
# Endogenous effects: five configurations on a layer
# =============================================================================
#
#   DyadEffect        the dyad and the reversed dyad    (inertia, reciprocity)
#   DegreeEffect      node degree                        (activity, popularity)
#   DyadDegreeEffect  two node degrees combined          (degree sum/min/max/diff)
#   TwoPathEffect     the two-path                       (OTP, ITP, OSP, ISP, balance)
#   FourCycleEffect   the three-path                     (four-cycle closure)
#
# plus the order-based devices that do not read a weight at all: recency ranks,
# time since the last event, and participation shifts.
#
# The named constructors (`Inertia`, `OTP`, …) are the vocabulary of the
# literature; the parametric types are what they all build. `effect_catalogue()`
# maps each name onto relevent, remstats, rem, goldfish and eventnet.

const _LAYER_KEYS = (:memory, :types, :weighted, :keep, :clock, :symmetric)

# Split constructor keywords into the layer's and the effect's
function _split_layer_kwargs(kwargs)
    layer_kw = Pair{Symbol,Any}[]
    rest = Pair{Symbol,Any}[]
    for (k, v) in kwargs
        push!(k in _LAYER_KEYS ? layer_kw : rest, k => v)
    end
    return layer_kw, rest
end

function _no_extra_kwargs(rest, what::AbstractString)
    isempty(rest) || throw(ArgumentError(
        "$what: unknown keyword$(length(rest) == 1 ? "" : "s") " *
        "$(join((":" * string(first(p)) for p in rest), ", "))"))
    return nothing
end

# -----------------------------------------------------------------------------
# The dyad and the reversed dyad
# -----------------------------------------------------------------------------

const _DENOMINATORS = (:sender_out, :sender_in, :receiver_out, :receiver_in)

"""
    DyadEffect(layer; direction=:out, scaling=:none, denominator=:sender_out,
               empty=0.0, transform=identity, name=nothing)

The weight a [`EventLayer`](@ref) puts on the candidate dyad `s → r`
(`direction=:out`), on the reversed dyad `r → s` (`:in`), or on both (`:sym`).
[`Inertia`](@ref), [`Reciprocation`](@ref) and [`DyadActivity`](@ref) are its
named forms.

`scaling=:prop` divides by an actor's degree — the `denominator`, one of
`:sender_out`, `:sender_in`, `:receiver_out`, `:receiver_in` — and returns
`empty` when that degree is zero. The zero-history value is a convention that
differs by package (relevent and remstats use `1/(n−1)`), so it is a keyword
here, not a hidden default.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 2, 3.0)])
raw = DyadEffect(EventLayer())
share = DyadEffect(EventLayer(); scaling=:prop, empty=0.5)
compute(raw, h, 1, 2, 4.0)      # 2.0
compute(share, h, 1, 2, 4.0)    # 0.666… — two of actor 1's three sends
compute(share, h, 3, 1, 4.0)    # 0.5 — actor 3 has sent nothing: `empty`
```
"""
struct DyadEffect{L<:EventLayer, F} <: AbstractRevelStatistic
    layer::L
    direction::Symbol
    scaling::Symbol
    denominator::Symbol
    empty::Float64
    transform::F
    label::String
end

function DyadEffect(layer::EventLayer; direction::Symbol=:out, scaling::Symbol=:none,
                    denominator::Symbol=:sender_out, empty::Real=0.0,
                    transform=identity, name=nothing, base::AbstractString="dyad")
    direction in (:out, :in, :sym) || throw(ArgumentError(
        "direction must be :out (s → r), :in (r → s) or :sym (both), got :$direction"))
    scaling in (:none, :prop) || throw(ArgumentError(
        "scaling must be :none or :prop, got :$scaling (wrap the statistic in " *
        "`Standardized` for remstats' \"std\")"))
    denominator in _DENOMINATORS || throw(ArgumentError(
        "denominator must be one of $(_DENOMINATORS), got :$denominator"))
    f = _transform_fn(transform)
    auto = _auto_name(base * (scaling === :prop ? ".prop" : ""), _suffix(layer), f)
    return DyadEffect{typeof(layer), typeof(f)}(layer, direction, scaling, denominator,
                                                Float64(empty), f, _label(name, auto))
end

function _value(stat::DyadEffect, events, s::Int, r::Int, t::Float64)
    L = stat.layer
    st = _sync!(L, events, t)
    v = stat.direction === :out ? _w(L, st, s, r) :
        stat.direction === :in  ? _w(L, st, r, s) :
                                  _w(L, st, s, r) + _w(L, st, r, s)
    if stat.scaling === :prop
        d = stat.denominator === :sender_out   ? _outdeg(L, st, s) :
            stat.denominator === :sender_in    ? _indeg(L, st, s) :
            stat.denominator === :receiver_out ? _outdeg(L, st, r) :
                                                 _indeg(L, st, r)
        v = d > 0 ? v / d : stat.empty
    end
    return Float64(stat.transform(v))
end

_interval_constant(stat::DyadEffect) = _layer_interval_constant(stat.layer)

"""
    Inertia(; scaling=:none, empty=0.0, transform=identity, name=nothing,
            layer=nothing, memory=FullMemory(), types=nothing, weighted=false,
            keep=nothing, clock=:time, symmetric=false)

Repetition: the weight of past `s → r` events (Brandes, Lerner & Snijders 2009
"inertia"; remstats `inertia()`; rem `inertiaStat`; goldfish
`inertia(weighted=TRUE)`; eventnet "repetition").

`scaling=:prop` is Butts's (2008) "persistence" — the share of the sender's past
sends that went to this receiver (remstats `inertia(scaling="prop")`, the
"embedding inertia" of Kitts et al. 2017, and relevent's documented `FrPSndSnd`,
whose output in relevent 1.2.1 differs from its documentation); pass
`empty=1/(n-1)` for the value those packages give a sender with no history.
`transform=:indicator` gives goldfish's default 0/1 form.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 2, 3.0)])
compute(Inertia(), h, 1, 2, 4.0)                         # 2.0
compute(Inertia(scaling=:prop), h, 1, 2, 4.0)            # 0.666…
compute(Inertia(memory=HalfLife(1.0)), h, 1, 2, 4.0)     # 0.5 + 0.125 = 0.625
compute(Inertia(transform=:indicator), h, 1, 3, 4.0)     # 1.0
```
"""
function Inertia(; layer=nothing, scaling::Symbol=:none, empty::Real=0.0,
                 transform=identity, name=nothing, kwargs...)
    layer_kw, rest = _split_layer_kwargs(kwargs)
    _no_extra_kwargs(rest, "Inertia")
    return DyadEffect(_resolve_layer(layer; layer_kw...); direction=:out,
                      scaling=scaling, denominator=:sender_out, empty=empty,
                      transform=transform, name=name, base="inertia")
end

"""
    Reciprocation(; scaling=:none, denominator=:sender_in, empty=0.0,
                  transform=identity, name=nothing, layer=nothing, memory=…, …)

Reciprocity: the weight of past `r → s` events on a candidate `s → r` (Brandes,
Lerner & Snijders 2009; remstats `reciprocity()`; rem `reciprocityStat`; goldfish
`recip`). Named `Reciprocation` because REM.jl exports a `Reciprocity` statistic
with a different (eventnet) parameterisation.

`scaling=:prop` divides by a degree chosen with `denominator`: `:sender_in` (the
default) is the share of the sender's past *receipts* that came from this
receiver — relevent `FrRecSnd`, remstats `reciprocity(scaling="prop")`, the
"dependence reciprocation" of Kitts et al. (2017); `:receiver_out` is their
"embedding reciprocation", the share of the receiver's past sends that went to
the sender.

# Example
```julia
using Revel
h = build_history([Event(2, 1, 1.0), Event(3, 1, 2.0), Event(2, 1, 3.0)])
compute(Reciprocation(), h, 1, 2, 4.0)                    # 2.0
compute(Reciprocation(scaling=:prop), h, 1, 2, 4.0)       # 0.666…
compute(Reciprocation(scaling=:prop, denominator=:receiver_out), h, 1, 2, 4.0)   # 1.0
```
"""
function Reciprocation(; layer=nothing, scaling::Symbol=:none,
                       denominator::Symbol=:sender_in, empty::Real=0.0,
                       transform=identity, name=nothing, kwargs...)
    layer_kw, rest = _split_layer_kwargs(kwargs)
    _no_extra_kwargs(rest, "Reciprocation")
    return DyadEffect(_resolve_layer(layer; layer_kw...); direction=:in,
                      scaling=scaling, denominator=denominator, empty=empty,
                      transform=transform, name=name, base="reciprocity")
end

"""
    DyadActivity(; transform=identity, name=nothing, layer=nothing, memory=…, …)

The weight of past events between the two actors in **either** direction
(eventnet `DYAD_STATISTIC` with direction `SYM`). For undirected data build the layer with
`symmetric=true` and use [`Inertia`](@ref) instead.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(2, 1, 2.0)])
compute(DyadActivity(), h, 1, 2, 3.0)    # 2.0
```
"""
function DyadActivity(; layer=nothing, transform=identity, name=nothing, kwargs...)
    layer_kw, rest = _split_layer_kwargs(kwargs)
    _no_extra_kwargs(rest, "DyadActivity")
    return DyadEffect(_resolve_layer(layer; layer_kw...); direction=:sym,
                      transform=transform, name=name, base="dyad_activity")
end

# -----------------------------------------------------------------------------
# Node degree
# -----------------------------------------------------------------------------

# A symmetric layer records every event in both directions, so an actor's out-
# and in-degree are the same number — the events it took part in — and its
# "total" degree is that number once, not twice.
@inline function _actor_degree(L::EventLayer, st::_LayerState, a::Int, kind::Symbol,
                               measure::Symbol)
    one_sided = L.symmetric && kind === :total
    if measure === :intensity
        # events per distinct partner (0 with no partner)
        m = _actor_degree(L, st, a, kind, :partners)
        return m > 0 ? _actor_degree(L, st, a, kind, :events) / m : 0.0
    end
    if measure === :events
        return (kind === :out || one_sided) ? _outdeg(L, st, a) :
               kind === :in ? _indeg(L, st, a) :
                              _outdeg(L, st, a) + _indeg(L, st, a)
    end
    return (kind === :out || one_sided) ? _out_partners(L, st, a) :
           kind === :in ? _in_partners(L, st, a) :
                          _out_partners(L, st, a) + _in_partners(L, st, a)
end

# What `scaling=:prop` divides a degree by: the events in memory, twice for a
# total degree on a directed layer (each event adds to one out- and one in-degree)
@inline _degree_share_base(L::EventLayer, st::_LayerState, kind::Symbol) =
    _mass(L, st) * ((kind === :total && !L.symmetric) ? 2.0 : 1.0)

function _check_degree_args(kind::Symbol, measure::Symbol, scaling::Symbol)
    kind in (:out, :in, :total) || throw(ArgumentError(
        "kind must be :out, :in or :total, got :$kind"))
    measure in (:events, :partners, :intensity) || throw(ArgumentError(
        "measure must be :events (event volume), :partners (distinct partners, the " *
        "\"degree\" of Vu et al. 2017) or :intensity (events per partner, their " *
        "\"intensity\"), got :$measure"))
    scaling in (:none, :prop) || throw(ArgumentError(
        "scaling must be :none or :prop, got :$scaling"))
    scaling === :prop && measure !== :events && throw(ArgumentError(
        "scaling=:prop divides event volume by the number of past events; it is " *
        "not defined for measure=:$measure"))
    return nothing
end

"""
    DegreeEffect(layer; role=:sender, kind=:out, measure=:events, scaling=:none,
                 empty=0.0, transform=identity, name=nothing)

A degree of the candidate's sender (`role=:sender`) or receiver (`:receiver`):
events sent (`kind=:out`), received (`:in`) or both (`:total`). The six named
forms are [`OutdegreeSender`](@ref), [`IndegreeSender`](@ref),
[`TotaldegreeSender`](@ref), [`OutdegreeReceiver`](@ref),
[`IndegreeReceiver`](@ref) and [`TotaldegreeReceiver`](@ref).

On a `symmetric=true` layer (undirected events) the three kinds coincide: each is
the number of events the actor took part in.

`measure=:partners` counts distinct partners instead of events, and
`measure=:intensity` divides the events by the partners — the "degree" and the
"intensity" (events per collaboration tie) of Vu, Lomi, Mascia & Pallotti
(2017, appendix eq. 2); `measure=:events` (default) is plain event volume.
`scaling=:prop` divides event volume by the number of past events (twice
that for `:total`), the normalised degree of relevent's `NIDSnd` family and
remstats' `scaling="prop"`; `empty` is returned before any event.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 2, 3.0)])
compute(DegreeEffect(EventLayer()), h, 1, 2, 4.0)                        # 3.0
compute(DegreeEffect(EventLayer(); measure=:partners), h, 1, 2, 4.0)     # 2.0
compute(DegreeEffect(EventLayer(); measure=:intensity), h, 1, 2, 4.0)    # 1.5
compute(DegreeEffect(EventLayer(); scaling=:prop), h, 1, 2, 4.0)         # 1.0
```
"""
struct DegreeEffect{L<:EventLayer, F} <: AbstractRevelStatistic
    layer::L
    role::Symbol
    kind::Symbol
    measure::Symbol
    scaling::Symbol
    empty::Float64
    transform::F
    label::String
end

function DegreeEffect(layer::EventLayer; role::Symbol=:sender, kind::Symbol=:out,
                      measure::Symbol=:events, scaling::Symbol=:none,
                      empty::Real=0.0, transform=identity, name=nothing)
    role in (:sender, :receiver) || throw(ArgumentError(
        "role must be :sender or :receiver, got :$role"))
    _check_degree_args(kind, measure, scaling)
    f = _transform_fn(transform)
    base = (kind === :total ? "totaldegree" : string(kind) * "degree") *
           (role === :sender ? "Sender" : "Receiver") *
           (measure === :events ? "" : ".$(measure)") *
           (scaling === :prop ? ".prop" : "")
    return DegreeEffect{typeof(layer), typeof(f)}(
        layer, role, kind, measure, scaling, Float64(empty), f,
        _label(name, _auto_name(base, _suffix(layer), f)))
end

function _value(stat::DegreeEffect, events, s::Int, r::Int, t::Float64)
    L = stat.layer
    st = _sync!(L, events, t)
    v = _actor_degree(L, st, stat.role === :sender ? s : r, stat.kind, stat.measure)
    if stat.scaling === :prop
        m = _degree_share_base(L, st, stat.kind)
        v = m > 0 ? v / m : stat.empty
    end
    return Float64(stat.transform(v))
end

_interval_constant(stat::DegreeEffect) = _layer_interval_constant(stat.layer)

function _degree_constructor(what::String, role::Symbol, kind::Symbol; layer, kwargs...)
    layer_kw, rest = _split_layer_kwargs(kwargs)
    return DegreeEffect(_resolve_layer(layer; layer_kw...); role=role, kind=kind, rest...)
end

"""
    OutdegreeSender(; measure=:events, scaling=:none, empty=0.0,
                    transform=identity, name=nothing, layer=nothing, memory=…, …)

Sender activity: events the candidate's sender has sent (remstats
`outdegreeSender()`; relevent `NODSnd` with `scaling=:prop`; rem `degreeStat`
sender-outdegree; goldfish `outdeg(type="ego")`). See [`DegreeEffect`](@ref).

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 1, 3.0)])
compute(OutdegreeSender(), h, 1, 2, 4.0)    # 2.0
```
"""
OutdegreeSender(; layer=nothing, kwargs...) =
    _degree_constructor("OutdegreeSender", :sender, :out; layer=layer, kwargs...)

"""
    IndegreeSender(; measure=:events, scaling=:none, empty=0.0, …)

Events the candidate's sender has received (remstats `indegreeSender()`;
relevent `NIDSnd` with `scaling=:prop`; goldfish `indeg(type="ego")`). See
[`DegreeEffect`](@ref).

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 1, 3.0)])
compute(IndegreeSender(), h, 1, 2, 4.0)    # 1.0
```
"""
IndegreeSender(; layer=nothing, kwargs...) =
    _degree_constructor("IndegreeSender", :sender, :in; layer=layer, kwargs...)

"""
    TotaldegreeSender(; measure=:events, scaling=:none, empty=0.0, …)

Events the candidate's sender has sent or received (remstats
`totaldegreeSender()`; relevent `NTDegSnd` with `scaling=:prop`). See
[`DegreeEffect`](@ref).

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 1, 3.0)])
compute(TotaldegreeSender(), h, 1, 2, 4.0)    # 3.0
```
"""
TotaldegreeSender(; layer=nothing, kwargs...) =
    _degree_constructor("TotaldegreeSender", :sender, :total; layer=layer, kwargs...)

"""
    OutdegreeReceiver(; measure=:events, scaling=:none, empty=0.0, …)

Events the candidate's receiver has sent (remstats `outdegreeReceiver()`;
relevent `NODRec` with `scaling=:prop`; goldfish `outdeg(type="alter")`). See
[`DegreeEffect`](@ref).

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 1, 3.0)])
compute(OutdegreeReceiver(), h, 1, 2, 4.0)    # 1.0
```
"""
OutdegreeReceiver(; layer=nothing, kwargs...) =
    _degree_constructor("OutdegreeReceiver", :receiver, :out; layer=layer, kwargs...)

"""
    IndegreeReceiver(; measure=:events, scaling=:none, empty=0.0, …)

Receiver popularity: events the candidate's receiver has received (remstats
`indegreeReceiver()`; relevent `NIDRec` with `scaling=:prop`; rem `degreeStat`
target-indegree; goldfish `indeg(type="alter")`). With `scaling=:prop` on the
`:total` kind this is Butts's (2008) preferential attachment — see
[`TotaldegreeReceiver`](@ref).

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(3, 2, 2.0), Event(2, 1, 3.0)])
compute(IndegreeReceiver(), h, 1, 2, 4.0)    # 2.0
```
"""
IndegreeReceiver(; layer=nothing, kwargs...) =
    _degree_constructor("IndegreeReceiver", :receiver, :in; layer=layer, kwargs...)

"""
    TotaldegreeReceiver(; measure=:events, scaling=:none, empty=0.0, …)

Events the candidate's receiver has sent or received (remstats
`totaldegreeReceiver()`). With `scaling=:prop` it is the preferential-attachment
statistic of Butts (2008) — the receiver's share of all past volume, relevent
`NTDegRec`.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(3, 2, 2.0), Event(2, 1, 3.0)])
compute(TotaldegreeReceiver(), h, 1, 2, 4.0)                  # 3.0
compute(TotaldegreeReceiver(scaling=:prop), h, 1, 2, 4.0)     # 0.5
```
"""
TotaldegreeReceiver(; layer=nothing, kwargs...) =
    _degree_constructor("TotaldegreeReceiver", :receiver, :total; layer=layer, kwargs...)

const _DEGREE_COMBINE = (:sum, :min, :max, :absdiff, :product)

"""
    DyadDegreeEffect(layer; sender_kind=:total, receiver_kind=:total, combine=:sum,
                     measure=:events, scaling=:none, empty=0.0,
                     transform=identity, name=nothing)

Two actor degrees combined into one dyad statistic: their `:sum`, `:min`,
`:max`, absolute difference (`:absdiff`) or `:product`. The named forms are
[`TotaldegreeDyad`](@ref), [`DegreeMin`](@ref), [`DegreeMax`](@ref),
[`DegreeDiff`](@ref) and [`DegreeAssortativity`](@ref). With `scaling=:prop` each
degree is divided by the number of past events (twice that for `:total`) before
combining.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 3, 3.0)])
compute(DyadDegreeEffect(EventLayer(); combine=:absdiff), h, 1, 3, 4.0)   # 0.0
compute(DyadDegreeEffect(EventLayer(); sender_kind=:out, receiver_kind=:in,
                         combine=:product), h, 1, 3, 4.0)                 # 4.0
```
"""
struct DyadDegreeEffect{L<:EventLayer, F} <: AbstractRevelStatistic
    layer::L
    sender_kind::Symbol
    receiver_kind::Symbol
    combine::Symbol
    measure::Symbol
    scaling::Symbol
    empty::Float64
    transform::F
    label::String
end

function DyadDegreeEffect(layer::EventLayer; sender_kind::Symbol=:total,
                          receiver_kind::Symbol=:total, combine::Symbol=:sum,
                          measure::Symbol=:events, scaling::Symbol=:none,
                          empty::Real=0.0, transform=identity, name=nothing,
                          base::Union{Nothing,AbstractString}=nothing)
    _check_degree_args(sender_kind, measure, scaling)
    _check_degree_args(receiver_kind, measure, scaling)
    combine in _DEGREE_COMBINE || throw(ArgumentError(
        "combine must be one of $(_DEGREE_COMBINE), got :$combine"))
    f = _transform_fn(transform)
    b = something(base, "degree_$(combine)($(sender_kind),$(receiver_kind))") *
        (measure === :events ? "" : ".$(measure)") * (scaling === :prop ? ".prop" : "")
    return DyadDegreeEffect{typeof(layer), typeof(f)}(
        layer, sender_kind, receiver_kind, combine, measure, scaling, Float64(empty), f,
        _label(name, _auto_name(b, _suffix(layer), f)))
end

function _value(stat::DyadDegreeEffect, events, s::Int, r::Int, t::Float64)
    L = stat.layer
    st = _sync!(L, events, t)
    a = _actor_degree(L, st, s, stat.sender_kind, stat.measure)
    b = _actor_degree(L, st, r, stat.receiver_kind, stat.measure)
    if stat.scaling === :prop
        _mass(L, st) > 0 || return Float64(stat.transform(stat.empty))
        a /= _degree_share_base(L, st, stat.sender_kind)
        b /= _degree_share_base(L, st, stat.receiver_kind)
    end
    c = stat.combine
    v = c === :sum ? a + b : c === :min ? min(a, b) : c === :max ? max(a, b) :
        c === :absdiff ? abs(a - b) : a * b
    return Float64(stat.transform(v))
end

_interval_constant(stat::DyadDegreeEffect) = _layer_interval_constant(stat.layer)

function _dyad_degree_constructor(base::String, combine::Symbol; layer, sender_kind=:total,
                                  receiver_kind=:total, kwargs...)
    layer_kw, rest = _split_layer_kwargs(kwargs)
    return DyadDegreeEffect(_resolve_layer(layer; layer_kw...); sender_kind=sender_kind,
                            receiver_kind=receiver_kind, combine=combine, base=base,
                            rest...)
end

"""
    TotaldegreeDyad(; scaling=:none, transform=identity, name=nothing, layer=nothing, …)

The sum of the two actors' total degrees (remstats `totaldegreeDyad()`), the
degree effect available for undirected events.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 3, 3.0)])
compute(TotaldegreeDyad(), h, 1, 2, 4.0)    # 4.0
```
"""
TotaldegreeDyad(; layer=nothing, kwargs...) =
    _dyad_degree_constructor("totaldegreeDyad", :sum; layer=layer, kwargs...)

"""
    DegreeMin(; scaling=:none, transform=identity, name=nothing, layer=nothing, …)

The smaller of the two actors' total degrees (remstats `degreeMin()`, for
undirected events — build it with `symmetric=true`, under which an actor's
degree is the number of events it took part in). `sender_kind=`/`receiver_kind=`
choose other degrees.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 2, 3.0)])
compute(DegreeMin(), h, 1, 3, 4.0)    # 1.0
```
"""
DegreeMin(; layer=nothing, kwargs...) =
    _dyad_degree_constructor("degreeMin", :min; layer=layer, kwargs...)

"""
    DegreeMax(; scaling=:none, transform=identity, name=nothing, layer=nothing, …)

The larger of the two actors' total degrees (remstats `degreeMax()`, for
undirected events).

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 2, 3.0)])
compute(DegreeMax(), h, 1, 3, 4.0)    # 3.0
```
"""
DegreeMax(; layer=nothing, kwargs...) =
    _dyad_degree_constructor("degreeMax", :max; layer=layer, kwargs...)

"""
    DegreeDiff(; transform=identity, name=nothing, layer=nothing, …)

The absolute difference of the two actors' total degrees (remstats
`degreeDiff()`; with a negative sign, the degree assortativity of Lerner,
Hâncean & Perc 2025).

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(1, 2, 3.0)])
compute(DegreeDiff(), h, 1, 3, 4.0)    # 2.0
```
"""
DegreeDiff(; layer=nothing, kwargs...) =
    _dyad_degree_constructor("degreeDiff", :absdiff; layer=layer, kwargs...)

"""
    DegreeAssortativity(; sender_kind=:out, receiver_kind=:in, transform=identity,
                        name=nothing, layer=nothing, …)

The product of the sender's and the receiver's degrees — by default sender
out-degree × receiver in-degree in events, the activity × popularity term of
Lerner & Lomi (2020) (who take `log1p` of each degree first: build it from
`DyadDegreeEffect` on transformed parts, or use an [`Interaction`](@ref) of
two transformed degrees). `measure=:partners` gives the "assortativity by
degree" of Vu, Lomi, Mascia & Pallotti (2017, eq. 9), the product of the
distinct partners. A negative coefficient is read as disassortative
(core–periphery) mixing. It is an endogenous × endogenous interaction, and the
literature's hierarchy principle asks for both main effects alongside it.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 3, 3.0)])
compute(DegreeAssortativity(), h, 1, 3, 4.0)    # 2 × 2 = 4.0
h2 = build_history([Event(1, 2, 1.0), Event(1, 2, 2.0), Event(2, 3, 3.0)])
compute(DegreeAssortativity(measure=:partners), h2, 1, 3, 4.0)   # 1 × 1 = 1.0
```
"""
DegreeAssortativity(; layer=nothing, sender_kind::Symbol=:out,
                    receiver_kind::Symbol=:in, kwargs...) =
    _dyad_degree_constructor("assortativity", :product; layer=layer,
                             sender_kind=sender_kind, receiver_kind=receiver_kind,
                             kwargs...)

# -----------------------------------------------------------------------------
# The two-path
# -----------------------------------------------------------------------------

const _TWOPATH_COMBINE = (:min, :product, :harmonic, :sum, :max, :count)

"""
    TwoPathEffect(leg1, leg2=leg1; dir1=:out, dir2=:in, combine=:min, root=false,
                  order=:none, third=nothing, transform=identity, name=nothing)

The general two-path statistic: a sum over third actors `k ∉ {s, r}` of the two
legs linking the candidate's sender and receiver through `k`.

- `leg1`, `dir1` — the sender's leg: `:out` reads `w₁(s, k)`, `:in` reads
  `w₁(k, s)`, `:sym` their sum.
- `leg2`, `dir2` — the receiver's leg: `:in` reads `w₂(k, r)`, `:out` reads
  `w₂(r, k)`, `:sym` their sum. A *different* layer per leg gives cross-network
  and signed closure (goldfish `mixedTrans`; the balance statistics of Brandes,
  Lerner & Snijders 2009; rem `triadStat(eventtypevalues=…)`).
- `combine` — how the two legs of one path are combined. The packages and
  papers disagree, which is why "the same" triadic effect differs in value
  across them: `:min` (Butts 2008; relevent; remstats; the default of eventnet's
  dyadic triangle statistics, as its documentation states — not verified by
  running it), `:product` (rem; Vu et al. 2011; Perry & Wolfe 2013),
  `:harmonic` — `2ab/(a + b)`, the harmonic mean of Vu, Lomi, Mascia & Pallotti
  (2017, eq. 12) — `:sum`, `:max`, or `:count`, the number of distinct third
  actors (goldfish; remstats `unique=TRUE`).
- `root` — take the square root of the total (rem's `triadStat`; the balance
  statistics).
- `order` — `:leg1_first` or `:leg2_first` keep a third actor only if some event
  on the named leg precedes some event on the other (the dyads' first and last
  events over the whole history, whatever the memory); `:none` ignores timing.
  This is a cheaper device than the time-ordered transitivity of Arena, Mulder &
  Leenders (2024, eq. 12), which counts the time-ordered *pairs* of events on
  the two legs within a lag window, and it agrees with that statistic only when
  each leg holds a single event.
- `third` — a weight `(s, k, r) -> Real` on the third actor: the hook for
  closure among actors sharing an attribute (see [`matching_third`](@ref)) and
  for eventnet's node-attribute-on-the-broker closure.

[`OTP`](@ref), [`ITP`](@ref), [`OSP`](@ref), [`ISP`](@ref),
[`SharedPartners`](@ref) and [`BalanceEffect`](@ref) are its named forms.

# Example
```julia
using Revel
h = build_history([Event(1, 3, 1.0), Event(1, 3, 2.0), Event(3, 2, 3.0)])
L = EventLayer()
compute(TwoPathEffect(L), h, 1, 2, 4.0)                       # min(2, 1) = 1.0
compute(TwoPathEffect(L; combine=:product), h, 1, 2, 4.0)     # 2.0
compute(TwoPathEffect(L; combine=:harmonic), h, 1, 2, 4.0)    # 2·2·1/(2 + 1) ≈ 1.33
compute(TwoPathEffect(L; combine=:count), h, 1, 2, 4.0)       # 1.0
```
"""
struct TwoPathEffect{L1<:EventLayer, L2<:EventLayer, G, F} <: AbstractRevelStatistic
    leg1::L1
    dir1::Symbol
    leg2::L2
    dir2::Symbol
    combine::Symbol
    root::Bool
    order::Symbol
    third::G
    transform::F
    label::String
end

function TwoPathEffect(leg1::EventLayer, leg2::EventLayer=leg1; dir1::Symbol=:out,
                       dir2::Symbol=:in, combine::Symbol=:min, root::Bool=false,
                       order::Symbol=:none, third=nothing, transform=identity,
                       name=nothing, base::Union{Nothing,AbstractString}=nothing)
    dir1 in (:out, :in, :sym) || throw(ArgumentError("dir1 must be :out, :in or :sym, got :$dir1"))
    dir2 in (:out, :in, :sym) || throw(ArgumentError("dir2 must be :out, :in or :sym, got :$dir2"))
    combine in _TWOPATH_COMBINE || throw(ArgumentError(
        "combine must be one of $(_TWOPATH_COMBINE), got :$combine"))
    order in (:none, :leg1_first, :leg2_first) || throw(ArgumentError(
        "order must be :none, :leg1_first or :leg2_first, got :$order"))
    order === :none || (dir1 !== :sym && dir2 !== :sym) || throw(ArgumentError(
        "a time order between the legs is not defined for a :sym leg"))
    f = _transform_fn(transform)
    b = something(base, "twopath($(dir1),$(dir2))") *
        (combine === :min ? "" : ".$(combine)") * (root ? ".root" : "") *
        (order === :none ? "" : ".ordered") * (third === nothing ? "" : ".third")
    suffix = leg1 === leg2 ? _suffix(leg1) :
             "[" * _layer_label(leg1) * "|" * _layer_label(leg2) * "]"
    return TwoPathEffect{typeof(leg1), typeof(leg2), typeof(third), typeof(f)}(
        leg1, dir1, leg2, dir2, combine, root, order, third, f,
        _label(name, _auto_name(b, suffix, f)))
end

@inline _leg1(L, st, dir::Symbol, s::Int, k::Int) =
    dir === :out ? _w(L, st, s, k) : dir === :in ? _w(L, st, k, s) :
                   _w(L, st, s, k) + _w(L, st, k, s)
@inline _leg2(L, st, dir::Symbol, k::Int, r::Int) =
    dir === :in ? _w(L, st, k, r) : dir === :out ? _w(L, st, r, k) :
                  _w(L, st, k, r) + _w(L, st, r, k)

@inline function _twopath_term(stat::TwoPathEffect, st1, st2, s::Int, k::Int, r::Int)
    (k == s || k == r) && return 0.0
    a = _leg1(stat.leg1, st1, stat.dir1, s, k)
    a > 0 || return 0.0
    b = _leg2(stat.leg2, st2, stat.dir2, k, r)
    b > 0 || return 0.0
    if stat.order !== :none
        f1 = stat.dir1 === :out ? _first_t(st1, s, k) : _first_t(st1, k, s)
        l1 = stat.dir1 === :out ? _last_t(st1, s, k) : _last_t(st1, k, s)
        f2 = stat.dir2 === :in ? _first_t(st2, k, r) : _first_t(st2, r, k)
        l2 = stat.dir2 === :in ? _last_t(st2, k, r) : _last_t(st2, r, k)
        ok = stat.order === :leg1_first ? f1 < l2 : f2 < l1
        ok || return 0.0
    end
    c = stat.combine
    v = c === :min ? min(a, b) : c === :product ? a * b :
        c === :harmonic ? 2a * b / (a + b) : c === :sum ? a + b :
        c === :max ? max(a, b) : 1.0
    stat.third === nothing || (v *= Float64(stat.third(s, k, r)))
    return v
end

function _value(stat::TwoPathEffect, events, s::Int, r::Int, t::Float64)
    st1 = _sync!(stat.leg1, events, t)
    st2 = stat.leg2 === stat.leg1 ? st1 : _sync!(stat.leg2, events, t)
    total = 0.0
    if stat.dir1 === :sym
        for k in 1:st1.n
            total += _twopath_term(stat, st1, st2, s, k, r)
        end
    else
        # A third actor with no weight on the sender's leg contributes nothing
        # under any `combine`, so the sender's neighbours are all that is needed
        for k in (stat.dir1 === :out ? _out_nb(st1, s) : _in_nb(st1, s))
            total += _twopath_term(stat, st1, st2, s, k, r)
        end
    end
    stat.root && (total = sqrt(total))
    return Float64(stat.transform(total))
end

_interval_constant(stat::TwoPathEffect) =
    _layer_interval_constant(stat.leg1) &&
    _layer_interval_constant(stat.leg2)

function _twopath_constructor(base::String, dir1::Symbol, dir2::Symbol, order_when_true::Symbol;
                              layer, ordered::Bool=false, order::Symbol=:none, kwargs...)
    layer_kw, rest = _split_layer_kwargs(kwargs)
    L = _resolve_layer(layer; layer_kw...)
    ordered && order !== :none && throw(ArgumentError(
        "pass either ordered=true or order=…, not both"))
    ordered && order_when_true === :none && throw(ArgumentError(
        "$base has no natural leg order; pass order=:leg1_first or :leg2_first"))
    return TwoPathEffect(L, L; dir1=dir1, dir2=dir2, base=base,
                         order=ordered ? order_when_true : order, rest...)
end

"""
    OTP(; combine=:min, root=false, ordered=false, third=nothing,
        transform=identity, name=nothing, layer=nothing, memory=…, …)

Outgoing two-paths `s → k → r`, the transitive-closure statistic: for each third
actor the combined weight of `s → k` and `k → r` (Butts 2008; relevent
`OTPSnd`; remstats `otp()`; goldfish `trans`; eventnet "transitive_tie"; Perry &
Wolfe's "2-send"). `combine=:count` gives goldfish's count of distinct
intermediaries and remstats' `unique=TRUE`; `combine=:product, root=true` gives
rem's `triadStat`. `ordered=true` requires `s → k` to have happened before
`k → r`. See [`TwoPathEffect`](@ref).

# Example
```julia
using Revel
h = build_history([Event(1, 3, 1.0), Event(3, 2, 2.0), Event(1, 4, 3.0), Event(4, 2, 4.0)])
compute(OTP(), h, 1, 2, 5.0)                  # 2.0 — through 3 and through 4
compute(OTP(), h, 2, 1, 5.0)                  # 0.0
compute(OTP(ordered=true), h, 1, 2, 5.0)      # 2.0 — both first legs came first
```
"""
OTP(; layer=nothing, kwargs...) =
    _twopath_constructor("otp", :out, :in, :leg1_first; layer=layer, kwargs...)

"""
    ITP(; combine=:min, root=false, ordered=false, third=nothing,
        transform=identity, name=nothing, layer=nothing, memory=…, …)

Incoming two-paths `r → k → s`, the cyclic-closure statistic (Butts 2008;
relevent `ITPSnd`; remstats `itp()`; goldfish `cycle`; Perry & Wolfe's
"2-receive"). `ordered=true` requires `r → k` to have happened before `k → s`.
See [`TwoPathEffect`](@ref).

# Example
```julia
using Revel
h = build_history([Event(2, 3, 1.0), Event(3, 1, 2.0)])
compute(ITP(), h, 1, 2, 3.0)    # 1.0 — closing the cycle 2 → 3 → 1 → 2
compute(ITP(), h, 2, 1, 3.0)    # 0.0
```
"""
ITP(; layer=nothing, kwargs...) =
    _twopath_constructor("itp", :in, :out, :leg2_first; layer=layer, kwargs...)

"""
    OSP(; combine=:min, root=false, order=:none, third=nothing,
        transform=identity, name=nothing, layer=nothing, memory=…, …)

Outbound shared partners `s → k ← r`: third actors both have sent to (Butts
2008; relevent `OSPSnd`, whose output in relevent 1.2.1 differs from this
definition — see the concordance; remstats `osp()`; goldfish `commonReceiver`;
Perry & Wolfe's "cosibling"). amorem's "sending balance" is the same
configuration, but whether it combines the legs by the minimum has not been
checked. Mind the vocabulary: the partner is *shared as a receiver* — "common
receiver" — while the sender and the receiver of the candidate both *send*; a
package that names the effect after the candidate's side calls it the opposite
of one that names it after the partner. See [`TwoPathEffect`](@ref).

# Example
```julia
using Revel
h = build_history([Event(1, 3, 1.0), Event(2, 3, 2.0)])
compute(OSP(), h, 1, 2, 3.0)    # 1.0 — both sent to 3
compute(ISP(), h, 1, 2, 3.0)    # 0.0
```
"""
OSP(; layer=nothing, kwargs...) =
    _twopath_constructor("osp", :out, :out, :none; layer=layer, kwargs...)

"""
    ISP(; combine=:min, root=false, order=:none, third=nothing,
        transform=identity, name=nothing, layer=nothing, memory=…, …)

Inbound shared partners `s ← k → r`: third actors who have sent to both (Butts
2008; relevent `ISPSnd`; remstats `isp()`; goldfish `commonSender`; amorem
"receiving balance"; Perry & Wolfe's "sibling"). See [`TwoPathEffect`](@ref).

# Example
```julia
using Revel
h = build_history([Event(3, 1, 1.0), Event(3, 2, 2.0)])
compute(ISP(), h, 1, 2, 3.0)    # 1.0 — 3 sent to both
compute(OSP(), h, 1, 2, 3.0)    # 0.0
```
"""
ISP(; layer=nothing, kwargs...) =
    _twopath_constructor("isp", :in, :in, :none; layer=layer, kwargs...)

"""
    SharedPartners(; combine=:min, transform=identity, name=nothing,
                   layer=nothing, memory=…, symmetric=true, …)

Shared partners for **undirected** events: third actors that have interacted
with both (remstats `sp()`; goldfish `trans` in the coordination model; eventnet
closure with direction `SYM`). The layer is symmetric by default, so each event
counts for both of its participants.

# Example
```julia
using Revel
h = build_history([Event(1, 3, 1.0), Event(3, 2, 2.0)])
compute(SharedPartners(), h, 1, 2, 3.0)    # 1.0 — 3 has interacted with both
compute(SharedPartners(), h, 2, 1, 3.0)    # 1.0 — and the statistic is symmetric
```
"""
function SharedPartners(; layer=nothing, kwargs...)
    layer_kw, rest = _split_layer_kwargs(kwargs)
    if layer === nothing && !any(p -> first(p) === :symmetric, layer_kw)
        push!(layer_kw, :symmetric => true)
    end
    L = _resolve_layer(layer; layer_kw...)
    return TwoPathEffect(L, L; dir1=:out, dir2=:out, base="sp", rest...)
end

const _BALANCE_KINDS = (:friend_of_friend, :friend_of_enemy, :enemy_of_friend,
                        :enemy_of_enemy)

"""
    BalanceEffect(kind; positive=:positive, negative=:negative, root=true,
                  transform=identity, name=nothing, memory=…, weighted=false,
                  clock=:time)

The structural-balance statistics of Brandes, Lerner & Snijders (2009) for
signed events: the two-path through a third actor `k` whose first leg (`s`–`k`)
and second leg (`k`–`r`) are read off the **undirected** positive or negative
event weights, the legs multiplied, the paths summed and the square root taken.

`kind` names what the receiver `r` is to the sender `s`, through `k`, with the
signs of the (`s`–`k`, `k`–`r`) legs as in Brandes et al.'s definitions:
`:friend_of_friend` (+,+), `:friend_of_enemy` (−,+: `k` is `s`'s enemy and
`r`'s friend — friendOfEnemy(a, b) = √Σᵢ ω⁻(a,i)·ω⁺(i,b)), `:enemy_of_friend`
(+,−: `k` is `s`'s friend and `r`'s enemy) or `:enemy_of_enemy` (−,−).
`positive`/`negative` name the event types carrying each sign. Balance theory
predicts positive events towards friends of friends and enemies of enemies, and
negative events towards the other two. The same construction is rem's
`triadStat(eventtypevalues=…)` and eventnet's "enemy of friend" closure.

# Example
```julia
using Revel
h = build_history([Event(1, 3, 1.0; eventtype=:positive),
                   Event(3, 2, 2.0; eventtype=:negative)])
# 3 is 1's friend and 2's enemy: 2 is an enemy of 1's friend
compute(BalanceEffect(:enemy_of_friend), h, 1, 2, 3.0)     # 1.0
compute(BalanceEffect(:friend_of_enemy), h, 1, 2, 3.0)     # 0.0
compute(BalanceEffect(:friend_of_enemy), h, 2, 1, 3.0)     # 1.0 — 1 is a friend of 2's enemy
```
"""
function BalanceEffect(kind::Symbol; positive=:positive, negative=:negative,
                       root::Bool=true, transform=identity, name=nothing,
                       memory::AbstractMemory=FullMemory(), weighted::Bool=false,
                       clock::Symbol=:time)
    kind in _BALANCE_KINDS || throw(ArgumentError(
        "kind must be one of $(_BALANCE_KINDS), got :$kind"))
    pos = EventLayer(memory=memory, types=positive, weighted=weighted, clock=clock,
                     symmetric=true)
    neg = EventLayer(memory=memory, types=negative, weighted=weighted, clock=clock,
                     symmetric=true)
    # The sender's leg is positive when `k` is the sender's friend, the
    # receiver's leg when `k` is the receiver's friend
    leg1 = kind in (:friend_of_friend, :enemy_of_friend) ? pos : neg
    leg2 = kind in (:friend_of_friend, :friend_of_enemy) ? pos : neg
    auto = string(kind) * _suffix(EventLayer(memory=memory, weighted=weighted, clock=clock))
    # Both layers are symmetric, so `:out` on each leg reads the undirected weight
    return TwoPathEffect(leg1, leg2; dir1=:out, dir2=:out, combine=:product, root=root,
                         transform=transform, name=something(name, auto))
end

"""
    matching_third(x) -> Function

A `third=` weight for [`TwoPathEffect`](@ref) (and [`OTP`](@ref), [`ITP`](@ref),
[`OSP`](@ref), [`ISP`](@ref)) that keeps only third actors whose covariate value
equals the sender's: closure among same-attribute actors. The review found this
interaction only as a filtered statistic in the literature, never as a product
term, and the two are different models — this is the filter.

`x` is a vector indexed by actor ID.

# Example
```julia
using Revel
team = [1, 1, 2, 1]
h = build_history([Event(1, 2, 1.0), Event(2, 4, 2.0), Event(1, 3, 3.0), Event(3, 4, 4.0)])
compute(OTP(), h, 1, 4, 5.0)                               # 2.0 — via 2 and via 3
compute(OTP(third=matching_third(team)), h, 1, 4, 5.0)     # 1.0 — only 2 is on 1's team
```
"""
function matching_third(x::AbstractVector)
    values = collect(x)
    return (s, k, r) -> (s <= length(values) && k <= length(values) &&
                         values[s] == values[k]) ? 1.0 : 0.0
end

# -----------------------------------------------------------------------------
# The three-path (four-cycle)
# -----------------------------------------------------------------------------

"""
    FourCycleEffect(layer; combine=:min, root=false, transform=identity, name=nothing)
    FourCycleEffect(; combine=:min, root=false, …, layer=nothing, memory=…, …)

Four-cycle closure: for a candidate `s → r`, the three-paths
`s → a ← b → r` — another sender `b` shares a target `a` with `s` and also
targets `r`. It is the closure statistic of two-mode event networks, where
triads do not exist (Lerner & Lomi 2020; goldfish `four`; rem `fourCycleStat`;
eventnet `FOUR_CYCLE_STATISTIC`; "shared support" in Haunss & Hollway 2023).

At least three non-equivalent versions are in use, selected by `combine` and
`root`: `:min` — the sum of the minima of the three weights (Lerner & Lomi
2020); `combine=:product, root=true` — the cube root of the summed products
(rem); `:count` — the number of three-paths on the binary network (goldfish).

# Example
```julia
using Revel
# actors 1–2 are senders, 11–12 targets: 1 and 2 both used 11, and 2 used 12
h = build_history([Event(1, 11, 1.0), Event(2, 11, 2.0), Event(2, 12, 3.0)])
compute(FourCycleEffect(), h, 1, 12, 4.0)    # 1.0 — closing 1 → 11 ← 2 → 12
compute(FourCycleEffect(), h, 1, 11, 4.0)    # 0.0
```
"""
struct FourCycleEffect{L<:EventLayer, F} <: AbstractRevelStatistic
    layer::L
    combine::Symbol
    root::Bool
    transform::F
    label::String
end

function FourCycleEffect(layer::EventLayer; combine::Symbol=:min, root::Bool=false,
                         transform=identity, name=nothing)
    combine in (:min, :product, :count) || throw(ArgumentError(
        "combine must be :min, :product or :count, got :$combine"))
    f = _transform_fn(transform)
    b = "fourcycle" * (combine === :min ? "" : ".$(combine)") * (root ? ".root" : "")
    return FourCycleEffect{typeof(layer), typeof(f)}(
        layer, combine, root, f, _label(name, _auto_name(b, _suffix(layer), f)))
end

function FourCycleEffect(; layer=nothing, kwargs...)
    layer_kw, rest = _split_layer_kwargs(kwargs)
    return FourCycleEffect(_resolve_layer(layer; layer_kw...); rest...)
end

function _value(stat::FourCycleEffect, events, s::Int, r::Int, t::Float64)
    L = stat.layer
    st = _sync!(L, events, t)
    total = 0.0
    for a in _out_nb(st, s)
        (a == r || a == s) && continue
        w1 = _w(L, st, s, a)
        w1 > 0 || continue
        for b in _in_nb(st, a)
            (b == s || b == r) && continue
            w2 = _w(L, st, b, a)
            w2 > 0 || continue
            w3 = _w(L, st, b, r)
            w3 > 0 || continue
            total += stat.combine === :min ? min(w1, w2, w3) :
                     stat.combine === :product ? w1 * w2 * w3 : 1.0
        end
    end
    stat.root && (total = cbrt(total))
    return Float64(stat.transform(total))
end

_interval_constant(stat::FourCycleEffect) = _layer_interval_constant(stat.layer)

# -----------------------------------------------------------------------------
# Order-based devices: recency ranks, time since the last event
# -----------------------------------------------------------------------------

"""
    RecencyRank(mode=:receive; name=nothing, layer=nothing, types=…, keep=…)

Butts's (2008) rank-based recency. `mode=:receive` is the inverse of the rank of
the candidate's receiver among the actors who most recently **sent to** the
sender (`1` for the latest, `1/2` for the one before, `0` if never) — relevent
`RRecSnd`, remstats `rrankReceive()`. `mode=:send` ranks the receiver among the
actors the sender most recently **sent to** — relevent `RSndSnd`, remstats
`rrankSend()`, first printed by DuBois, Butts, McFarland & Smyth (2013).

Ranks depend on event order only, so the statistic takes no memory kernel (a
`memory=` keyword, or a shared `layer=` with one, is refused); `types`/`keep`
select which events are ranked.

# Example
```julia
using Revel
h = build_history([Event(2, 1, 1.0), Event(3, 1, 2.0), Event(1, 4, 3.0)])
compute(RecencyRank(:receive), h, 1, 3, 4.0)    # 1.0 — 3 wrote to 1 most recently
compute(RecencyRank(:receive), h, 1, 2, 4.0)    # 0.5
compute(RecencyRank(:send), h, 1, 4, 4.0)       # 1.0
```
"""
struct RecencyRank{L<:EventLayer} <: AbstractRevelStatistic
    layer::L
    mode::Symbol
    label::String
end

function RecencyRank(mode::Symbol=:receive; layer=nothing, name=nothing, kwargs...)
    mode in (:send, :receive) || throw(ArgumentError(
        "mode must be :send or :receive, got :$mode"))
    layer_kw, rest = _split_layer_kwargs(kwargs)
    _no_extra_kwargs(rest, "RecencyRank")
    L = _resolve_layer(layer; layer_kw...)
    _no_memory(L, "RecencyRank", "a rank depends on the order of events only")
    auto = (mode === :send ? "rrankSend" : "rrankReceive") * _suffix(L)
    return RecencyRank{typeof(L)}(L, mode, _label(name, auto))
end

function _value(stat::RecencyRank, events, s::Int, r::Int, t::Float64)
    st = _sync!(stat.layer, events, t)
    if stat.mode === :send
        idx = _last_i(st, s, r)
        idx == 0 && return 0.0
        rank = 1
        for k in _out_nb(st, s)
            _last_i(st, s, k) > idx && (rank += 1)
        end
        return 1.0 / rank
    end
    idx = _last_i(st, r, s)
    idx == 0 && return 0.0
    rank = 1
    for k in _in_nb(st, s)
        _last_i(st, k, s) > idx && (rank += 1)
    end
    return 1.0 / rank
end

_interval_constant(::RecencyRank) = true

# Statistics that read the order or the timing of the last event, not weights
function _no_memory(L::EventLayer, what::AbstractString, why::AbstractString)
    L.memory isa FullMemory || throw(ArgumentError(
        "$what takes no memory kernel: $why, so a $(_memory_label(L.memory)) " *
        "memory would be ignored. Leave `memory` out (and pass a `layer=` without one)."))
    return nothing
end

"""
    inverse_gap(Δ) -> Float64

`1 / (Δ + 1)`: remstats' transform of the time `Δ` since an event, the default
of [`TimeSince`](@ref).

# Example
```julia
using Revel
inverse_gap(0.0), inverse_gap(3.0)    # (1.0, 0.25)
```
"""
inverse_gap(Δ) = 1.0 / (Δ + 1.0)

const _TIMESINCE_TARGETS = (:dyad, :reverse, :pair, :send_sender, :send_receiver,
                            :receive_sender, :receive_receiver)

"""
    TimeSince(target=:dyad; transform=inverse_gap, empty=0.0, name=nothing,
              layer=nothing, types=…, keep=…, clock=:time)

A function of the time elapsed since the most recent event of a given kind:

| `target` | the clock started at | remstats |
|---|---|---|
| `:dyad` | the last `s → r` event | `recencyContinue()` |
| `:reverse` | the last `r → s` event | — |
| `:pair` | the last event between the two, either way | `recencyContinue()` (undirected) |
| `:send_sender` | the sender last sent | `recencySendSender()` |
| `:send_receiver` | the receiver last sent | `recencySendReceiver()` |
| `:receive_sender` | the sender last received | `recencyReceiveSender()` |
| `:receive_receiver` | the receiver last received | `recencyReceiveReceiver()` |

"Recency" names at least four different statistics in the literature, and
`transform` selects among them: [`inverse_gap`](@ref) `1/(Δ+1)` (remstats, the
default), `Δ -> exp(-b*Δ)` (an exponentially fading recency), `Δ -> 1 - exp(-Δ)` (Boschi
& Wit 2026, who code "never happened" as 1 — pass `empty=1.0` with it, since
the default `empty=0.0` would equal "just happened") or `identity` for the raw
gap time (Zappa & Vu 2021). `empty` is returned when no such event has
happened. With `clock=:order` the gap is measured in events. The statistic
reads the time of the last event, not a weight, so it takes no memory kernel.

On the time clock the statistic changes continuously between events, so it is
not admissible in the exact-time likelihood; with `clock=:order` it is.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(3, 1, 4.0)])
compute(TimeSince(:dyad), h, 1, 2, 5.0)                       # 1/(4 + 1) = 0.2
compute(TimeSince(:receive_sender), h, 1, 2, 5.0)             # 1/(1 + 1) = 0.5
compute(TimeSince(:dyad; transform=identity, empty=-1.0), h, 2, 3, 5.0)   # -1.0
```
"""
struct TimeSince{L<:EventLayer, F} <: AbstractRevelStatistic
    layer::L
    target::Symbol
    empty::Float64
    transform::F
    label::String
end

const _TIMESINCE_NAMES = Dict(
    :dyad => "recencyContinue", :reverse => "recencyReverse", :pair => "recencyPair",
    :send_sender => "recencySendSender", :send_receiver => "recencySendReceiver",
    :receive_sender => "recencyReceiveSender",
    :receive_receiver => "recencyReceiveReceiver")

function TimeSince(target::Symbol=:dyad; layer=nothing, transform=inverse_gap,
                   empty::Real=0.0, name=nothing, kwargs...)
    target in _TIMESINCE_TARGETS || throw(ArgumentError(
        "target must be one of $(_TIMESINCE_TARGETS), got :$target"))
    layer_kw, rest = _split_layer_kwargs(kwargs)
    _no_extra_kwargs(rest, "TimeSince")
    L = _resolve_layer(layer; layer_kw...)
    _no_memory(L, "TimeSince", "it reads the time since the last event")
    f = transform isa Symbol ? _transform_fn(transform) : transform
    tl = f === inverse_gap ? "" : f === identity ? ".gap" : "." * string(nameof(f))
    auto = _TIMESINCE_NAMES[target] * tl * _suffix(L)
    return TimeSince{typeof(L), typeof(f)}(L, target, Float64(empty), f, _label(name, auto))
end

@inline _latest(a::Float64, b::Float64) = isnan(a) ? b : isnan(b) ? a : max(a, b)

function _value(stat::TimeSince, events, s::Int, r::Int, t::Float64)
    st = _sync!(stat.layer, events, t)
    tg = stat.target
    last = tg === :dyad ? _last_t(st, s, r) :
           tg === :reverse ? _last_t(st, r, s) :
           tg === :pair ? _latest(_last_t(st, s, r), _last_t(st, r, s)) :
           tg === :send_sender ? (1 <= s <= st.n ? st.last_out_t[s] : NaN) :
           tg === :send_receiver ? (1 <= r <= st.n ? st.last_out_t[r] : NaN) :
           tg === :receive_sender ? (1 <= s <= st.n ? st.last_in_t[s] : NaN) :
                                    (1 <= r <= st.n ? st.last_in_t[r] : NaN)
    isnan(last) && return stat.empty
    return Float64(stat.transform(st.now - last))
end

# On the event clock the gap changes only when an event happens
_interval_constant(stat::TimeSince) = stat.layer.clock === :order

# -----------------------------------------------------------------------------
# Participation shifts beyond Gibson's thirteen
# -----------------------------------------------------------------------------

"""
    PShiftABAB(; name="PSAB-AB") <: AbstractRevelStatistic

The participation shift in which the same dyad repeats immediately: previous
event `A → B`, candidate `A → B` (remstats `psABAB()`). It has no counterpart
among Gibson's thirteen shifts, which [`PShift`](@ref) implements.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0)])
compute(PShiftABAB(), h, 1, 2, 2.0)       # 1.0
compute(PShiftABAB(), h, 2, 1, 2.0)       # 0.0 — that is PShift(:AB_BA)
compute(PShift(:AB_BA), h, 2, 1, 2.0)     # 1.0
```
"""
struct PShiftABAB <: AbstractRevelStatistic
    label::String
end
PShiftABAB(; name::AbstractString="PSAB-AB") = PShiftABAB(String(name))

function _value(::PShiftABAB, events, s::Int, r::Int, t::Float64)
    k = _n_before(events, t)
    k == 0 && return 0.0
    a, b, _ = _sig(events[k])
    return (s == a && r == b) ? 1.0 : 0.0
end

_interval_constant(::PShiftABAB) = true

"""
    UndirectedPShift(kind; name=nothing) <: AbstractRevelStatistic

Participation shifts for **undirected** events, where a pair has no sender and
no receiver (remstats' two shifts for undirected data):

- `:AB_AB` — the previous event's pair interacts again (`psABAB()`);
- `:AB_AY` — exactly one actor of the previous pair takes part, with someone new
  (`psABAY()`).

The orientation in which a pair happens to be stored does not matter. For
directed events use [`PShift`](@ref) and [`PShiftABAB`](@ref).

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0)])
compute(UndirectedPShift(:AB_AB), h, 2, 1, 2.0)    # 1.0 — the same pair
compute(UndirectedPShift(:AB_AY), h, 2, 3, 2.0)    # 1.0 — 2 carries on with 3
compute(UndirectedPShift(:AB_AY), h, 3, 4, 2.0)    # 0.0 — nobody from the last event
```
"""
struct UndirectedPShift <: AbstractRevelStatistic
    kind::Symbol
    label::String
end

function UndirectedPShift(kind::Symbol; name=nothing)
    kind in (:AB_AB, :AB_AY) || throw(ArgumentError(
        "kind must be :AB_AB or :AB_AY, got :$kind (the directed shifts are " *
        "PShift's)"))
    return UndirectedPShift(kind, _label(name, "PS" * replace(String(kind), "_" => "-") *
                                                ".undirected"))
end

function _value(stat::UndirectedPShift, events, s::Int, r::Int, t::Float64)
    k = _n_before(events, t)
    k == 0 && return 0.0
    a, b, _ = _sig(events[k])
    shared = (s == a || s == b) + (r == a || r == b)
    return stat.kind === :AB_AB ? Float64(shared == 2) : Float64(shared == 1)
end

_interval_constant(::UndirectedPShift) = true

# -----------------------------------------------------------------------------
# Neighbourhood statistics: embeddedness and structural similarity
# -----------------------------------------------------------------------------

"""
    NodeTransitivity(; role=:receiver, transform=identity, name=nothing,
                     layer=nothing, memory=…, …)

Node-level embeddedness: the number of transitive structures `v → a`, `v → b`,
`a → b` in which actor `v` — the candidate's receiver (`role=:receiver`) or
sender (`:sender`) — is the source (goldfish `nodeTrans`, with `type="alter"` or
`"ego"`). Computed on the binary network of the layer.

# Example
```julia
using Revel
h = build_history([Event(1, 2, 1.0), Event(1, 3, 2.0), Event(2, 3, 3.0)])
compute(NodeTransitivity(role=:sender), h, 1, 4, 4.0)    # 1.0 — 1 → 2, 1 → 3, 2 → 3
compute(NodeTransitivity(), h, 1, 4, 4.0)                # 0.0 — about actor 4
```
"""
struct NodeTransitivity{L<:EventLayer, F} <: AbstractRevelStatistic
    layer::L
    role::Symbol
    transform::F
    label::String
end

function NodeTransitivity(; layer=nothing, role::Symbol=:receiver, transform=identity,
                          name=nothing, kwargs...)
    role in (:sender, :receiver) || throw(ArgumentError(
        "role must be :sender or :receiver, got :$role"))
    layer_kw, rest = _split_layer_kwargs(kwargs)
    _no_extra_kwargs(rest, "NodeTransitivity")
    L = _resolve_layer(layer; layer_kw...)
    f = _transform_fn(transform)
    auto = _auto_name("nodeTrans" * (role === :sender ? "Sender" : "Receiver"),
                      _suffix(L), f)
    return NodeTransitivity{typeof(L), typeof(f)}(L, role, f, _label(name, auto))
end

function _value(stat::NodeTransitivity, events, s::Int, r::Int, t::Float64)
    L = stat.layer
    st = _sync!(L, events, t)
    v = stat.role === :sender ? s : r
    total = 0.0
    nb = _out_nb(st, v)
    for a in nb
        _w(L, st, v, a) > 0 || continue
        for b in nb
            (b == a) && continue
            _w(L, st, v, b) > 0 || continue
            _w(L, st, a, b) > 0 && (total += 1.0)
        end
    end
    return Float64(stat.transform(total))
end

_interval_constant(stat::NodeTransitivity) = _layer_interval_constant(stat.layer)

"""
    StructuralSimilarity(; measure=:jaccard, direction=:out, transform=identity,
                         name=nothing, layer=nothing, memory=…, …)

How alike the candidate's sender and receiver are in whom they have sent to
(`direction=:out`) or received from (`:in`): the Jaccard index of their partner
sets (`measure=:jaccard`) or the cosine of their weight profiles (`:cosine`).
These are eventnet's `JACCARD_SIM_STATISTIC` and `COSINE_SIM_STATISTIC`, a
structural-equivalence effect. The two actors themselves are left out of the
profiles.

This is not the `similarityStat` of the `rem` package, which compares a sender
with the *other senders* of the current target; for that construction use
[`FourCycleEffect`](@ref), which counts the same co-usage paths.

# Example
```julia
using Revel
h = build_history([Event(1, 3, 1.0), Event(1, 4, 2.0), Event(2, 3, 3.0)])
compute(StructuralSimilarity(), h, 1, 2, 4.0)                   # 0.5 — {3,4} vs {3}
compute(StructuralSimilarity(measure=:cosine), h, 1, 2, 4.0)    # 1/√2
```
"""
struct StructuralSimilarity{L<:EventLayer, F} <: AbstractRevelStatistic
    layer::L
    measure::Symbol
    direction::Symbol
    transform::F
    label::String
end

function StructuralSimilarity(; layer=nothing, measure::Symbol=:jaccard,
                              direction::Symbol=:out, transform=identity,
                              name=nothing, kwargs...)
    measure in (:jaccard, :cosine) || throw(ArgumentError(
        "measure must be :jaccard or :cosine, got :$measure"))
    direction in (:out, :in) || throw(ArgumentError(
        "direction must be :out or :in, got :$direction"))
    layer_kw, rest = _split_layer_kwargs(kwargs)
    _no_extra_kwargs(rest, "StructuralSimilarity")
    L = _resolve_layer(layer; layer_kw...)
    f = _transform_fn(transform)
    auto = _auto_name("similarity.$(measure).$(direction)", _suffix(L), f)
    return StructuralSimilarity{typeof(L), typeof(f)}(L, measure, direction, f,
                                                      _label(name, auto))
end

function _value(stat::StructuralSimilarity, events, s::Int, r::Int, t::Float64)
    L = stat.layer
    st = _sync!(L, events, t)
    out = stat.direction === :out
    wt(a, k) = out ? _w(L, st, a, k) : _w(L, st, k, a)
    ns = out ? _out_nb(st, s) : _in_nb(st, s)
    nr = out ? _out_nb(st, r) : _in_nb(st, r)
    dot = 0.0; ss = 0.0; rr = 0.0; both = 0.0; cs = 0.0; cr = 0.0
    for k in ns
        (k == s || k == r) && continue
        a = wt(s, k)
        a > 0 || continue
        cs += 1.0; ss += a * a
        b = wt(r, k)
        b > 0 && (both += 1.0; dot += a * b)
    end
    for k in nr
        (k == s || k == r) && continue
        b = wt(r, k)
        b > 0 || continue
        cr += 1.0; rr += b * b
    end
    v = if stat.measure === :jaccard
        union = cs + cr - both
        union > 0 ? both / union : 0.0
    else
        (ss > 0 && rr > 0) ? dot / sqrt(ss * rr) : 0.0
    end
    return Float64(stat.transform(v))
end

_interval_constant(stat::StructuralSimilarity) =
    _layer_interval_constant(stat.layer)
