# =============================================================================
# Memory kernels
# =============================================================================
#
# How much a past event still counts is a modelling dimension of its own in the
# REM literature: full accumulated history (Butts 2008), exponential half-life decay (Brandes,
# Lerner & Snijders 2009), sliding windows (de Nooy 2011; Quintane et al. 2013),
# an interval partition of the past (Perry & Wolfe 2013), power-law decay (Vu et
# al. 2017) and linear decay (Arena, Mulder & Leenders 2023). Every endogenous
# effect in this package takes any of them through its `memory=` keyword, so the
# memory model is orthogonal to the structural configuration.

"""
    AbstractMemory

Supertype of the memory kernels: a rule giving the weight `kernel_weight(m, age)`
a past event of age `age ≥ 0` carries in an endogenous statistic. The concrete
kernels are [`FullMemory`](@ref), [`HalfLife`](@ref), [`Window`](@ref),
[`IntervalMemory`](@ref), [`PowerLaw`](@ref), [`LinearDecay`](@ref) and
[`KernelMemory`](@ref).

# Example
```julia
using Revel
kernel_weight(FullMemory(), 10.0)     # 1.0
kernel_weight(HalfLife(5.0), 5.0)     # 0.5
kernel_weight(Window(3.0), 4.0)       # 0.0
```
"""
abstract type AbstractMemory end

"""
    FullMemory() <: AbstractMemory

Every past event counts with weight 1, however old (Butts 2008; remstats
`memory = "full"`). Its statistics are constant between events, which is what
the interval-timing likelihood requires; the only other kernel with that
property is `HalfLife(Inf)`, which is the same thing.

# Example
```julia
using Revel
kernel_weight(FullMemory(), 1e6)   # 1.0
```
"""
struct FullMemory <: AbstractMemory end

"""
    HalfLife(halflife; normalized=false) <: AbstractMemory

Exponential decay: an event of age `a` weighs `exp(-a·ln2/halflife)`, so its
weight halves every `halflife` time units (Brandes, Lerner & Snijders 2009;
eventnet; remstats `memory = "decay"`).

Two normalisations coexist in the literature and they are *not* interchangeable
in a fixture: `normalized=false` (Lerner & Lomi 2020; Arena et al. 2023) is the
plain weight above; `normalized=true` multiplies it by `ln2/halflife` (Brandes et
al. 2009; the `rem` package), which rescales the coefficient by a constant.
`halflife=Inf` is no decay.

# Example
```julia
using Revel
kernel_weight(HalfLife(10.0), 10.0)                    # 0.5
kernel_weight(HalfLife(10.0; normalized=true), 0.0)    # log(2)/10
```
"""
struct HalfLife <: AbstractMemory
    halflife::Float64
    normalized::Bool
    function HalfLife(halflife::Real; normalized::Bool=false)
        halflife > 0 || throw(ArgumentError("halflife must be positive, got $halflife"))
        normalized && !isfinite(halflife) && throw(ArgumentError(
            "normalized=true multiplies the weight by ln2/halflife, which is zero " *
            "for halflife=Inf; use normalized=false with an infinite half-life"))
        new(Float64(halflife), normalized)
    end
end

"""
    Window(width) <: AbstractMemory

Sliding window: an event counts with weight 1 while its age is at most `width`
and not at all afterwards (de Nooy 2011; Quintane et al. 2013; goldfish
`window`; remstats `memory = "window"`). An event exactly `width` old still
counts, as in REM.jl.

# Example
```julia
using Revel
kernel_weight(Window(24.0), 24.0)   # 1.0
kernel_weight(Window(24.0), 24.5)   # 0.0
```
"""
struct Window <: AbstractMemory
    width::Float64
    function Window(width::Real)
        width > 0 || throw(ArgumentError("window width must be positive, got $width"))
        new(Float64(width))
    end
end

"""
    IntervalMemory(lo, hi) <: AbstractMemory

Events whose age lies in `(lo, hi]` count with weight 1 (Perry & Wolfe 2013;
remstats `memory = "interval"`). A set of adjacent intervals partitions the past,
and giving each its own coefficient estimates the decay *shape* rather than
assuming it — see [`interval_partition`](@ref). (It is not called `Interval`
so that it does not clash with `IntervalSets.Interval`.)

# Example
```julia
using Revel
kernel_weight(IntervalMemory(1.0, 7.0), 1.0)   # 0.0 — the lower bound is open
kernel_weight(IntervalMemory(1.0, 7.0), 7.0)   # 1.0
```
"""
struct IntervalMemory <: AbstractMemory
    lo::Float64
    hi::Float64
    function IntervalMemory(lo::Real, hi::Real)
        0 <= lo < hi || throw(ArgumentError(
            "an IntervalMemory needs 0 <= lo < hi, got ($lo, $hi)"))
        new(Float64(lo), Float64(hi))
    end
end

"""
    PowerLaw(exponent; offset=0.0, support=Inf) <: AbstractMemory

Power-law decay: an event of age `a` weighs `(a + offset)^(-exponent)` (the
Lomi/Vu line: Vu, Lomi, Mascia & Pallotti 2017; Bianchi & Lomi 2023, with the
exponent chosen by grid search — see [`profile_memory`](@ref)). With
`offset = 0` the weight is undefined at age zero, which only arises for events
tied with the one being explained under `ties=:ordered`; pass a positive
`offset` for such data.

A power law never reaches zero, so every new clock re-reads the whole history:
a sequence of `E` events costs `O(E²)`. `support` truncates the kernel — an
event older than `support` weighs 0 — which bounds that cost by the events
inside it; choose it where the weights have become negligible.

# Example
```julia
using Revel
kernel_weight(PowerLaw(1.0), 4.0)                      # 0.25
kernel_weight(PowerLaw(0.5; offset=1.0), 0.0)          # 1.0
kernel_weight(PowerLaw(1.0; support=100.0), 200.0)     # 0.0
```
"""
struct PowerLaw <: AbstractMemory
    exponent::Float64
    offset::Float64
    support::Float64
    function PowerLaw(exponent::Real; offset::Real=0.0, support::Real=Inf)
        exponent > 0 || throw(ArgumentError("exponent must be positive, got $exponent"))
        offset >= 0 || throw(ArgumentError("offset must be non-negative, got $offset"))
        support > 0 || throw(ArgumentError("support must be positive, got $support"))
        new(Float64(exponent), Float64(offset), Float64(support))
    end
end

"""
    LinearDecay(span) <: AbstractMemory

Linear decay: an event of age `a` weighs `max(0, 1 - a/span)` — the linear
decay of Arena, Mulder & Leenders (2023, eq. 8), whose parameter is the
half-life, the age at which the weight is 1/2: `span` is twice their θ. A
profile over `span` and one over their half-life are the same profile on
different scales.

# Example
```julia
using Revel
kernel_weight(LinearDecay(10.0), 2.5)    # 0.75
kernel_weight(LinearDecay(10.0), 12.0)   # 0.0
```
"""
struct LinearDecay <: AbstractMemory
    span::Float64
    function LinearDecay(span::Real)
        span > 0 || throw(ArgumentError("span must be positive, got $span"))
        new(Float64(span))
    end
end

"""
    KernelMemory(f; support=Inf) <: AbstractMemory

A user-supplied kernel: `f(age)` is the weight of an event of age `age`.
`support` is the age beyond which `f` is zero (it lets the history scan stop
early); leave it at `Inf` when the kernel never vanishes.

# Example
```julia
using Revel
m = KernelMemory(a -> a <= 2 ? 1.0 : 0.5^(a - 2))    # a plateau, then decay
kernel_weight(m, 1.0), kernel_weight(m, 3.0)          # (1.0, 0.5)
```
"""
struct KernelMemory{F} <: AbstractMemory
    f::F
    support::Float64
    function KernelMemory(f::F; support::Real=Inf) where F
        support > 0 || throw(ArgumentError("support must be positive, got $support"))
        new{F}(f, Float64(support))
    end
end

"""
    kernel_weight(memory::AbstractMemory, age::Real) -> Float64

The weight a past event of age `age` (time units, or events when the layer's
clock is `:order`) carries under `memory`.

# Example
```julia
using Revel
kernel_weight(HalfLife(2.0), 4.0)        # 0.25
kernel_weight(IntervalMemory(0.0, 1.0), 0.5)   # 1.0
```
"""
kernel_weight(::FullMemory, age::Real) = 1.0
function kernel_weight(m::HalfLife, age::Real)
    w = isfinite(m.halflife) ? exp(-age * log(2) / m.halflife) : 1.0
    return m.normalized ? w * log(2) / m.halflife : w
end
kernel_weight(m::Window, age::Real) = age <= m.width ? 1.0 : 0.0
kernel_weight(m::IntervalMemory, age::Real) = (m.lo < age <= m.hi) ? 1.0 : 0.0
function kernel_weight(m::PowerLaw, age::Real)
    age > m.support && return 0.0
    base = age + m.offset
    base > 0 || throw(ArgumentError(
        "PowerLaw memory is undefined for an event of age 0 (a past event tied " *
        "with the one being explained); construct it with a positive `offset`, " *
        "e.g. PowerLaw($(m.exponent); offset=1.0)"))
    return base^(-m.exponent)
end
kernel_weight(m::LinearDecay, age::Real) = max(0.0, 1.0 - age / m.span)
kernel_weight(m::KernelMemory, age::Real) = Float64(m.f(age))

# The age beyond which an event contributes nothing (the history scan stops there)
_support(::AbstractMemory) = Inf
_support(m::Window) = m.width
_support(m::IntervalMemory) = m.hi
_support(m::LinearDecay) = m.span
_support(m::KernelMemory) = m.support
_support(m::PowerLaw) = m.support

# Accumulating kernels are absorbed once per event into running totals; every
# other kernel is re-read off the retained event list when the clock moves.
_accumulates(::AbstractMemory) = false
_accumulates(::FullMemory) = true
_accumulates(::HalfLife) = true

# Statistics under this kernel do not change between events (what the exact-time
# likelihood needs; see `is_interval_constant`)
_memory_interval_constant(::AbstractMemory) = false
_memory_interval_constant(::FullMemory) = true
_memory_interval_constant(m::HalfLife) = !isfinite(m.halflife)

# … and under this layer: on the event clock (`clock=:order`) every kernel's
# ages change only when an event happens
_layer_interval_constant(L) = L.clock === :order || _memory_interval_constant(L.memory)

_memory_label(::FullMemory) = ""
_memory_label(m::HalfLife) = "halflife=$(m.halflife)" * (m.normalized ? ",normalized" : "")
_memory_label(m::Window) = "window=$(m.width)"
_memory_label(m::IntervalMemory) = "interval=($(m.lo),$(m.hi)]"
_memory_label(m::PowerLaw) = "powerlaw=$(m.exponent)" *
                             (isfinite(m.support) ? ",support=$(m.support)" : "")
_memory_label(m::LinearDecay) = "linear=$(m.span)"
_memory_label(::KernelMemory) = "kernel"

"""
    interval_partition(breaks) -> Vector{IntervalMemory}

The adjacent [`IntervalMemory`](@ref) kernels `(b₁, b₂], (b₂, b₃], …` cut by the
increasing `breaks`. Fitting one copy of an effect per interval gives the
piecewise-constant decay profile of Perry & Wolfe (2013), who used seven
intervals, and the stepwise interval effects of Arena, Mulder & Leenders (2024).

# Example
```julia
using Revel
parts = interval_partition([0.0, 1.0, 24.0, 168.0])
length(parts)                                  # 3
[Inertia(memory=m) for m in parts]             # one inertia effect per interval
```
"""
function interval_partition(breaks::AbstractVector{<:Real})
    length(breaks) >= 2 || throw(ArgumentError("need at least two break points"))
    issorted(breaks; lt=<=) || throw(ArgumentError("breaks must be strictly increasing"))
    return [IntervalMemory(breaks[k], breaks[k + 1]) for k in 1:(length(breaks) - 1)]
end
