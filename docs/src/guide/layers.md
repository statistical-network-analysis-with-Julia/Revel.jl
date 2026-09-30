# Memory and Layers

Every endogenous effect reads a weighted network `w_t(i, j)`: how much past
interaction from `i` to `j` still counts at time `t`. The rule that produces
that network is an [`EventLayer`](@ref), and it is independent of what the
effect then does with it. This is the part of a specification that differs most
between papers and between packages, and the part most often left implicit.

```julia
using Revel

events = [Event(1, 2, 0.0), Event(1, 2, 6.0), Event(2, 1, 9.0), Event(1, 3, 10.0)]
history = build_history(events)
t = 12.0
```

## Memory kernels

How much an event weighs depends on its age. Six kernels cover the memory
devices in the literature:

| Kernel | Weight of an event of age `a` | Source |
|---|---|---|
| [`FullMemory`](@ref)`()` | `1` | Butts 2008 |
| [`HalfLife`](@ref)`(h)` | `exp(-a·ln2/h)` | Brandes, Lerner & Snijders 2009 |
| [`Window`](@ref)`(w)` | `1` if `a ≤ w` | de Nooy 2011; Quintane et al. 2013 |
| [`Interval`](@ref)`(lo, hi)` | `1` if `lo < a ≤ hi` | Perry & Wolfe 2013 |
| [`PowerLaw`](@ref)`(α)` | `a^(-α)` | Vu, Lomi, Mascia & Pallotti 2017 |
| [`LinearDecay`](@ref)`(s)` | `max(0, 1 − a/s)` | Arena, Mulder & Leenders 2023 |

and [`KernelMemory`](@ref) takes any function of the age.

```julia
compute(Inertia(), history, 1, 2, t)                          # 2.0
compute(Inertia(memory=HalfLife(6.0)), history, 1, 2, t)      # 0.25 + 0.5
compute(Inertia(memory=Window(6.0)), history, 1, 2, t)        # 1.0 — age 6 still counts
compute(Inertia(memory=Interval(6.0, 12.0)), history, 1, 2, t)  # 1.0 — only the first
compute(Inertia(memory=LinearDecay(12.0)), history, 1, 2, t)  # 0.0 + 0.5
```

Three conventions are worth knowing, because a fixture generated under one does
not validate the other:

- **Two normalisations of the exponential kernel coexist.** `HalfLife(h)` is the
  plain weight (Lerner & Lomi 2020; Arena et al. 2023; remstats).
  `HalfLife(h; normalized=true)` multiplies it by `ln2/h` (Brandes et al. 2009;
  the rem package). They rescale the coefficient by a constant.
- **A window is closed, an interval half-open.** An event exactly `w` old is
  inside `Window(w)`; `Interval(lo, hi)` excludes `lo` and includes `hi`, so
  adjacent intervals partition the past.
- **The decay is evaluated at the time of the event being explained.** remstats
  4.1.0 evaluates it at the time of the *previous* event, contrary to its
  documentation; see the [concordance](concordance.md).

### Estimating the memory instead of assuming it

Half-lives and windows in applied work span orders of magnitude, and a closure
estimate can move from zero to above one across plausible values.
[`interval_partition`](@ref) gives the piecewise-constant decay profile of Perry
& Wolfe — one copy of an effect per interval, each with its own coefficient:

```julia
profile = [Inertia(memory=m) for m in interval_partition([0.0, 1.0, 24.0, 168.0])]
name.(profile)
```

and [`profile_memory`](@ref) profiles the likelihood over a single memory
parameter (see [Fitting](fitting.md)).

## Which events count

```julia
typed = [Event(1, 2, 1.0; eventtype=:praise), Event(1, 2, 2.0; eventtype=:blame, weight=3.0)]
h2 = build_history(typed)

compute(Inertia(types=:praise), h2, 1, 2, 3.0)                     # 1.0
compute(Inertia(weighted=true), h2, 1, 2, 3.0)                     # 4.0
compute(Inertia(keep=(s, r, t, w, ty) -> w > 2, name="heavy"), h2, 1, 2, 3.0)   # 1.0
```

- `types` restricts the layer to some **past** event types — the type-split
  interaction.
- `weighted=true` adds event weights instead of counting events.
- `keep` is an arbitrary predicate on the past event — the attribute-filtered
  interaction of Brandenberger's rem package.

## The clock

`clock=:order` measures age in events rather than in clock units: the most
recent event is one event old. Use it for order-only data, and for a half-life
stated in events.

```julia
compute(Inertia(memory=HalfLife(2.0), clock=:order), history, 1, 2, t)   # 0.5^2 + 0.5^1.5
```

## Undirected events

`symmetric=true` records each event for both of its participants, so that the
layer is an undirected weighted network. On such a layer an actor's out-, in-
and total degree are the same number: the events it took part in.

```julia
compute(Inertia(symmetric=true), history, 2, 1, t)             # 3.0
compute(TotaldegreeSender(symmetric=true), history, 3, 1, t)   # 1.0
```

## Sharing a layer

Each constructor builds a private layer from its keywords. To make several
effects read one index of the history — less memory, one pass — build the layer
once and pass it:

```julia
recent = EventLayer(memory=HalfLife(30.0))
stats = [Inertia(layer=recent), Reciprocation(layer=recent), OTP(layer=recent)]
```

A layer is a mutable cache of the history it was last evaluated on. Do not share
one layer, or one statistic, between tasks that fit concurrently.

## Cost

[`FullMemory`](@ref) and [`HalfLife`](@ref) are accumulated: each event is
absorbed once and every read is `O(1)`. The other kernels are re-read off the
retained events whenever the clock moves, walking back only as far as the
kernel's support — a window costs the events inside it. Storage is dense in the
number of actors (a few `n × n` matrices per layer).
