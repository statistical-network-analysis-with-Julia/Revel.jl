# Endogenous Effects

An endogenous effect is a function of the event history. Revel groups them by
the configuration they measure; every one of them accepts the measurement
keywords of an [`EventLayer`](@ref) (`memory`, `types`, `weighted`, `keep`,
`clock`, `symmetric`), a `transform`, and a `name`.

```julia
using Revel

events = [Event(1, 2, 1.0), Event(1, 3, 2.0), Event(3, 2, 3.0), Event(2, 1, 4.0),
          Event(1, 2, 5.0), Event(4, 2, 6.0)]
history = build_history(events)
t = 7.0
```

## The dyad and the reversed dyad

| Effect | Value for a candidate `s → r` |
|---|---|
| [`Inertia`](@ref) | weight of past `s → r` events |
| [`Reciprocation`](@ref) | weight of past `r → s` events |
| [`DyadActivity`](@ref) | both |

```julia
compute(Inertia(), history, 1, 2, t)          # 2.0
compute(Reciprocation(), history, 1, 2, t)    # 1.0
```

`scaling=:prop` turns a count into a share. For inertia it is Butts's (2008)
persistence — the share of the sender's past sends that went to this receiver:

```julia
compute(Inertia(scaling=:prop), history, 1, 2, t)                      # 2/3
compute(Reciprocation(scaling=:prop), history, 1, 2, t)                # 1.0
compute(Inertia(scaling=:prop, empty=1/3), history, 4, 1, t)           # 0.0 — 4 has sent
```

The value for an actor with no history is a convention that differs between
packages (relevent and remstats use `1/(n−1)`), so it is the `empty` keyword
and not a hidden default.

## Node degree

Six effects cross the actor (sender or receiver) with the degree (out, in,
total): [`OutdegreeSender`](@ref), [`IndegreeSender`](@ref),
[`TotaldegreeSender`](@ref), [`OutdegreeReceiver`](@ref),
[`IndegreeReceiver`](@ref), [`TotaldegreeReceiver`](@ref).

```julia
compute(OutdegreeSender(), history, 1, 2, t)                  # 3.0 — activity
compute(IndegreeReceiver(), history, 1, 2, t)                 # 4.0 — popularity
compute(IndegreeReceiver(measure=:partners), history, 1, 2, t)  # 3.0 distinct senders
compute(TotaldegreeReceiver(scaling=:prop), history, 1, 2, t) # 5/12: preferential attachment
```

`measure=:partners` counts distinct partners instead of events — the
degree/intensity distinction of the Lomi–Vu line. Dyad-level combinations are
[`TotaldegreeDyad`](@ref), [`DegreeMin`](@ref), [`DegreeMax`](@ref),
[`DegreeDiff`](@ref) and [`DegreeAssortativity`](@ref).

## The two-path

Four effects, read off the third actors `k` that link sender and receiver:

| Effect | Configuration | Also known as |
|---|---|---|
| [`OTP`](@ref) | `s → k → r` | transitive closure; goldfish `trans` |
| [`ITP`](@ref) | `r → k → s` | cyclic closure; goldfish `cycle` |
| [`OSP`](@ref) | `s → k ← r` | shared targets; goldfish `commonReceiver` |
| [`ISP`](@ref) | `s ← k → r` | shared sources; goldfish `commonSender` |

```julia
compute(OTP(), history, 1, 2, t)    # 1.0 — through 3
compute(ISP(), history, 3, 2, t)    # 1.0 — 1 sent to both
```

Packages agree on the configuration and disagree on how the two legs of a path
are combined, which is why a triadic coefficient cannot be compared across
them. `combine` selects the version:

```julia
two = build_history([Event(1, 3, 1.0), Event(1, 3, 2.0), Event(3, 2, 3.0)])
compute(OTP(), two, 1, 2, 4.0)                              # min(2, 1) = 1  (relevent, remstats)
compute(OTP(combine=:product), two, 1, 2, 4.0)              # 2·1 = 2        (Perry & Wolfe)
compute(OTP(combine=:product, root=true), two, 1, 2, 4.0)   # √2             (rem)
compute(OTP(combine=:count), two, 1, 2, 4.0)                # 1 intermediary (goldfish)
```

All four are forms of [`TwoPathEffect`](@ref), whose two legs may read
**different layers**. That gives cross-network closure (goldfish's `mixedTrans`),
the structural-balance statistics of Brandes, Lerner & Snijders
([`BalanceEffect`](@ref)), and typed closure:

```julia
praise, blame = EventLayer(types=:praise), EventLayer(types=:blame)
enemy_of_friend = TwoPathEffect(praise, blame; name="praised_a_critic")
```

`ordered=true` keeps a two-path only if its first step happened before its
second, and `third=` weights the intermediary — see
[Interactions](interactions.md). [`SharedPartners`](@ref) is the undirected
version.

!!! warning "Closure is fragile"
    Closure is the second most reported effect in the literature, and the two
    studies that test it against actor heterogeneity and dyadic repetition both
    find that it largely disappears. Fit degree and repetition terms first, and
    ask with [`score_test`](@ref) whether a triadic term still has something to
    explain.

## The three-path

[`FourCycleEffect`](@ref) closes `s → a ← b → r`: another sender shares a
target with `s` and also targets `r`. It is the closure statistic of two-mode
networks, where triads do not exist.

```julia
twomode = build_history([Event(1, 11, 1.0), Event(2, 11, 2.0), Event(2, 12, 3.0)])
compute(FourCycleEffect(), twomode, 1, 12, 4.0)    # 1.0
```

## Order and recency

| Effect | Value |
|---|---|
| [`RecencyRank`](@ref)`(:receive)` | `1/rank` of `r` among those who most recently sent to `s` |
| [`RecencyRank`](@ref)`(:send)` | `1/rank` of `r` among those `s` most recently sent to |
| [`TimeSince`](@ref)`(target)` | a function of the time since the last event of a kind |
| `PShift(shift)` | Gibson's thirteen participation shifts (from Relevent.jl) |
| [`PShiftABAB`](@ref), [`UndirectedPShift`](@ref) | the shifts remstats adds |

```julia
compute(RecencyRank(:receive), history, 2, 4, t)   # 1.0 — 4 wrote to 2 last
compute(TimeSince(:dyad), history, 1, 2, t)        # 1/(2 + 1)
compute(PShift(:AB_BA), history, 2, 4, t)          # 1.0 — 2 answers 4
```

"Recency" names at least four statistics in the literature; `TimeSince` takes
the transform as an argument rather than choosing one.

## Neighbourhoods

[`NodeTransitivity`](@ref) counts the transitive structures an actor is the
source of (goldfish `nodeTrans`), and [`StructuralSimilarity`](@ref) compares
the partner profiles of sender and receiver (eventnet's Jaccard and cosine
statistics).

## Scaling and explosion

Cumulative statistics grow over a sequence. With a positive coefficient they
feed back on themselves, and a simulated process can lock into a single dyad.
The literature's two remedies are both available:

```julia
Inertia(transform=:log1p)             # log(1 + x)
Standardized(Inertia(), 8)            # z-score across the risk set, per event
```

A decaying memory is the third.
