# Relational Hyperevents

A meeting, a coauthored paper, a co-offence and an e-mail with five recipients
are single events involving *sets* of actors. A relational hyperevent model
(RHEM) puts the event rate on the hyperedge — the whole set — rather than on the
dyad (Lerner, Tranmer, Mowbray & Hâncean 2019; Lerner, Lomi, Mowbray, Rollings &
Tranmer 2021; Lerner & Lomi 2023). This page covers the hyperevent module of
Revel: the data structure, the statistics, the sampling of non-events, fitting,
and what is not implemented.

```julia
using Revel, Random
```

## When a dyadic model is not enough

The usual workaround is to break a hyperevent into dyadic events — a meeting of
three becomes three pairs. That expansion loses information, and no dyadic
statistic can recover it. Take two histories: in the first, actors 1, 2 and 3
met once as a group; in the second, actors 4, 5 and 6 met in three separate
pairs. After the expansion both are a triangle with every pair weight equal to
one.

```julia
history = build_hyper_history([
    HyperEvent([1, 2, 3], 1.0),                                        # one meeting of three
    HyperEvent([4, 5], 2.0), HyperEvent([4, 6], 3.0), HyperEvent([5, 6], 4.0)])

# what a dyadic model sees: every pair has met once, in both triples
compute(SubsetRepetition(2), history, [1, 2, 3], Int[], 5.0)    # 1.0
compute(SubsetRepetition(2), history, [4, 5, 6], Int[], 5.0)    # 1.0

# what it cannot see: only one of the triples has ever met as a triple
compute(SubsetRepetition(3), history, [1, 2, 3], Int[], 5.0)    # 1.0
compute(SubsetRepetition(3), history, [4, 5, 6], Int[], 5.0)    # 0.0
compute(ExactRepetition(), history, [1, 2, 3], Int[], 5.0)      # 1.0
compute(ExactRepetition(), history, [4, 5, 6], Int[], 5.0)      # 0.0
```

Exact repetition and subset repetition of order three or more "introduce
dependencies that cannot be expressed in models specifying only dyadic event
rates" (Lerner et al. 2019). For directed hyperevents Lerner & Lomi (2023)
classify their effects the same way: reciprocation, out-in popularity and the
four triadic effects are sums of dyadic covariates over the receivers, whereas
exact and unordered repetition, partial receiver-set repetition of order above
one and interaction among receivers are not. A hyperevent model is needed when

- the *composition* of the group is the outcome — who is invited together, which
  team forms, which receiver list an e-mail goes to;
- familiarity is a property of subgroups: do pairs, triads or whole teams that
  worked together before do so again;
- the *size* of the event matters, or moderates other effects;
- duplicating a multi-actor event into pairs would inflate the number of
  observations and make the pairs of one event look like independent events.

When every event has exactly two participants the hyperevent statistics reduce
to the dyadic ones: subset repetition of order `(1, 1)` on one-sender,
one-receiver hyperevents is [`Inertia`](@ref), and exact repetition on
two-actor undirected hyperevents is `Inertia(symmetric=true)`.

## Hyperevents and their history

A [`HyperEvent`](@ref) is undirected when it is given one actor set and directed
when it is given two:

```julia
meeting = HyperEvent([3, 1, 2], 1.0)                 # participants, time
meeting.senders, meeting.receivers                   # ([1, 2, 3], Int64[])

mail = HyperEvent([1], [4, 2], 2.0; eventtype=:email, weight=1.0)   # senders, receivers, time
is_directed(mail), participants(mail)                # (true, [1, 2, 4])
```

Actor sets are stored sorted. An undirected hyperevent keeps its participants in
`senders` and has no receivers; a directed one usually has one sender and a set
of receivers (the multicast events of Perry & Wolfe 2013), though several
senders are allowed. An actor cannot be both sender and receiver. `weight` holds
an event weight or an outcome; `eventtype` is what the `types=` filter of a
statistic matches.

A [`HyperHistory`](@ref) is the past against which a statistic is evaluated.
[`build_hyper_history`](@ref) makes one from a vector of events and
[`update_hyper_history!`](@ref) appends to it; a history holds either directed or
undirected hyperevents, not both.

```julia
h = build_hyper_history([HyperEvent([1, 2, 3], 1.0), HyperEvent([1, 2], 2.0),
                         HyperEvent([2, 3, 4], 3.0)])
length(h)                                                  # 3
# the candidate hyperedge is (senders, receivers); receivers is empty when undirected
compute(SubsetRepetition(2), h, [1, 2, 3], Int[], 4.0)     # 5/3
compute(SubsetRepetition(2), h, HyperEvent([1, 2, 3], 4.0))   # the same call
```

Hyperevent statistics are subtypes of [`AbstractHyperStatistic`](@ref) and add
methods to the ecosystem's shared `compute` and `name` generics, with the
signature `compute(stat, history, senders, receivers, time)`.

## The algebra of hyperedge statistics

The literature's vocabulary is better read as an algebra than as a list. Past
events are first aggregated into a **hyperedge attribute**,

| attribute | definition |
|---|---|
| activity | `Σ w(t − tₑ) · χ(hₑ = h)` — past events on *exactly* `h` |
| degree | `Σ w(t − tₑ) · χ(h ⊆ hₑ)` — past events on `h` or any superset |
| directed degree | `Σ w(t − tₑ) · χ(a′ ⊆ Iₑ ∧ b′ ⊆ Jₑ)` — `a′` among the senders `Iₑ`, `b′` among the receivers `Jₑ` |
| outcome-weighted degree | `Σ w(t − tₑ) · yₑ · χ(h ⊆ hₑ)` |

and the attribute is then aggregated over the **sub-hyperedges of a given
order** of the candidate. Four choices are therefore orthogonal, and every
endogenous statistic on this page takes them as keywords:

- the subset order — a positional argument, `p` or `(p, q)`;
- `aggregate` — `:mean` (the default), `:sum`, `:min`, `:max`, `:sd`, `:absdiff`;
- `memory` — any memory kernel of the package ([`FullMemory`](@ref),
  [`HalfLife`](@ref), [`Window`](@ref), [`Interval`](@ref), [`PowerLaw`](@ref),
  [`LinearDecay`](@ref), [`KernelMemory`](@ref)). The papers use an exponential
  half-life following Brandes, Lerner & Snijders (2009);
- `weighted` and `types` — add each past event's `weight` instead of counting
  it, and restrict the past to some event types.

`transform` (`:log1p`, `:sqrt`, a function) rescales the result and `name`
overrides the column name.

```julia
compute(SubsetRepetition(2; aggregate=:sum), h, [1, 2, 3], Int[], 4.0)            # 5.0
compute(SubsetRepetition(2; memory=HalfLife(1.0)), h, [1, 2], Int[], 4.0)         # 1/8 + 1/4
compute(SubsetRepetition(2; memory=Window(1.5)), h, [2, 3], Int[], 4.0)           # 1.0
compute(SubsetRepetition(2; transform=:log1p), h, [1, 2, 3], Int[], 4.0)          # log(1 + 5/3)
name(SubsetRepetition(2; memory=HalfLife(30.0), transform=:sqrt))   # "sqrt(subrep(2)[halflife=30.0])"
```

The history indexes each subset order the first time a statistic asks for it:
for every sub-hyperedge, the list of past events containing it. A statistic is
read as `Σ kernel_weight(memory, t − tₑ)·wₑ` over that list, walking backwards
and stopping at the kernel's support, so one index serves every memory kernel
and type filter. Indexing order `p` costs `binomial(size, p)` entries per event —
the price of high-order subset repetition on large hyperedges.

## Statistics for undirected hyperevents

| Statistic | Value on the candidate hyperedge `h` | Source |
|---|---|---|
| [`HyperedgeSize`](@ref) | `abs(h)`, the number of participants | Lerner et al. 2019 |
| [`ExactRepetition`](@ref) | `activity(h)` | Lerner et al. 2019 |
| [`SubsetRepetition`](@ref)`(p)` | mean over `p`-subsets `h′ ⊆ h` of `deg(h′)` | Lerner et al. 2019, 2021; Lerner & Hâncean 2023 |
| `SubsetRepetition(p; aggregate=:sum)` | `Σ deg(h′)`, unnormalised | Lerner, Hâncean & Perc 2025 |
| `SubsetRepetition(p; aggregate=:assortativity)` | `−Σ abs(deg(h′) − deg(h″))` over pairs of `p`-subsets | Lerner, Hâncean & Perc 2025 |
| [`SharedPriorEvents`](@ref)`(p)` | `Σ w · χ(abs(hₑ ∩ h) ≥ p)` | Lerner et al. 2019 |
| [`HyperClosure`](@ref) | mean over pairs `{u, v} ⊆ h` of `Σ_w min[deg({u, w}), deg({v, w})]` | Lerner et al. 2021 |
| [`PriorSuccess`](@ref)`(p)` | `Σ performance(h′) / Σ deg(h′)` over `p`-subsets | Lerner & Hâncean 2023 |
| [`HyperCovariate`](@ref) | a summary of an actor covariate over `h` | eventnet; Lerner et al. 2021 |

**Subset repetition** of order one is individual activity (preferential
attachment), order two dyadic familiarity, order three triadic familiarity.
Lower orders belong in the model alongside higher ones; Lerner & Lomi (2023)
report that the order-one effect turns non-significant once higher orders enter.

```julia
# deg(1) = 2, deg(2) = 3, deg(3) = 2
compute(SubsetRepetition(1), h, [1, 2, 3], Int[], 4.0)                     # 7/3
# deg{1,2} = 2, deg{1,3} = 1, deg{2,3} = 2
compute(SubsetRepetition(2), h, [1, 2, 3], Int[], 4.0)                     # 5/3
compute(SubsetRepetition(3), h, [1, 2, 3], Int[], 4.0)                     # 1.0
# each past event once, however large the overlap
compute(SharedPriorEvents(2), h, [1, 2, 3], Int[], 4.0)                    # 3.0
```

**Closure** is built on the dyadic projection of the past hyperevents, `W(u, w)
= deg({u, w})`: for each pair of the candidate, the two-paths through third
actors `w`, each valued by the weaker of its two legs. The papers find a
*negative* closure effect next to positive subset repetition — overlapping but
stable groups that do not merge (Lerner et al. 2021; Lerner & Hâncean 2023).

```julia
# pair {1,4}: through 2, min(W12, W42) = min(2, 1) = 1; through 3, min(W13, W43) = 1
compute(HyperClosure(), h, [1, 4], Int[], 4.0)                             # 2.0
compute(HyperClosure(combine=:product), h, [1, 4], Int[], 4.0)             # 2·1 + 1·1 = 3.0
compute(HyperClosure(normalize=:thirds, n_actors=4), h, [1, 4], Int[], 4.0)   # 2/(4 − 2) = 1.0
```

`combine` is `:min` (the papers; the default) or `:product` (eventnet's
default); `parallel` combines the paths through different third actors by `:sum`
or `:max`; `aggregate` combines the pairs.

**Prior success** reads the event `weight` as the outcome of a past hyperevent:

```julia
papers = build_hyper_history([HyperEvent([1, 2], 1.0; weight=10.0),
                              HyperEvent([2, 3], 2.0; weight=4.0)])
compute(PriorSuccess(1), papers, [1, 2, 3], Int[], 3.0)    # (10 + 14 + 4)/(1 + 2 + 1) = 7.0
compute(PriorSuccess(2), papers, [1, 2, 3], Int[], 3.0)    # (10 + 0 + 4)/(1 + 0 + 1) = 7.0
# the outcome-weighted hyperedge degree alone
compute(SubsetRepetition(2; weighted=true, aggregate=:sum), papers, [1, 2, 3], Int[], 3.0)   # 14.0
```

**Hyperedge size** has no counterpart in a dyadic model, and a caveat of its
own: the design below compares each event with alternatives of the *same* size,
so a statistic that depends on the size alone is constant within every stratum
and cannot be estimated. Use it as a moderator:

```julia
by_size = Interaction(HyperedgeSize(), SubsetRepetition(2))
compute(by_size, h, [1, 2, 3], Int[], 4.0)                 # 3 × 5/3 = 5.0
name(by_size)                                              # "size:subrep(2)"
```

[`Interaction`](@ref) and [`Transformed`](@ref) accept hyperevent statistics as
they accept dyadic ones.

## Statistics for directed hyperevents

For a candidate with senders `a` and receivers `b`,
[`DirectedSubsetRepetition`](@ref)`(p, q)` averages the directed degree over the
sub-hyperedges with `p` of the senders and `q` of the receivers. `direction=:in`
takes the sub-hyperedges of the *reversed* hyperedge (subset reciprocation), and
`direction=:sym` ignores roles. The named effects of Lerner & Lomi (2023), where
a hyperevent has one sender `i` and a receiver set `J`, are thin constructors:

| Statistic | Order, direction | Definition (Lerner & Lomi 2023 unless noted) |
|---|---|---|
| [`ExactRepetition`](@ref) | — | `Σ w · 1(iₘ = i ∧ Jₘ = J)` |
| [`UnorderedRepetition`](@ref) | — | `Σ w · 1({iₘ} ∪ Jₘ = {i} ∪ J)`, "reply to all" |
| `ExactRepetition(direction=:in)` | — | exact reciprocation (the "reciprocation" of Lerner et al. 2019) |
| [`ReceiverSetRepetition`](@ref)`(p)` | `(0, p)`, `:out` | mean over `J′ ⊆ J` of `Σ w · 1(J′ ⊆ Jₘ)` |
| [`SenderReceiverSetRepetition`](@ref)`(p)` | `(1, p)`, `:out` | mean over `J′ ⊆ J` of `Σ w · 1(i = iₘ ∧ J′ ⊆ Jₘ)` |
| [`HyperSenderActivity`](@ref) | `(1, 0)`, `:out` | past events sent (Lerner et al. 2019) |
| [`HyperReceiverPopularity`](@ref) | `(0, 1)`, `:out` | mean in-degree of the receivers |
| [`HyperReciprocation`](@ref) | `(1, 1)`, `:in` | `Σ_{j ∈ J} hy_deg(j, {i}) / abs(J)` |
| [`OutInPopularity`](@ref) | `(1, 0)`, `:in` | `Σ_{j ∈ J} deg_out(j) / abs(J)` |
| [`InteractionAmongReceivers`](@ref)`(p)` | — | `Σ_{j ∈ J, J′ ⊆ J∖{j}} hy_deg(j, J′) / (abs(J) · C(abs(J) − 1, p))` |

```julia
mails = build_hyper_history([HyperEvent([1], [2, 3], 1.0), HyperEvent([1], [2, 4], 2.0),
                             HyperEvent([2], [1, 3], 3.0), HyperEvent([4], [2, 3], 4.0)])
S, R = [1], [2, 3]                      # candidate: 1 → {2, 3}
compute(ExactRepetition(), mails, S, R, 5.0)                 # 1.0 — the first mail
compute(UnorderedRepetition(), mails, S, R, 5.0)             # 2.0 — mails 1 and 3 were among {1,2,3}
compute(SenderReceiverSetRepetition(1), mails, S, R, 5.0)    # (2 + 1)/2: 1 → 2 twice, 1 → 3 once
compute(SenderReceiverSetRepetition(2), mails, S, R, 5.0)    # 1.0 — 1 addressed {2,3} together once
compute(ReceiverSetRepetition(2), mails, S, R, 5.0)          # 2.0 — {2,3} co-received twice
compute(HyperReciprocation(), mails, S, R, 5.0)              # (1 + 0)/2: 2 replied to 1, 3 did not
compute(OutInPopularity(), mails, S, R, 5.0)                 # (1 + 0)/2: 2 has sent once, 3 never
compute(InteractionAmongReceivers(1), mails, S, R, 5.0)      # (1 + 0)/2: 2 → 3 once, 3 → 2 never
```

[`HyperClosure`](@ref) takes the direction of each leg, read on the projection
`W(i, j) = Σ w · χ(i ∈ Iₑ ∧ j ∈ Jₑ)`. For a sender `i`, a receiver `j` and a
third actor `k`:

| `HyperClosure(kind)` | legs | names in the literature |
|---|---|---|
| `:transitive` (the default directions) | `W(i,k)`, `W(k,j)` | transitive closure |
| `:cyclic` | `W(k,i)`, `W(j,k)` | cyclic closure |
| `:shared_senders` | `W(k,i)`, `W(k,j)` | shared senders (2019), incoming balance (2023), sender balance, sibling |
| `:shared_receivers` | `W(i,k)`, `W(j,k)` | shared receivers (2019), outgoing balance (2023), receiver balance, cosibling |

```julia
# 1 → {3}: through 2, min(W12, W23) = min(2, 1); through 4, min(W14, W43) = min(1, 1)
compute(HyperClosure(:transitive), mails, [1], [3], 5.0)        # 2.0
# through 2, min(W21, W23) = 1; actor 4 never addressed 1
compute(HyperClosure(:shared_senders), mails, [1], [3], 5.0)    # 1.0
```

## Covariates on a hyperedge

[`HyperCovariate`](@ref) summarises an actor covariate — a vector indexed by
actor, or a [`Covariate`](@ref), which may vary over time — over the candidate.
`endpoint` selects the participants (`:all`), the `:senders` or the `:receivers`;
`aggregate` is `:mean`, `:sum`, `:min`, `:max`, `:sd`, `:samplesd`, `:absdiff`
(mean absolute difference over pairs) or `:catdiff` (share of pairs with
different values, for a categorical covariate). With `endpoint=:all` on a
directed hyperedge the pairs of `:absdiff` and `:catdiff` are the (sender,
receiver) pairs, as in eventnet.

| Effect | Arguments |
|---|---|
| covariate average (Lerner et al. 2021) | `aggregate=:mean` |
| covariate homogeneity / heterophily (Lerner et al. 2021) | `aggregate=:absdiff` or `:sd` |
| receiver-set average (Lerner & Lomi 2023) | `endpoint=:receivers, aggregate=:mean` |
| sender–receiver heterophily (Lerner & Lomi 2023) | `endpoint=:all, aggregate=:absdiff` on directed hyperedges |
| receiver-set heterophily (Lerner & Lomi 2023) | `endpoint=:receivers, aggregate=:absdiff` |

```julia
age = [30.0, 40.0, 50.0, 20.0]
compute(HyperCovariate(age), h, [1, 2, 3], Int[], 4.0)                              # 40.0
compute(HyperCovariate(age; aggregate=:absdiff), h, [1, 2, 3], Int[], 4.0)          # (10 + 20 + 10)/3
compute(HyperCovariate(age; endpoint=:receivers), mails, [1], [2, 4], 5.0)          # 30.0
compute(HyperCovariate(age; aggregate=:absdiff), mails, [1], [2, 4], 5.0)           # (10 + 10)/2
compute(HyperCovariate(age; endpoint=:receivers, aggregate=:absdiff), mails, [1], [2, 4], 5.0)   # 20.0
```

## Sampling non-events

Among `n` actors there are `binomial(n, p)` hyperedges of size `p`: about
`1.7 × 10¹³` groups of ten among a hundred actors. The risk set cannot be
enumerated, so the model is estimated from a **case-control design** (Lerner &
Lomi 2023; eventnet): each observed hyperevent is compared with a few
alternative hyperedges *of the same size*, drawn uniformly without replacement
from those the actors at risk could have formed. Conditioning on the size
corresponds to a baseline rate stratified by hyperedge size.

[`hyper_design`](@ref) builds that design as a `DataFrame`:

```julia
events = [HyperEvent([1, 2], 1.0), HyperEvent([1, 2, 3], 2.0), HyperEvent([1, 2], 3.0)]
design = hyper_design(events, [SubsetRepetition(2)], 5; n_controls=3, rng=Xoshiro(1))
size(design, 1)                           # 12 — three events × (1 case + 3 controls)
design.risk_set_size[design.is_event]     # [10, 10, 10] — C(5,2), C(5,3), C(5,2)
design.sampling_prob[design.is_event]     # [1/3, 1/3, 1/3] — 3 of the 9 alternatives
design[design.is_event, "subrep(2)"]      # [0.0, 1/3, 2.0] — read before each event
```

The columns are those `REM.fit_rem(::DataFrame, names)` reads (`event_index`,
`is_event`, `stratum`, `risk_set_size`, `sampling_prob`, `tie_weight`), `time`,
the list columns `senders` and `receivers` describing each row's hyperedge, and
one column per statistic. The frame is also what to take to another estimator.

- When the alternatives of a size number `n_controls` or fewer they are **all**
  included, and the conditional likelihood is exact for the size-stratified risk
  set.
- `risk_set_size` is the number of possible hyperedges of the observed size —
  `C(n, p)` undirected, `C(n, p)·C(n − p, q)` directed — and saturates at
  `typemax(Int)` instead of overflowing.
- `sampler=:receivers` keeps the observed senders and draws receiver sets only.
  This is the design of Lerner & Lomi (2023), whose baseline is stratified by
  sender and receiver-set size; a statistic of the senders alone is then
  constant within every stratum.
- `actors` restricts the actors at risk; `ties` is `:error` (the default),
  `:ordered` or `:breslow`, with the meaning they have in
  [`each_risk_set`](@ref). `:efron` and `:batch` are refused with the reason.
- All randomness comes from `rng`.

## Fitting

[`fit_rhem`](@ref) (alias [`rhem`](@ref)) builds the design and fits it with
`REM.fit_rem`, the ecosystem's conditional logit — Revel hosts no optimizer of
its own. The result is a [`HyperFit`](@ref), which answers the StatsAPI verbs
and the result-metadata protocol by forwarding them to the underlying fit.

```julia
x = [0.0, 1.0, 0.0, 1.0, 0.5, -1.0, 2.0, 0.3]
truth = [SubsetRepetition(2; memory=HalfLife(30.0), transform=:log1p),
         HyperCovariate(x; name="x")]
meetings = simulate_hyperevents(truth, [0.8, 0.6], 8, 400; sizes=[2, 3, 4], rng=Xoshiro(1))

fit = fit_rhem(meetings, truth, 8; n_controls=20, rng=Xoshiro(2))
coefnames(fit)                       # ["log1p(subrep(2)[halflife=30.0])", "x"]
round.(coef(fit); digits=1)          # close to [0.8, 0.6]
stderror(fit)
nobs(fit), fit.n_controls            # (400, 20)
```

Sampling non-events keeps the estimator consistent, but the estimates vary with
the draw of controls: fix `rng` for reproducibility and raise `n_controls` to
reduce that variation. `se=:sandwich` requests event-clustered standard errors.
Non-convergence, collinearity and separation are reported by `REM.fit_rem` as
for any other model.

The guidance of the dyadic literature carries over. Cumulative statistics grow
without bound, so scale them — the papers apply a square root or `log1p` — or
use a decaying memory; and a model with subset repetition of order `p` should
include the lower orders.

```julia
richer = [SubsetRepetition(1; transform=:log1p), SubsetRepetition(2; transform=:log1p),
          HyperClosure(transform=:log1p),
          Interaction(HyperedgeSize(), SubsetRepetition(2; transform=:log1p))]
fit2 = rhem(meetings, richer, 8; n_controls=20, rng=Xoshiro(3))
length(coef(fit2))                   # 4
```

A directed model is fitted the same way. `sampler=:receivers` gives the
receiver-choice design of Lerner & Lomi (2023):

```julia
dstats = [SenderReceiverSetRepetition(1; transform=:log1p),
          ReceiverSetRepetition(2; transform=:log1p),
          HyperReciprocation(transform=:log1p)]
sent = simulate_hyperevents(dstats, [0.7, 0.4, 0.5], 6, 300;
                            sizes=[(1, 2), (1, 3)], rng=Xoshiro(4))
dfit = fit_rhem(sent, dstats, 6; n_controls=15, sampler=:receivers, rng=Xoshiro(5))
length(coef(dfit))                   # 3
```

## Simulation

[`simulate_hyperevents`](@ref) draws, at each step, the size of the next
hyperedge uniformly from `sizes` and then the hyperedge itself among those of
that size with probability proportional to `exp(θ′x)`. `sizes` holds integers
(undirected) or `(senders, receivers)` pairs (directed); repeating a size makes
it more frequent.

The choice is among *all* hyperedges of the drawn size while there are at most
`candidates` of them (1000 by default). Beyond that it is among `candidates`
hyperedges sampled uniformly, which is an approximation: keep the network small,
or raise `candidates`, when the simulation must follow the model exactly, as in
a parameter-recovery study.

```julia
sim = simulate_hyperevents([SubsetRepetition(2; transform=:log1p)], [1.0], 6, 50;
                           sizes=[2, 3, 3], rng=Xoshiro(6))
length(sim), sim[end].time           # (50, 50.0)
```

## Conventions that change the number

- **Mean or sum.** The 2019 to 2023 papers *average* the hyperedge degree over
  sub-hyperedges; Lerner, Hâncean & Perc (2025) *sum* it. `aggregate=:mean` is
  the default and `:sum` the alternative. Under the size-stratified design the
  two differ by a factor that is constant within a stratum, but the factor
  varies between strata of different sizes, so the coefficients are not
  interchangeable.
- **Closure normalisation.** The 2019 preprint divides closure by the number of
  possible third actors; the papers from 2021 onward do not.
  `HyperClosure(normalize=:thirds, n_actors=n)` gives the former (dividing by
  `n − 2`), the default `normalize=:none` the latter.
- **Min or product.** The papers combine the two legs of a two-path by their
  minimum; eventnet's closure statistic defaults to the product.
- **Terminology.** Sub-repetition (2019), subset repetition (2021), partial
  receiver-set repetition (2023) and `subrep` (2025) are one family; shared
  senders / shared receivers (2019), incoming / outgoing balance (2023), sender
  / receiver balance (Poda, Vinciotti & Wit 2025) and sibling / cosibling (Perry
  & Wolfe 2013) are one pair.
- **"Reply to all"** is operationalised twice: as unordered repetition (2023),
  which is [`UnorderedRepetition`](@ref), and as switch reciprocation of order
  one (2019), which is not implemented. They are different statistics.
- **The exponential kernel** has two normalisations; see
  [Memory and Layers](layers.md).

Three readings were chosen where the sources leave room, and are stated in the
docstrings: `direction=:sym` of [`DirectedSubsetRepetition`](@ref)`(p, q)` is
subset repetition of order `p + q` on the participant sets; the third actor of a
closure may be another member of the candidate (as in Lerner & Lomi 2023, `a ≠
i, j`); and the 2019 third-actor count is taken to be `n − 2`.

## Not implemented

Each of these is refused with an `ArgumentError` where there is an entry point
for it, rather than approximated:

- **Two-mode hyperevents** — authors publishing a paper that cites a set of
  papers, and the `subrep⁽ᵏ'ˡ⁾`, cocitation and tripartite closure effects built
  on them (Lerner, Hâncean & Lomi 2025; Espinosa-Rada, Lerner & Fritz 2025;
  Fabbrucci Barbagli et al. 2026). Senders and receivers are drawn from one set
  of actors at risk.
- **Geometrically weighted subset repetition** (Fabbrucci Barbagli, Lerner,
  Amati & De Stefano 2026; eventnet `*_GW_SUB_REP_STAT`).
- **Generalised hyperevents** (eventnet's `GHE_` family).
- **Closure of general order `(p, q, l)`** and **switch reciprocation of order
  `l`** (Lerner et al. 2019); only closure with single actors at the three
  corners is available.
- **Prior success disparity** (Lerner & Hâncean 2023) as a named statistic:
  `SubsetRepetition(1; weighted=true, aggregate=:sd)` is the standard deviation
  of the members' summed past outcomes, one reading of its definition.
- **Subset repetition summed over all orders**, retaliation and interval-censored
  hyperevents (Poda, Vinciotti & Wit 2025).
- **eventnet's four-cycle and neighbour statistics**, the node attribute on the
  third actor of a closure, and the `PRODUCT` aggregation function.
- **Time-varying and non-linear effects** of hyperedge statistics (Boschi,
  Lerner & Wit 2026), the **relational hyperevent outcome model** and the
  group-oriented factorisation into an author model and a citation model.
- **Goodness of fit** for hyperevent fits: `gof`, [`event_diagnostics`](@ref)
  and [`prediction_summary`](@ref) are defined on dyads. Simulate from the fitted
  model with [`simulate_hyperevents`](@ref) and compare summaries of your own.
- **The Efron tie correction** and a likelihood for the *timing* of hyperevents
  (the design models which hyperedge, given that an event of that size occurs).

## References

- Lerner, J., Tranmer, M., Mowbray, J. & Hâncean, M.-G. (2019). REM beyond
  dyads: relational hyperevent models for multi-actor interaction networks.
  arXiv:1912.07403.
- Lerner, J., Lomi, A., Mowbray, J., Rollings, N. & Tranmer, M. (2021). Dynamic
  network analysis of contact diaries. *Social Networks* 66, 224–236.
- Lerner, J. & Lomi, A. (2023). Relational hyperevent models for polyadic
  interaction networks. *Journal of the Royal Statistical Society A* 186(3),
  577–600.
- Lerner, J. & Hâncean, M.-G. (2023). Micro-level network dynamics of scientific
  collaboration and impact: relational hyperevent models for the analysis of
  coauthor networks. *Network Science* 11(1), 5–35.
- Lerner, J., Hâncean, M.-G. & Lomi, A. (2025). Relational hyperevent models for
  the coevolution of coauthoring and citation networks. *Journal of the Royal
  Statistical Society A* 188(2), 583–607.
- Lerner, J., Hâncean, M.-G. & Perc, M. (2025). Modeling temporal hypergraphs.
  *Journal of Complex Networks* 13(6), cnaf054.
- Perry, P. O. & Wolfe, P. J. (2013). Point process modelling for directed
  interaction networks. *Journal of the Royal Statistical Society B* 75(5),
  821–849.
- eventnet: RHEM effects reference guide,
  <https://github.com/juergenlerner/eventnet/wiki/RHEM-effects-(reference-guide)>.
