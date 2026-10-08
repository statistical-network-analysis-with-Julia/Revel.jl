# Concordance with Other Packages

The same configuration carries a different name and a different default
measurement in each package: relevent normalises, remstats counts, rem applies a
mandatory normalised half-life decay, goldfish dichotomises. "Equivalent"
effects therefore agree in sign but not in value, and a coefficient cannot be
carried from one package to another without knowing four things: the kernel
normalisation, the zero-history value, how the legs of a triad are combined, and
the scaling.

The tables below restate the cross-package concordance of the literature review
behind this package in terms of the Revel call that reproduces each package's
measurement. They are
[`effect_catalogue`](@ref) rendered — `effect_catalogue()` returns the same rows
as a `DataFrame`.

```@example concordance
using Revel

catalogue = effect_catalogue()
catalogue[catalogue.relevent .== "NODSnd", [:revel, :relevent]]
catalogue[catalogue.goldfish .== "commonReceiver", :revel]      # ["OSP(combine=:count)"]
```

## How far each column has been verified

| Column | Status |
|---|---|
| remstats | **Pinned numerically**, every statistic call of the column, against remstats 4.1.0 by a golden fixture (`test/fixtures/revel_remstats.toml`): 141 statistic arrays, directed and undirected, under full, window, interval and decay memory, with `prop` and `std` scaling, event weights, `consider_type = "separate"`, `a:b` products and `event()`. The fitting functions in the table (`remstimate`) are not pinned. |
| relevent | **Pinned numerically** against `relevent::rem.dyad` 1.2.1 by two golden fixtures: its design statistics (`test/fixtures/relevent_catalogue.toml`: every relevent name of the column with an unmarked cell, 24 design columns, exact) and its fitted ordinal and interval-timing models (`test/fixtures/relevent_rem_dyad.toml`, see [Coming from relevent](#Coming-from-relevent)) — except the three names marked **†** (`FrPSndSnd`, `FrRecSnd`, `OSPSnd`). Those are relevent's *documented* definitions; relevent 1.2.1's output differs from its documentation (a "fraction" above 1, shared partners counted twice), and Revel reproduces the documentation, not the output. |
| rem, goldfish, eventnet | From those packages' documentation and source, as read for the review. **Not executed.** |

## Six traps

1. **`FrPSndSnd` is a proportion.** Its documented definition is
   `Inertia(scaling=:prop)`, as is remstats `inertia(scaling = "prop")`, not
   the default `inertia()`. goldfish's default `inertia` is a 0/1 indicator.
2. **Shared-partner names follow the partner, not the pair.** goldfish
   `commonReceiver` is [`OSP`](@ref): the third actor is a common *receiver*,
   so the sender and the receiver of the candidate both *sent* to it ("outbound"
   from their side). `commonSender` is [`ISP`](@ref).
3. **`CovInt` is a sum, not an interaction.** It is [`SumEffect`](@ref): the
   sender's value plus the receiver's under one coefficient.
4. **Triads are aggregated at least five ways.** Sum of minima (relevent,
   remstats; eventnet's dyadic triangle statistics by its documentation), square
   root of summed products (rem), count of distinct third actors (goldfish,
   remstats `unique = TRUE`), sum of products (Perry & Wolfe; Vu et al. 2011),
   sum of harmonic means (Vu, Lomi, Mascia & Pallotti 2017): the `combine` and
   `root` keywords. (eventnet's hyperevent closure is a different statistic;
   see [`HyperClosure`](@ref).)
5. **There are at least three four-cycle statistics**: see
   [`FourCycleEffect`](@ref).
6. **The zero-history value differs, between the packages too.** For a
   proportion with an empty denominator (inertia, reciprocity) relevent and
   remstats both return `1/(n−1)`. For a degree share before the first event
   relevent returns `1/(n−1)` and remstats `1/n`, so the degree-share rows come
   in pairs. Revel's default is `0`; the `empty` keyword sets it.

## Deliberate differences from remstats

The golden fixture pins three places where Revel does not reproduce remstats
4.1.0, each by an assertion of the exact relationship:

- **Decay is evaluated at the event being explained.** Under
  `memory = "decay"` remstats weighs a past event by its age at the time of the
  *previous* event, not of the event being explained — contrary to its help
  page. Revel follows the documented definition, so a decayed statistic is
  smaller than remstats' by exactly `2^(−(t − t_prev)/halflife)`. Ratios
  (`scaling=:prop`), counts of distinct partners, ranks and recencies are
  unaffected.
- **`std` uses the sample standard deviation** in remstats: pass
  `Standardized(stat, n; corrected=true)`. The default divides by the number of
  dyads.
- **A share with nothing in memory.** remstats returns `1/n` for a degree share
  at the first time point and `0` at a later point whose memory window is empty;
  Revel returns `empty` whenever there is nothing to take a share of.

Two remstats conventions that *are* reproduced and worth knowing: undirected
`degreeMin`/`degreeMax` shares divide by the number of past events while
`totaldegreeDyad` divides by twice that number; and `remify(ordinal = TRUE)`
replaces the clock by the event index, which is `clock=:order`.

## Coming from relevent

`relevent::rem.dyad(edgelist, n, effects, ordinal = TRUE)` is
`fit_revel(events, statistics, n)`, and `ordinal = FALSE` is
`fit_revel(events, statistics, n; model=:timing, t_end=…)`. Both likelihoods
are exact over the full risk set, and both are pinned against relevent 1.2.1 by
a golden fixture (`test/fixtures/relevent_rem_dyad.toml`: CovSnd, CovRec and
four participation shifts on 100 simulated events, coefficients, standard
errors and log-likelihoods within `1e-6`, the reference optimizer's termination
slack). Three points of translation:

- **Maximum likelihood, not BPM.** R's default `fit.method = "BPM"` adds a prior;
  Revel maximises the likelihood. Compare with R's `fit.method = "MLE"`.
- **The timing model has an intercept.** Revel always fits `log λ₀`; relevent's
  temporal likelihood has none. Give R a constant `CovSnd` column and its
  coefficient is `log λ₀`.
- **The end of observation.** relevent reads the last edgelist row as the time
  observation stopped and adds the eventless tail to the likelihood; pass that
  time as `t_end`.

Each effect is a Revel call (the `relevent` column of the tables below):

| relevent | Revel | Statistic |
|---|---|---|
| `NIDSnd`, `NIDRec` | `IndegreeSender`, `IndegreeReceiver` with `scaling=:prop, empty=1/(n-1)` | indegree divided by the number of past events |
| `NODSnd`, `NODRec` | `OutdegreeSender`, `OutdegreeReceiver` with `scaling=:prop, empty=1/(n-1)` | outdegree divided by the number of past events |
| `NTDegSnd`, `NTDegRec` | `TotaldegreeSender`, `TotaldegreeReceiver` with `scaling=:prop, empty=1/(n-1)` | total degree divided by twice the number of past events |
| `RRecSnd`, `RSndSnd` | `RecencyRank(:receive)`, `RecencyRank(:send)` | inverse rank of the partner, by the order of the last event |
| `OTPSnd`, `ITPSnd`, `ISPSnd` | `OTP()`, `ITP()`, `ISP()` | sum over third actors of the smaller event count of the two legs |
| `FESnd`, `FERec`, `FEInt` | `SendEffect((1:n) .== k)`, `ReceiveEffect(…)`, `SumEffect(…)` | one indicator per actor `k` in `2:n` (actor 1 the reference) |
| `CovSnd`, `CovRec`, `CovInt` | `SendEffect(x)`, `ReceiveEffect(x)`, `SumEffect(x)` | static sender, receiver, sender-plus-receiver covariate |
| `CovEvent` | `TieEffect(X)` | static dyadic covariate `X[sender, receiver]` |
| `PSAB-BA`, … (13 shifts) | `PShift(:AB_BA)`, … ([`pshift_types`](@ref)) | Gibson's participation shifts |

```@example concordance
n = 4
h = build_history([Event(1, 2, 1.0), Event(2, 3, 2.0)])
compute(OTP(), h, 1, 3, 3.0)                                  # 1.0 — relevent OTPSnd
compute(OutdegreeSender(scaling=:prop, empty=1/(n-1)), h, 1, 2, 3.0)   # 0.5 — NODSnd
```

The degree shares are `1/(n−1)` before the first event, as in relevent; an
unseen recency partner has value zero; ranks use event order, not elapsed time;
the two-path and shared-partner effects use cumulative event counts with the
minimum along each path. The design fixture (`relevent_catalogue.toml`) compares
every one of these calls with relevent's own design statistics — read off
`rem.dyad.lambda` with one unit coefficient — on every candidate dyad before each
of 14 events, including an isolated actor and repeated dyads, at `1e-12`.

What relevent offers and Revel does not:

- **`FrPSndSnd`, `FrRecSnd` and `OSPSnd` as relevent computes them.** A probe of
  relevent 1.2.1 (`test/fixtures/r/relevent_deferred_probe.R`) finds output that
  departs from the documentation: after one `1 → 2` event `FrPSndSnd` gives zero
  on `1 → 2` rather than the documented fraction one, and `OSPSnd` gives
  asymmetric values on a repeated-dyad sequence. In the
  [source](https://github.com/cran/relevent/blob/7f7748ffa3b89bc8829f4e9159e76bbcb779611c/src/relevent.c),
  `lambda_R` passes the whole accumulated interaction list to `logrm_normint`,
  and `acl_tri_R`'s reverse shared-partner update addresses the same entries as
  its forward update. The catalogue rows marked † follow the documentation.
- Event-indexed covariate arrays (one statistic per covariate column instead), the
  Bayesian posterior mode, BSIR and prior controls, relevent's residuals and
  simulation, `conditioned.obs`, and events addressed to the group (the null
  actor). [`PShift`](@ref) evaluates the null-actor shifts (`receiver == 0`), but
  the risk sets are the directed dyads among `1:n`.

## The concordance

<!-- catalogue:begin -->

### Endogenous effects

| Revel | Measures | relevent | remstats | rem | goldfish | eventnet | Source |
|---|---|---|---|---|---|---|---|
| `Inertia()` | past s → r events |  | `inertia()` |  | `inertia(weighted = TRUE)` | DYAD_STATISTIC, dir OUT | Brandes, Lerner & Snijders 2009 |
| `Inertia(scaling=:prop, empty=1/(n-1))` | share of the sender's past sends that went to r | `FrPSndSnd †` | `inertia(scaling = "prop")` |  |  |  | Butts 2008; Kitts et al. 2017 |
| `Inertia(transform=:indicator)` | has s ever sent to r |  |  |  | `inertia` |  | Stadtfeld & Block 2017 |
| `Inertia(memory=HalfLife(h; normalized=true), weighted=true)` | half-life weighted s → r volume |  |  | `inertiaStat` |  |  | Brandes, Lerner & Snijders 2009 |
| `Reciprocation()` | past r → s events |  | `reciprocity()` |  | `recip(weighted = TRUE)` | DYAD_STATISTIC, dir IN | Brandes, Lerner & Snijders 2009 |
| `Reciprocation(scaling=:prop, empty=1/(n-1))` | share of the sender's past receipts that came from r | `FrRecSnd †` | `reciprocity(scaling = "prop")` |  |  |  | Butts 2008; Kitts et al. 2017 |
| `Reciprocation(scaling=:prop, denominator=:receiver_out)` | share of the receiver's past sends that went to s |  |  |  |  |  | Kitts et al. 2017 |
| `Reciprocation(memory=HalfLife(h; normalized=true), weighted=true)` | half-life weighted r → s volume |  |  | `reciprocityStat` |  |  | Brandes, Lerner & Snijders 2009 |
| `DyadActivity()` | past events between s and r, either way |  |  |  |  | DYAD_STATISTIC, dir SYM | eventnet |
| `OutdegreeSender()` | events sent by the sender |  | `outdegreeSender()` | `degreeStat (sender-outdegree)` | `outdeg(type = "ego")` | DEGREE_STATISTIC, OUT/SOURCE | Vu et al. 2011 |
| `OutdegreeSender(scaling=:prop, empty=1/(n-1))` | sender's share of all past sends (relevent: 1/(n−1) before any event) | `NODSnd` |  |  |  |  | relevent |
| `OutdegreeSender(scaling=:prop, empty=1/n)` | sender's share of all past sends (remstats: 1/n before any event) |  | `outdegreeSender(scaling = "prop")` |  |  |  | remstats |
| `IndegreeSender()` | events received by the sender |  | `indegreeSender()` | `degreeStat (sender-indegree)` | `indeg(type = "ego")` |  | Vu et al. 2011 |
| `IndegreeSender(scaling=:prop, empty=1/(n-1))` | sender's share of all past receipts (relevent: 1/(n−1) before any event) | `NIDSnd` |  |  |  |  | relevent |
| `IndegreeSender(scaling=:prop, empty=1/n)` | sender's share of all past receipts (remstats: 1/n before any event) |  | `indegreeSender(scaling = "prop")` |  |  |  | remstats |
| `IndegreeReceiver()` | events received by the receiver |  | `indegreeReceiver()` | `degreeStat (target-indegree)` | `indeg(type = "alter")` | DEGREE_STATISTIC, IN/TARGET | Vu et al. 2011 |
| `IndegreeReceiver(scaling=:prop, empty=1/(n-1))` | receiver's share of all past receipts (relevent: 1/(n−1) before any event) | `NIDRec` |  |  |  |  | relevent |
| `IndegreeReceiver(scaling=:prop, empty=1/n)` | receiver's share of all past receipts (remstats: 1/n before any event) |  | `indegreeReceiver(scaling = "prop")` |  |  |  | remstats |
| `OutdegreeReceiver()` | events sent by the receiver |  | `outdegreeReceiver()` | `degreeStat (target-outdegree)` | `outdeg(type = "alter")` |  | Vu et al. 2011 |
| `OutdegreeReceiver(scaling=:prop, empty=1/(n-1))` | receiver's share of all past sends (relevent: 1/(n−1) before any event) | `NODRec` |  |  |  |  | relevent |
| `OutdegreeReceiver(scaling=:prop, empty=1/n)` | receiver's share of all past sends (remstats: 1/n before any event) |  | `outdegreeReceiver(scaling = "prop")` |  |  |  | remstats |
| `TotaldegreeSender()` | events sent or received by the sender |  | `totaldegreeSender()` |  |  | degree statistic, SYM | remstats |
| `TotaldegreeSender(scaling=:prop, empty=1/(n-1))` | sender's share of all past volume (relevent: 1/(n−1) before any event) | `NTDegSnd` |  |  |  |  | relevent |
| `TotaldegreeSender(scaling=:prop, empty=1/n)` | sender's share of all past volume (remstats: 1/n before any event) |  | `totaldegreeSender(scaling = "prop")` |  |  |  | remstats |
| `TotaldegreeReceiver()` | events sent or received by the receiver |  | `totaldegreeReceiver()` |  |  | degree statistic, SYM | remstats |
| `TotaldegreeReceiver(scaling=:prop, empty=1/(n-1))` | receiver's share of all past volume (preferential attachment) (relevent: 1/(n−1) before any event) | `NTDegRec` |  |  |  |  | Butts 2008 |
| `TotaldegreeReceiver(scaling=:prop, empty=1/n)` | receiver's share of all past volume (preferential attachment) (remstats: 1/n before any event) |  | `totaldegreeReceiver(scaling = "prop")` |  |  |  | remstats |
| `OutdegreeSender(measure=:partners)` | distinct partners rather than events ("degree") |  |  |  | `outdeg(weighted = FALSE)` |  | Vu, Lomi, Mascia & Pallotti 2017 |
| `OutdegreeSender(measure=:intensity)` | events per distinct partner ("intensity") |  |  |  |  |  | Vu, Lomi, Mascia & Pallotti 2017 |
| `TotaldegreeDyad()` | sum of the two actors' total degrees |  | `totaldegreeDyad()` |  |  |  | remstats |
| `DegreeMin(symmetric=true)` | undirected: smaller of the two actors' event counts |  | `degreeMin()` |  |  |  | remstats |
| `DegreeMax(symmetric=true)` | undirected: larger of the two actors' event counts |  | `degreeMax()` |  |  |  | remstats |
| `DegreeDiff(symmetric=true)` | undirected: absolute difference of the two actors' event counts |  | `degreeDiff()` |  |  |  | remstats; Lerner, Hâncean & Perc 2025 |
| `Inertia(symmetric=true)` | undirected: past events of the pair |  | `inertia() (undirected)` |  | `inertia (choice_coordination)` |  | remstats |
| `DegreeAssortativity()` | sender out-degree × receiver in-degree |  | `outdegreeSender():indegreeReceiver()` |  |  |  | Lerner & Lomi 2020 |
| `DegreeAssortativity(measure=:partners)` | sender's × receiver's distinct partners ("assortativity by degree") |  |  |  |  |  | Vu, Lomi, Mascia & Pallotti 2017 |
| `OTP()` | s → k → r, Σ min of the legs (transitive closure) | `OTPSnd` | `otp()` |  |  | TRIANGLE_STATISTIC ("transitive_tie") | Butts 2008 |
| `OTP(combine=:count)` | number of distinct intermediaries |  | `otp(unique = TRUE)` |  | `trans` |  | Stadtfeld & Block 2017 |
| `OTP(combine=:product, root=true, memory=HalfLife(h; normalized=true))` | √ Σ products of half-life weights |  |  | `triadStat` |  |  | Brandes, Lerner & Snijders 2009 |
| `OTP(combine=:product)` | Σ products of the legs |  |  |  |  |  | Vu et al. 2011; Perry & Wolfe 2013 ("2-send") |
| `OTP(combine=:harmonic)` | Σ harmonic means of the legs |  |  |  |  |  | Vu, Lomi, Mascia & Pallotti 2017 |
| `ITP()` | r → k → s (cyclic closure) | `ITPSnd` | `itp()` |  |  | TRIANGLE_STATISTIC ("cyclical_tie") | Butts 2008 |
| `ITP(combine=:count)` | number of distinct intermediaries |  | `itp(unique = TRUE)` |  | `cycle` |  | Stadtfeld & Block 2017 |
| `OSP()` | s → k ← r (shared targets) | `OSPSnd †` | `osp()` |  |  | closure, both directions OUT | Butts 2008 |
| `OSP(combine=:count)` | number of shared targets |  | `osp(unique = TRUE)` |  | `commonReceiver` |  | Stadtfeld & Block 2017 |
| `ISP()` | s ← k → r (shared sources) | `ISPSnd` | `isp()` |  |  | closure, both directions IN | Butts 2008 |
| `ISP(combine=:count)` | number of shared sources |  | `isp(unique = TRUE)` |  | `commonSender` |  | Stadtfeld & Block 2017 |
| `SharedPartners()` | undirected shared partners |  | `sp()` |  | `trans (choice_coordination)` | closure, SYM | remstats |
| `TwoPathEffect(layer_a, layer_b)` | two-path across two event networks or types |  |  | `triadStat(eventtypevalues = )` | `mixedTrans, mixedCycle, mixedCommonSender, mixedCommonReceiver` | different attributes per leg | Stadtfeld & Block 2017 |
| `BalanceEffect(:friend_of_friend)` | signed two-path on undirected ± weights (also :friend_of_enemy, :enemy_of_friend, :enemy_of_enemy) |  |  | `triadStat(eventtypevalues = )` |  | "enemy of friend" closure | Brandes, Lerner & Snijders 2009 |
| `OTP(ordered=true)` | two-path whose first leg came first |  |  |  |  |  | Arena, Mulder & Leenders 2024 (approximation) |
| `FourCycleEffect()` | s → a ← b → r, Σ min of three weights |  |  |  |  | FOUR_CYCLE_STATISTIC | Lerner & Lomi 2020 |
| `FourCycleEffect(combine=:product, root=true, memory=HalfLife(h; normalized=true))` | cube root of Σ products |  |  | `fourCycleStat` |  |  | Brandenberger (rem) |
| `FourCycleEffect(combine=:count)` | number of three-paths |  |  |  | `four` |  | Stadtfeld & Block 2017; Haunss & Hollway 2023 |
| `RecencyRank(:receive)` | 1/rank of r among those who most recently sent to s | `RRecSnd` | `rrankReceive()` |  |  |  | Butts 2008 |
| `RecencyRank(:send)` | 1/rank of r among those s most recently sent to | `RSndSnd` | `rrankSend()` |  |  |  | DuBois, Butts, McFarland & Smyth 2013 |
| `TimeSince(:dyad)` | 1/(time since the last s → r event + 1) |  | `recencyContinue()` |  |  | last-event-time attribute | remstats |
| `TimeSince(:send_sender)` | 1/(time since s last sent + 1) |  | `recencySendSender()` |  |  |  | remstats |
| `TimeSince(:send_receiver)` | 1/(time since r last sent + 1) |  | `recencySendReceiver()` |  |  |  | remstats |
| `TimeSince(:receive_sender)` | 1/(time since s last received + 1) |  | `recencyReceiveSender()` |  |  |  | remstats |
| `TimeSince(:receive_receiver)` | 1/(time since r last received + 1) |  | `recencyReceiveReceiver()` |  |  |  | remstats |
| `TimeSince(:dyad; transform=identity)` | raw gap time |  |  |  |  |  | Zappa & Vu 2021 |
| `TimeSince(:pair)` | undirected: 1/(time since the pair last interacted + 1) |  | `recencyContinue() (undirected)` |  |  |  | remstats |
| `PShift(:AB_BA)` | participation shift (also :AB_BY, :AB_XA, :AB_XB, :AB_XY, :AB_AY and the seven group-addressed shifts) | `PSAB-BA` | `psABBA()` |  |  |  | Butts 2008 |
| `PShiftABAB()` | the same dyad repeats immediately |  | `psABAB()` |  |  |  | remstats |
| `UndirectedPShift(:AB_AY)` | undirected: one actor of the last pair carries on with someone new (also :AB_AB) |  | `psABAY() (undirected)` |  |  |  | remstats |
| `NodeTransitivity()` | transitive structures in which the actor is the source |  |  |  | `nodeTrans` |  | Stadtfeld & Block 2017 |
| `StructuralSimilarity()` | Jaccard / cosine similarity of the two actors' partner profiles |  |  |  |  | JACCARD_SIM_STATISTIC, COSINE_SIM_STATISTIC | eventnet |

### Exogenous effects

| Revel | Measures | relevent | remstats | rem | goldfish | eventnet | Source |
|---|---|---|---|---|---|---|---|
| `SendEffect(x)` | sender's value | `CovSnd` | `send()` |  | `ego()` | NODE_STATISTIC, SOURCE | Butts 2008 |
| `ReceiveEffect(x)` | receiver's value | `CovRec` | `receive()` |  | `alter()` | NODE_STATISTIC, TARGET | Butts 2008 |
| `SumEffect(x)` | sender's plus receiver's value | `CovInt` |  |  |  |  | relevent |
| `SendEffect((1:n) .== k)` | fixed effect of actor k as sender (one per actor in 2:n; actor 1 the reference) | `FESnd` |  |  |  |  | relevent |
| `ReceiveEffect((1:n) .== k)` | fixed effect of actor k as receiver | `FERec` |  |  |  |  | relevent |
| `SumEffect((1:n) .== k)` | fixed effect of actor k in either role | `FEInt` |  |  |  |  | relevent |
| `AverageEffect(x)` | mean of the two values |  | `average()` |  |  |  | remstats |
| `MatchEffect(x)` | same category |  | `same()` |  | `same()` | CATDIFF aggregation | Perry & Wolfe 2013 |
| `DiffEffect(x)` | absolute difference |  | `difference()` |  | `diff()` | ABSDIFF aggregation | remstats; goldfish |
| `SimEffect(x)` | negative absolute difference |  |  |  | `sim()` |  | goldfish |
| `MinimumEffect(x)` | smaller of the two values |  | `minimum()` |  |  | MIN aggregation | remstats |
| `MaximumEffect(x)` | larger of the two values |  | `maximum()` |  |  | MAX aggregation | remstats |
| `ProductEffect(x, y)` | sender's x × receiver's y |  | `send("x"):receive("y")` |  | `egoAlterInt()` | PRODUCT aggregation | Perry & Wolfe 2013 |
| `TieEffect(matrix)` | dyadic covariate | `CovEvent` | `tie()` |  | `tie()` | DYAD_STATISTIC on an exogenous attribute | Butts 2008 |
| `GlobalEffect(f)` | covariate of time alone (a moderator) |  | `event()` |  | `global attribute` | NETWORK_STATISTIC | Lembo, Juozaitienė, Vinciotti & Wit 2026 |

### Interactions

| Revel | Measures | relevent | remstats | rem | goldfish | eventnet | Source |
|---|---|---|---|---|---|---|---|
| `Interaction(a, b)` | product term |  | `a:b` |  |  |  | remstats |
| `Interaction(GlobalEffect(times, z), stat)` | an effect moderated by an attribute of the event being explained |  | `event("z"):stat()` |  |  |  | remstats |
| `EventLayer(types=…); split_by_type` | statistic split by the type of the past events |  | `consider_type = "separate"` | `eventtypevar` | `one network per type` | one attribute per event type | Brandes, Lerner & Snijders 2009 |
| `EventLayer(keep=…)` | statistic on an attribute-filtered history |  |  | `eventfiltervar` |  |  | Brandenberger (rem) |
| `MatchedDegree(x)` | degree over third actors who match the other endpoint on x |  |  | `degreeStat with filters` |  | *_NEIGHBOR_STAT | Malang, Brandenberger & Leifeld 2019 |
| `OTP(third=matching_third(x))` | closure through third actors who match the sender on x |  |  | `triadStat(eventfilterAI = )` |  | closure with a node attribute | rem; eventnet |
| `TertiusEffect(x)` | aggregate of x over the receiver's in-neighbours |  |  |  | `tertius` | *_NEIGHBOR_STAT | Stadtfeld & Block 2017; Haunss & Hollway 2023 |
| `TertiusEffect(x; aggregate=:entropy)` | Shannon entropy of the categories of the receiver's in-neighbours |  |  |  |  |  | Haunss & Hollway 2023 ("tertius party diversity") |
| `TertiusEffect(x; difference=true)` | abs(sender's x − that aggregate): homophily at distance two |  |  |  | `tertiusDiff` |  | Haunss & Hollway 2023 |
| `fit_stratified(…; by)` | separate fits per stratum of events |  |  |  |  |  | Vu, Lomi, Mascia & Pallotti 2017 |
| `fit_moving_window(…; width)` | time-varying coefficients |  |  |  |  |  | Mulder & Leenders 2019 |
| `Interaction(GlobalEffect(f), stat)` | effect moderated by a period or a time of day |  |  |  |  | network statistic interacted downstream | Lembo et al. 2026 |

### Hyperevents

| Revel | Measures | relevent | remstats | rem | goldfish | eventnet | Source |
|---|---|---|---|---|---|---|---|
| `HyperedgeSize()` | number of participants (a moderator only) |  |  |  |  | UHE_SIZE_STAT, DHE_SIZE_STAT | Lerner, Tranmer, Mowbray & Hâncean 2019 |
| `ExactRepetition()` | past events on exactly this hyperedge |  |  |  |  | UHE_REPETITION_STAT, DHE_REPETITION_STAT | Lerner, Tranmer, Mowbray & Hâncean 2019 |
| `SubsetRepetition(p)` | past events containing each p-subset of the participants, averaged |  |  |  |  | UHE_SUB_REPETITION_STAT | Lerner, Tranmer, Mowbray & Hâncean 2019 |
| `SharedPriorEvents(p)` | past events sharing at least p participants |  |  |  |  |  | Lerner, Tranmer, Mowbray & Hâncean 2019 |
| `PriorSuccess(p)` | outcome-weighted over unweighted subset degree |  |  |  |  |  | Lerner & Hâncean 2023 |
| `DirectedSubsetRepetition(p, q)` | past events containing p of the senders and q of the receivers |  |  |  |  | DHE_SUB_REPETITION_STAT | Lerner, Tranmer, Mowbray & Hâncean 2019 |
| `UnorderedRepetition()` | past events among the same actors in any roles ("reply to all") |  |  |  |  | DHE_REPETITION_STAT, dir SYM | Lerner & Lomi 2023 |
| `ReceiverSetRepetition(p)` | partial receiver-set repetition of order p |  |  |  |  | DHE_SUB_REPETITION_STAT, endpoint TARGET | Lerner & Lomi 2023 |
| `SenderReceiverSetRepetition(p)` | sender-specific partial receiver-set repetition |  |  |  |  | DHE_SUB_REPETITION_STAT | Lerner & Lomi 2023 |
| `HyperSenderActivity()` | past events sent by the senders |  |  |  |  | DHE_SUB_REPETITION_STAT (1, 0) | Lerner, Tranmer, Mowbray & Hâncean 2019 |
| `HyperReceiverPopularity()` | past events received by the receivers |  |  |  |  | DHE_SUB_REPETITION_STAT (0, 1) | Lerner & Lomi 2023 |
| `HyperReciprocation()` | past events from the receivers to the sender |  |  |  |  | DHE_SUB_REPETITION_STAT, dir IN | Lerner & Lomi 2023 |
| `OutInPopularity()` | past events sent by the receivers |  |  |  |  |  | Lerner & Lomi 2023 |
| `InteractionAmongReceivers(p)` | past interaction among the receivers |  |  |  |  |  | Lerner & Lomi 2023 |
| `HyperClosure()` | two-path closure on the dyadic projection, over the hyperedge's pairs |  |  |  |  | UHE_CLOSURE_STAT, DHE_CLOSURE_STAT | Lerner, Lomi, Mowbray, Rollings & Tranmer 2021 |
| `HyperCovariate(x)` | summary of an actor covariate over the hyperedge |  |  |  |  | UHE_NODE_STAT, DHE_NODE_STAT | Lerner & Lomi 2023 |

### Memory and event weights

| Revel | Measures | relevent | remstats | rem | goldfish | eventnet | Source |
|---|---|---|---|---|---|---|---|
| `memory=HalfLife(h)` | exponential decay |  | `memory = "decay"` | `halflife (normalized=true)` |  | attribute half-life | Brandes, Lerner & Snijders 2009 |
| `memory=Window(w)` | sliding window |  | `memory = "window"` |  | `window` |  | de Nooy 2011; Quintane et al. 2013 |
| `memory=IntervalMemory(a, b)` | one interval of the past |  | `memory = "interval"` |  |  |  | Perry & Wolfe 2013 |
| `memory=PowerLaw(α)` | power-law decay |  |  |  |  |  | Vu, Lomi, Mascia & Pallotti 2017 |
| `memory=LinearDecay(span)` | linear decay |  |  |  |  |  | Arena, Mulder & Leenders 2023 |
| `clock=:order` | age measured in events |  |  |  |  |  | Malang, Brandenberger & Leifeld 2019 |
| `weighted=true` | event weights instead of counts |  | `weight column` | `weight` | `weighted = TRUE` | event weight response | Brandes, Lerner & Snijders 2009 |

### Scaling

| Revel | Measures | relevent | remstats | rem | goldfish | eventnet | Source |
|---|---|---|---|---|---|---|---|
| `Standardized(stat, n; corrected=true)` | z-score across the risk set (with riskset=:full) |  | `scaling = "std"` |  |  |  | remstats |
| `transform=:log1p` | log(1 + x) |  |  |  | `transformFun` | LOG1P function | Fritz, Rastelli, Fop & Caimo 2025 |

<!-- catalogue:end -->

## Not implemented

These appear in the literature and are refused or absent here. Each is listed
in the package README and CHANGELOG as well.

| Feature | Where it exists | Status in Revel |
|---|---|---|
| Random effects, frailties, random slopes, cross-level interactions | remstimate `remfrailty`, goldfish.latent, remx | not implemented; take [`event_design`](@ref) to a mixed-model package |
| Smooth (non-linear) and spline-based time-varying effects | mgcv-based REMs, amorem | not implemented; [`fit_moving_window`](@ref) and [`interval_partition`](@ref) are the piecewise alternatives |
| Dyad × event-type risk set (`consider_type = "interact"`, `FEtype`) | remify / remstats | not implemented; [`fit_stratified`](@ref) by outcome type |
| Events with duration, active-state statistics | remstats `active*`, durem, redeem | not implemented (an `Event` is an instant) |
| Sender-rate step of actor-oriented models; DyNAM-i group joining/leaving | goldfish | not implemented; the receiver-choice step is [`fit_receiver_choice`](@ref) |
| Weibull/Gompertz baselines, integrated time-varying hazards | — | not implemented; the timing model is exponential with statistics constant between events |
| Group-addressed participation shifts as a risk set | relevent | the statistics exist ([`PShift`](@ref)); events "to the group" are not in Revel's dyadic risk sets |
| Pairwise time-ordered transitivity (Arena, Mulder & Leenders 2024) | bremory | approximated by `OTP(ordered=true)` |
| Max-based turn-taking and turn-continuing (Juozaitienė & Wit 2024) | amorem | not implemented |
| informR sequence statistics (S-forms) | informR | not implemented |
| Bayesian estimation, penalisation, mixtures | remstimate, relevent | not implemented |
| Main effects of global covariates from time-shifted controls (Lembo, Juozaitienė, Vinciotti & Wit 2026) | — | not implemented; a [`GlobalEffect`](@ref) outside an [`Interaction`](@ref) is refused, because the ordinary partial likelihood does not identify it |
| A `missing=` policy for covariates | — | not implemented; a `missing` covariate value is refused |
| Internal times and decile statistics (Amati, Lomi & Snijders 2024); auxiliary-statistic score processes and their Cauchy combination (Boschi & Wit 2026); simulation of a held-out segment (Brandenberger 2019) | — | not implemented; [`closing_times`](@ref), [`score_process_test`](@ref), [`prediction_summary`](@ref) and `gof` are the related tools, with the differences stated in their docstrings |
| Two-mode and generalised hyperevents, geometric weighting, closure of order `(p, q, l)`, switch reciprocation, eventnet's four-cycle and neighbour statistics, time-varying hyperedge effects, the outcome model, Efron ties and a timing likelihood for hyperevents, goodness of fit for hyperevent fits | eventnet; the hyperevent papers | not implemented; see [Relational hyperevents](hyperevents.md) |
