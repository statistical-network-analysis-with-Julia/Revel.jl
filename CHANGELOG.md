# Changelog

All notable changes to Revel.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First version. Revel.jl builds on Relevent.jl and REM.jl and implements the
effect catalogue of a scoping review of the relational event model literature
(210 works, 2008–2026): endogenous effects, covariates, effect × covariate
interactions, goodness-of-fit diagnostics and relational hyperevents.

### Added

- **Memory kernels** — `FullMemory`, `HalfLife` (plain or `normalized`),
  `Window`, `IntervalMemory`, `PowerLaw`, `LinearDecay`, `KernelMemory`;
  `kernel_weight`, `interval_partition`.
- **`EventLayer`** — the measurement every endogenous effect reads: memory
  kernel, past-event types, attribute filter, event weights, clock (`:time` or
  `:order`), symmetric (undirected) recording. Accumulating kernels absorb each
  event once; the others are re-read from the retained events up to the
  kernel's support.
- **Endogenous effects** — the parametric `DyadEffect`, `DegreeEffect`,
  `DyadDegreeEffect`, `TwoPathEffect`, `FourCycleEffect` and their named forms
  `Inertia`, `Reciprocation`, `DyadActivity`, `OutdegreeSender`,
  `IndegreeSender`, `TotaldegreeSender`, `OutdegreeReceiver`,
  `IndegreeReceiver`, `TotaldegreeReceiver`, `TotaldegreeDyad`, `DegreeMin`,
  `DegreeMax`, `DegreeDiff`, `DegreeAssortativity`, `OTP`, `ITP`, `OSP`, `ISP`,
  `SharedPartners`, `BalanceEffect`; `RecencyRank`, `TimeSince`, `PShiftABAB`,
  `UndirectedPShift`, `NodeTransitivity`, `StructuralSimilarity`.
- **Covariates** — `Covariate` (static, categorical, time-varying),
  `CovariateEffect` and `SendEffect`, `ReceiveEffect`, `MatchEffect`,
  `DiffEffect`, `SimEffect`, `AverageEffect`, `MinimumEffect`,
  `MaximumEffect`, `SumEffect`, `ProductEffect`; `TieEffect`; `GlobalEffect`.
- **Interactions** — `Interaction` (product terms over any statistics of the
  ecosystem), `Transformed`, `Standardized`, `split_by_type`, `matching_third`,
  `MatchedDegree`, `TertiusEffect`.
- **Designs and fitting** — `each_risk_set`, `RiskSetView`, `event_design`,
  `two_mode_dyads`; `fit_revel` (alias `revel`) and `RevelFit`;
  `fit_receiver_choice`, `fit_stratified`, `fit_moving_window`,
  `compare_coefficients`, `profile_memory`; `simulate_events`.
- **Goodness of fit** — `event_diagnostics`, `prediction_summary`,
  `score_process_test`, `score_test`, `gof` (a method of the shared
  `Networks.gof`), `mechanism_shares`, `closing_times`,
  `statistic_collinearity`.
- **Relational hyperevents** — `HyperEvent`, `HyperHistory`, subset repetition,
  closure and attribute statistics, `hyper_design`, `fit_rhem` (alias `rhem`),
  `simulate_hyperevents`.
- **`effect_catalogue`** — the concordance with relevent, remstats, rem,
  goldfish and eventnet.
- Golden fixture `test/fixtures/revel_remstats.toml` (remstats 4.1.0, remify
  4.1.0; 141 statistic arrays), regenerated with
  `Rscript test/fixtures/r/revel_remstats.R > test/fixtures/revel_remstats.toml`.

### Fixed (adversarial panel review, 2026-09-30)

- `score_process_test`: the `p_kolmogorov` column and the `statistic` standardised
  the score process by √(I⁻¹)ₖₖ instead of 1/√Iₖₖ, which inflated them whenever
  the statistics were correlated: true models were rejected 18–58 % of the time
  at the 5 % level. The resampling `p_value` was unaffected. The docstring and
  the guide no longer read the per-effect rows as an attribution.
- `BalanceEffect(:friend_of_enemy)` and `:enemy_of_friend` were swapped relative
  to Brandes, Lerner & Snijders (2009), friendOfEnemy(a,b) = √Σ ω⁻(a,i)·ω⁺(i,b).
- `compute` counted events of the history later than the evaluation time (with
  a decaying memory, with weights above 1). Statistics now read only the events
  at or before `time`; a history out of time order and non-finite times are
  refused. The fitters were unaffected, since they pass pre-event histories.
- `event_diagnostics` and `prediction_summary` under `ties=:efron` applied the
  tie weight to the observed case's own probability, and used the unweighted
  risk-set size in the null deviance; the deviance residuals now add up to
  `-2loglikelihood(fit)`.
- `gof` on undirected fits compared simulated pairs written `(min, max)` with the
  observed orientation, and used directed auxiliary statistics: a correctly
  specified model was rejected on every statistic. Undirected fits now use
  undirected auxiliaries on orientation-free events.
- Undirected designs (`directed=false`) now write every event as `(min, max)`, so
  no statistic depends on how a pair happened to be stored.
- `fit_stratified(...; cases=)`, `fit_moving_window(...; cases=)` and
  `fit_receiver_choice(...; riskset=)` silently let the user's keyword override
  the wrapper's own; they are now refused.
- `fit_stratified` with a stratifier that depends on the acting dyad (a
  sender's group) fitted each stratum against every dyad, biasing covariate
  effects (a sender effect of 0.5 came out at 2.05). Each event's risk set is now
  restricted to the dyads that would have produced an event of its stratum.
- `HalfLife` memory returned `NaN` on a far-negative clock (`0 · exp(λ·|t|)`).
- `Covariate` treated `missing` as a category, so two missing values matched;
  missing values are now refused.
- Hyperevents: the default design for directed hyperevents is now the
  sender-stratified design of Lerner & Lomi (2023) (`sampler=:auto`); the uniform
  design was biased when senders differ in activity. `actors` may be a function
  of the event, for actors who join or leave during the sequence.

### Changed and fixed (second review round, 2026-09-30)

Scale and robustness:

- **Layer storage is sparse in the dyads.** Everything a layer keeps per dyad
  lives in vectors indexed by the dyads that have a history, found through a
  dense index (with dense mirrors of the weights, for speed) below 2,048 actors
  and a hash map above. Six statistics over 2,000 actors, evaluated by hand, take
  an eighth of the memory they took (and a sixth of that inside a fit, where
  they share one index); a single event to actor 3,000 no longer allocates
  340 MiB, and actor IDs of 100,000 cost a few MiB, not 400 GB.
- **Sampled controls cost the rows drawn.** `event_design` and
  `fit_revel(…; n_controls)` draw the controls first and evaluate the
  statistics on them only.
- **Private copies of the statistics.** Every entry point (`each_risk_set`,
  `event_design`, `fit_revel`, `simulate_events`, the diagnostics) evaluates
  copies of the statistics it is given, which merge layers with identical
  keywords. One specification can now be fitted from several tasks at once (it
  crashed before), a `RevelFit` no longer keeps the caches of its history
  alive, and `[Inertia(), Reciprocation(), OTP()]` index the history once.
- A history rewritten in place is detected: the cache signature now covers the
  first and the last absorbed event, with their weight and type.
- The default `cache_bytes` of the Relevent route is a quarter of the free
  memory (256 MiB – 8 GiB), so moderate models no longer recompute every
  statistic on every optimizer pass.
- Hyperevent statistics on full memory read their lists by bisection, so a
  design costs `O(E log E)` instead of `O(E²/n)`.

Refusals instead of silent results:

- `fit_revel` refuses a REM.jl statistic (it has no history interface) and a
  `GlobalEffect` outside an `Interaction` (its main effect is not identified);
  both used to fail late or fit zeros.
- Self-loops, non-finite event times, a sender outside the actors of
  `riskset=:sender`, a function risk set listing a dyad twice, a `keep=`
  filter on REM's untyped event log, a stratum key of `missing`, and a
  `memory=` on `RecencyRank`/`TimeSince` (ignored before, yet named) are
  refused with an `ArgumentError`.
- `Standardized` must standardise over the model's risk set: it is refused for
  a different `n_actors`, `directed` or dyad set, and for risk sets that change
  from event to event; `Standardized(stat, dyads)` covers two-mode and other
  fixed dyad sets. Its docstring says that it changes the model (an effect of
  `θ/σ_t` per raw unit), not only the scale.
- The diagnostics refuse a fit that did not converge; `event_diagnostics`,
  `prediction_summary` and the score tests refuse a `HyperFit` with an
  `ArgumentError` (a `MethodError` before).

Statistics:

- `score_process_test`: the `GLOBAL` row is the maximum of the standardised
  suprema with its p-value resampled as a maximum (it was a Bonferroni bound
  next to a statistic it did not test); p-values are never exactly 0.
- `score_test`: a candidate close to, but not inside, the span of the fitted
  effects is tested instead of returning `NaN` (the threshold now follows the
  rounding error of the residual variance); new column `residual_share`.
- `prediction_summary`: recall credits dyads tied with the observed one as a
  random tie-break would (the expected recall), and the mean reciprocal rank is
  the expected `1/rank`; `event_diagnostics` gains `n_above`, `n_tied` and
  `reciprocal_rank`. The default `ks` are documented as ranks, not percentages.
- `gof` reports an overall Monte-Carlo p-value from the Mahalanobis distance of
  all auxiliary statistics (each point measured against the others, so the test
  is exact under the model); simulates tied timestamps with the history frozen
  under `ties=:breslow`/`:efron`, as the fit did; simulates a timing fit from its
  `t0` and adds waiting-time auxiliaries; censors a closing-time quartile at the
  sequence length when nothing closes (it was 0, "instantaneous"); refuses
  `riskset=:receiver` with the right reason. Its docstring discloses that it is
  a plug-in check with conservative, pointwise p-values.
- `statistic_collinearity`: an exact duplicate no longer makes every VIF
  infinite, and `statistic_collinearity(fit)` measures collinearity at the
  estimate (`I_jj (I⁻¹)_jj`).
- `profile_memory`: every grid value is fitted on the same draw of controls;
  `best` ignores unconverged and non-finite fits; a failing value is recorded,
  not fatal; new `in_ci` column (the profile-likelihood confidence set); `aic`
  and `bic` count the memory parameter.
- `compare_coefficients(fits; reference=)` adds Wald tests of the differences
  between strata (or non-overlapping windows).
- `fit_moving_window` computes window starts as `t_first + k·step` (no drift).
- `TertiusEffect(aggregate=:sd)` uses Welford's update (it returned 0 for large
  values with a small spread).

New measures:

- `measure=:intensity` (events per partner) for the degree effects, and
  `DegreeAssortativity(measure=:partners)`: the "intensity" and "assortativity
  by degree" of Vu, Lomi, Mascia & Pallotti (2017), which `:events` did not
  reproduce.
- `combine=:harmonic` for the two-path effects (Vu et al. 2017, eq. 12).
- `TertiusEffect(x; aggregate=:entropy)` for a categorical covariate, the
  "tertius party diversity" of Haunss & Hollway (2023).
- Hyperevents: `SubsetRepetition(…; aggregate=:samplesd)` (with `p = 1` and
  `weighted=true`, the prior success disparity of Lerner & Hâncean 2023) and
  `HyperCovariate(x; aggregate=:homogeneity)` (the covariate homogeneity of
  Lerner et al. 2021, which the concordance used to map onto `:absdiff`); the
  worked examples of Lerner & Lomi (2023, Figs 2–7) are pinned as a test.
- Layers on the event clock (`clock=:order`) and `TimeSince(…; clock=:order)` are
  admissible in the timing model.
- `Covariate(…; categorical=true)` treats numbers as category labels.
- `PowerLaw(…; support=)` truncates the kernel, which bounds the otherwise
  quadratic cost of an infinite-support kernel over a long sequence.
- `simulate_events(…; ties, t0)`; `RiskSetView.risk_set_size`; `RevelFit.t0`,
  `RevelFit.t_end`; a two-argument `show` for `RevelFit` and `HyperFit` (the
  table is the `text/plain` display).

Renamed:

- `Interval` → `IntervalMemory`, which no longer clashes with
  `IntervalSets.Interval`.

Documentation and attributions:

- The concordance: relevent's `FrPSndSnd`, `FrRecSnd` and `OSPSnd` are marked
  as relevent's documented definitions, which relevent 1.2.1's output does not
  follow; the degree shares are split into a relevent row (`1/(n−1)` before any
  event) and a remstats row (`1/n`); the remstats cell `consider_type =
  "interact"` of a two-layer two-path, which the package refuses, is removed; the
  unverified `remstimate::remwindow()` is removed; `DyadActivity`, `fit_stratified`
  and `HyperSenderActivity` are credited to their actual sources; the "pinned"
  claims are scoped to what the fixture covers.
- Corrected attributions: Lembo et al. (2026) do identify global main effects
  (with time-shifted controls); Stadtfeld & Block's sign reversal concerns
  "outdegree × friendship"; Vu et al.'s intensity is events per partner and
  their closure a harmonic mean; top-k recall is Meijerink-Bosman et al.
  (2023)'s; `closing_times` and `gof` follow Amati et al. (2024) in spirit, not
  in their definitions; Butts's "persistence" is the proportion; `MatchedDegree`
  is partisan *influence*; time-ordered transitivity is Arena et al. (2024,
  eq. 12), which `OTP(ordered=true)` matches only for single-event legs;
  Lerner et al.'s covariate homogeneity is not `:absdiff`.
- Wrong example values corrected (`closing_times`, `fit_revel`, `coef`), and
  every value a docstring or a guide page states in a comment is now checked by
  the test suite; the guides run in the tests.
- Framing: four structural configurations, five parametric statistics; six
  implemented moderation constructions; citations use the bibliography's
  (issue) years throughout; the unpublished review is no longer cited as an
  authority in docstrings.

Tests: a seed-tuned `gof` assertion is replaced by a calibrated one; size tests
for the resampling score-process test and the score test; a brute-force
log partial likelihood; two-mode recovery; parallel fits; sparse storage;
sampled-design cost.

### Design decisions

- Revel.jl hosts no optimizer and no likelihood kernel. `fit_revel` routes the
  ordinal model on the full directed risk set to `Relevent.fit_obpm`, every
  other ordinal model to `REM.fit_rem` on the design frame, and the timing model
  to `Relevent.fit_timing`.
- Every statistic implements both compute interfaces of the ecosystem
  (`InteractionHistory` and `REM.EventNetworkState`) through one internal
  method, and extends the shared `compute`/`name` generics by name.
- The zero-history value of a proportion is the `empty` keyword, not a hidden
  default; the way the legs of a triad are combined is the `combine` keyword.
- On a symmetric layer an actor's out-, in- and total degree coincide (the
  events it took part in), and a degree share is taken of the number of events.

### Known limitations

- **Not implemented** (refused or absent; see the README): random effects and
  random slopes; smooth non-linear and time-varying effects; a dyad × event-type
  risk set; events with duration; the sender-rate step of actor-oriented models
  and DyNAM-i; Weibull/Gompertz baselines and integrated time-varying hazards;
  main effects of global covariates from time-shifted controls; group-addressed
  participation shifts as a risk set; pairwise time-ordered transitivity;
  max-based turn-taking statistics; informR sequence statistics; Bayesian
  estimation, penalisation and mixtures; a `missing=` policy for covariates;
  Amati et al.'s internal times, Boschi & Wit's auxiliary-statistic processes,
  Brandenberger's segment prediction and a refitting `gof`; for hyperevents,
  two-mode and generalised hyperevents, geometric weighting, closure of order
  `(p, q, l)` and switch reciprocation, eventnet's four-cycle and neighbour
  statistics, time-varying
  hyperedge effects, the outcome model, Efron ties, a timing likelihood, and
  goodness of fit.
- **Deliberate differences from remstats 4.1.0**, pinned by the golden fixture:
  the decay kernel is evaluated at the time of the event being explained
  (remstats: the previous event); `Standardized` defaults to the population
  standard deviation (`corrected=true` gives remstats' `std`); a degree share
  with nothing in memory is `empty`.
- The rem, goldfish and eventnet columns of `effect_catalogue` follow those
  packages' documentation and are not checked numerically.
- `compute` called by hand on one statistic from several tasks at once is not
  safe (the entry points work on private copies and are).
- A memory kernel without finite support costs `O(E²)` over a sequence.
- `gof` is a plug-in check; its per-statistic p-values are conservative and
  pointwise.
- The sandwich standard errors treat each event as a cluster.
- `[compat]` cannot express the dependency on Networks.jl's thirty-halving
  `newton_fit` (commit `03aaa03`; every sibling is at 0.2.0, unreleased).
