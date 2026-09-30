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
  `Window`, `Interval`, `PowerLaw`, `LinearDecay`, `KernelMemory`;
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
  4.1.0), regenerated with
  `Rscript test/fixtures/r/revel_remstats.R > test/fixtures/revel_remstats.toml`.

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
  pairwise time-ordered transitivity; max-based turn-taking statistics; informR
  sequence statistics; Bayesian estimation, penalisation and mixtures; two-mode
  and generalised hyperevents; goodness of fit for hyperevent fits.
- **Deliberate differences from remstats 4.1.0**, pinned by the golden fixture:
  the decay kernel is evaluated at the time of the event being explained
  (remstats: the previous event); `Standardized` defaults to the population
  standard deviation (`corrected=true` gives remstats' `std`); a degree share
  with nothing in memory is `empty`.
- The rem, goldfish and eventnet columns of `effect_catalogue` follow those
  packages' documentation and are not checked numerically.
- With a Networks.jl checkout from before its `newton_fit` step-halving limit
  was raised from 10 to 30 (2026-09-30), a rare indicator statistic with a large
  coefficient next to a near-duplicate term can stop a fit after two iterations
  with `converged == false`.
- Layers store dense `n × n` matrices and cache the history they were last
  evaluated on; a statistic must not be shared between concurrently fitting
  tasks.
