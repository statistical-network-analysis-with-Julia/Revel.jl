# Changelog

All notable changes to Revel.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

The first version, not yet released. Revel.jl fits the exact full-risk-set
likelihoods of relevent's `rem.dyad` and implements the effect catalogue of a
scoping review of the relational event model literature (210 works,
2008–2026): endogenous effects, covariates, effect × covariate interactions,
goodness-of-fit diagnostics and relational hyperevents. It depends on
NetworkCore.jl and REM.jl.

### Added

- **Memory kernels:** `FullMemory`, `HalfLife` (plain or `normalized`),
  `Window`, `IntervalMemory`, `PowerLaw` (with an optional `support=`),
  `LinearDecay`, `KernelMemory`; `kernel_weight`, `interval_partition`.
- **`EventLayer`**, the measurement every endogenous effect reads: memory
  kernel, past-event types, attribute filter, event weights, clock (`:time` or
  `:order`) and symmetric (undirected) recording. Per-dyad storage is sparse,
  so actor IDs in the hundreds of thousands cost a few MiB.
- **Endogenous effects:** the parametric `DyadEffect`, `DegreeEffect`,
  `DyadDegreeEffect`, `TwoPathEffect` and `FourCycleEffect`, and their named
  forms `Inertia`, `Reciprocation`, `DyadActivity`, the sender and receiver
  degree effects, `TotaldegreeDyad`, `DegreeMin`, `DegreeMax`, `DegreeDiff`,
  `DegreeAssortativity`, `OTP`, `ITP`, `OSP`, `ISP`, `SharedPartners` and
  `BalanceEffect`; `RecencyRank`, `TimeSince`, `PShiftABAB`, `UndirectedPShift`,
  `NodeTransitivity`, `StructuralSimilarity`. Degree effects take
  `measure=:intensity` and two-path effects `combine=:harmonic` (Vu, Lomi,
  Mascia & Pallotti 2017).
- **Covariates:** `Covariate` (static, categorical or time-varying; missing
  values refused), `CovariateEffect` with `SendEffect`, `ReceiveEffect`,
  `MatchEffect`, `DiffEffect`, `SimEffect`, `AverageEffect`, `MinimumEffect`,
  `MaximumEffect`, `SumEffect` and `ProductEffect`; `TieEffect`;
  `GlobalEffect`.
- **Interactions:** `Interaction` (product terms over any statistics of the
  ecosystem), `Transformed`, `Standardized`, `split_by_type`,
  `matching_third`, `MatchedDegree`, `TertiusEffect` (including
  `aggregate=:entropy`).
- **The full-risk-set engine** (`src/engine/`), which was the never-released
  package Relevent.jl and is now part of Revel: the ordinal and the
  exponential interval-timing likelihoods of `relevent::rem.dyad` over every
  dyad (`Revel.fit_obpm`, `Revel.fit_timing`, and their results
  `Revel.OrdinalBPMResult`, `Revel.TimingModelResult`; `public`, not exported —
  `fit_revel` is the entry point), with risk sets streamed interval by
  interval under a bounded design cache (`cache=:auto|:all|:chunked|:none`,
  bit-identical fits); the shared tie policies (`:error`, `:ordered`,
  `:breslow`, `:efron` for the ordinal likelihood; `:error`, `:ordered`,
  `:batch` for the timing one); the shared separation verdict; the `t0`/`t_end`
  observation window; the StatsAPI surface (`coefnames` included) and the
  result-metadata protocol. `InteractionHistory`, `update_history!`, `PShift`
  and `pshift_types` are now defined in Revel, and the timing trait is
  `Revel.is_interval_constant` (`public`). Relevent.jl's own statistics are
  not carried over: relevent's effects are Revel calls listed in
  `effect_catalogue()` and in the concordance guide ("Coming from relevent"),
  where the actor fixed effects `FESnd`, `FERec` and `FEInt` gained catalogue
  rows.
- **Designs and fitting:** `each_risk_set`, `RiskSetView`, `event_design`,
  `two_mode_dyads`; `fit_revel` (alias `revel`) and `RevelFit`;
  `fit_receiver_choice`, `fit_stratified`, `fit_moving_window`,
  `compare_coefficients` (with Wald tests of differences between strata or
  non-overlapping windows), `profile_memory` (with a profile-likelihood
  confidence set); `simulate_events`. A `cases` predicate that depends on who
  acts restricts each case's risk set to the dyads it admits, which is the
  conditional likelihood of the selected events. Sampled controls cost only
  the rows drawn.
- **Goodness of fit:** `event_diagnostics`, `prediction_summary`,
  `score_process_test`, `score_test` (exact under `ties=:efron` for tie blocks
  of up to seven events), `gof` (a method of the shared `NetworkCore.gof`, with
  an overall Mahalanobis p-value; `refit=true` gives a refitting reference
  distribution, threaded, reproducible from `rng` and independent of the
  thread count), `mechanism_shares`, `closing_times`,
  `statistic_collinearity`.
- **Relational hyperevents:** `HyperEvent`, `HyperHistory`, subset
  repetition, closure and attribute statistics, `hyper_design` (by default the
  sender-stratified design of Lerner & Lomi 2023), `fit_rhem` (alias `rhem`),
  `simulate_hyperevents`.
- **`effect_catalogue`**, the concordance with relevent, remstats, rem,
  goldfish and eventnet, from which the documentation's tables are rendered.
- **Separation** follows the ecosystem's shared verdict and policy on every
  route: when no finite maximum exists, the fit warns, reports
  `converged == false`, names the separated coefficients in
  `fit.fit.separated`, and withholds z values, p-values and confidence
  intervals. The diagnostics refuse such a fit, and `compare_coefficients`
  withholds the tests that involve it.
- **Refusals instead of silent results:** a REM.jl statistic (no history
  interface), a `GlobalEffect` outside an `Interaction`, self-loops,
  non-finite event times, a `Standardized` statistic over a risk set other
  than the model's, `ties=:efron` with sampled controls (REM.jl's
  `check_tie_sampling`), and a diagnostic on a fit that did not converge or
  has singular information each raise an `ArgumentError` that names the
  reason.
- A golden fixture from remstats 4.1.0 and remify 4.1.0 (141 statistic
  arrays), regenerated with
  `Rscript test/fixtures/r/revel_remstats.R > test/fixtures/revel_remstats.toml`.
- Two golden fixtures from relevent 1.2.1, `relevent_catalogue.toml` (24 design
  columns, compared with Revel's statistics at `1e-12`) and
  `relevent_rem_dyad.toml` (fitted ordinal and timing models, `1e-6`), with
  their generating scripts in `test/fixtures/r/`.
- A PrecompileTools workload that replays the README's calls.

### Design decisions

- Revel.jl hosts no optimizer: every likelihood is maximised by
  `NetworkCore.newton_fit`. `fit_revel` routes the ordinal model on the full
  directed risk set to `Revel.fit_obpm` (`engine=:stream`), every other ordinal
  model to `REM.fit_rem` on the design frame (`engine=:design`), and the timing
  model to `Revel.fit_timing`.
- Every statistic implements both compute interfaces of the ecosystem
  (`InteractionHistory` and `REM.EventNetworkState`) through one internal
  method, and extends the shared `compute`/`name` generics by name.
- The entry points evaluate private copies of the statistics, so one
  specification can be fitted from several tasks at once.
- The zero-history value of a proportion is the `empty` keyword, not a hidden
  default; the way the legs of a triad are combined is the `combine` keyword.
- On a symmetric layer an actor's out-, in- and total degree coincide (the
  events it took part in), and a degree share is taken of the number of
  events.

### Known limitations

- **Not implemented** (refused or absent; see the README): random effects,
  frailties, random slopes and cross-level interactions; smooth non-linear and
  time-varying effects; a dyad × event-type risk set; events with duration and
  active-state statistics; the sender-rate step of
  actor-oriented models and DyNAM-i; Weibull/Gompertz baselines and integrated
  time-varying hazards; main effects of global covariates from time-shifted
  controls; group-addressed participation shifts as a risk set; pairwise
  time-ordered transitivity; max-based turn-taking statistics; informR
  sequence statistics; Bayesian estimation, penalisation and mixtures
  (relevent's BPM and BSIR included); relevent's event-indexed covariate
  arrays and `conditioned.obs`; relevent 1.2.1's output for `FrPSndSnd`,
  `FrRecSnd` and `OSPSnd`, which departs from its documentation; a `missing=`
  policy for covariates; Amati et al.'s internal times, Boschi &
  Wit's auxiliary-statistic processes, and Brandenberger's segment prediction;
  for hyperevents, two-mode and generalised hyperevents, geometric weighting,
  closure of order `(p, q, l)` and switch reciprocation, eventnet's four-cycle
  and neighbour statistics, time-varying hyperedge effects, the outcome model,
  Efron ties, a timing likelihood, and goodness of fit and the other
  diagnostics.
- **Deliberate differences from remstats 4.1.0**, pinned by the golden
  fixture: the decay kernel is evaluated at the time of the event being
  explained (remstats: the previous event); `Standardized` defaults to the
  population standard deviation (`corrected=true` gives remstats' `std`); a
  degree share with nothing in memory is `empty`.
- The rem, goldfish and eventnet columns of `effect_catalogue` follow those
  packages' documentation and are not checked numerically.
- `compute` called by hand on one statistic from several tasks at once is not
  safe (the entry points work on private copies and are).
- A memory kernel without finite support costs `O(E²)` over a sequence.
- `gof` is a plug-in check by default (conservative p-values, pointwise per
  statistic); `refit=true` removes the conservatism at the cost of `n_sim`
  refits.
- `score_test` under `ties=:efron` is refused for a tie block of more than
  seven events and for a fit on a subset of `cases` with ties (the orderings
  are enumerated).
- The sandwich standard errors treat each event as a cluster.
- `ties=:efron` is available with the full risk set only (refused with
  sampled controls).
- A dyad-dependent selection of cases given as indices or a mask cannot be
  detected; pass it as a predicate so that the risk set is restricted.
- The `[compat]` bounds on the sibling packages are `0.2`, which cannot tell
  states of their `main` branches apart until they are tagged.
