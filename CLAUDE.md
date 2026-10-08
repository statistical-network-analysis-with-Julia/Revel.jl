# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Revel.jl specifies, fits and checks relational event models: the exact full-risk-set likelihoods of R's `relevent::rem.dyad` (the ordinal and the exponential interval-timing model), relational event effects, effect × covariate interactions, goodness-of-fit diagnostics and relational hyperevents. Beyond the `rem.dyad` likelihoods it is not a port of one R package: it implements the effect catalogue of a scoping review of the relational event model literature (the review is unpublished; the report and notes live in the private development workspace, not in this repository; the bibliography is `docs/references.bib`).

It depends on the sibling checkouts `../NetworkCore.jl` and `../REM.jl` through `[sources]` paths; `../ERGM.jl`, `../SNA.jl` and `../Siena.jl` are test-only (the co-loading test).

**The engine was Relevent.jl.** Until 2026-10-07 the full-risk-set fitters, `InteractionHistory` and `PShift` lived in the sibling package Relevent.jl, which was never released; they were moved into `src/engine/` before Revel's first registration, so that a registered Revel never depends on Relevent's UUID. Relevent.jl's repository is frozen (its README says where each name went) and will be archived; do not add it back as a dependency, and do not add a forwarding shim. Relevent's own statistics (`PriorInteraction`, `CovSnd`, `NIDSnd`, `FESnd`, …) were not carried over: each relevent effect is a Revel call (`effect_catalogue()`, the concordance guide's "Coming from relevent").

The package was created on 2026-09-30. The Documenter build needs a git checkout with at least one commit (`git rev-parse HEAD`).

## Development Commands

```bash
julia --project -e 'using Pkg; Pkg.test()'

# Strict Documenter build (warnonly=false, checkdocs=:exports)
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl

# Allocation-regression gate (CI runs it on the ubuntu / 1.12 cell)
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/regression_tests.jl

# Regenerate the remstats golden fixture (needs R with remstats 4.1.0 and remify)
Rscript test/fixtures/r/revel_remstats.R > test/fixtures/revel_remstats.toml

# Regenerate the relevent golden fixtures (needs R with relevent 1.2.1)
Rscript test/fixtures/r/relevent_rem_dyad.R > test/fixtures/relevent_rem_dyad.toml
Rscript test/fixtures/r/relevent_catalogue.R > test/fixtures/relevent_catalogue.toml
# Evidence for the three relevent effects Revel does not reproduce (prints, no fixture)
Rscript test/fixtures/r/relevent_deferred_probe.R
```

`test/runtests.jl` is one file of flat `@testset` blocks. To run every testset and see all failures at once (a failing top-level testset otherwise aborts the file), wrap it: `julia --project -e 'using Test; @testset "all" begin include("test/runtests.jl") end'`.

## Architecture

The organising idea comes from the review: the literature contains about five structural configurations crossed with a few measurement choices. The code keeps the two apart.

| File | Holds |
|---|---|
| `src/engine/history.jl` | `InteractionHistory` (the time-ordered `events`, nothing else), `update_history!` |
| `src/engine/pshift.jl` | `PShift`, `pshift_types` — Gibson's thirteen shifts, null actor included |
| `src/engine/risksets.jl` | the tie policies of the two likelihoods, `_RiskSetPlan`, the cache policies (`_CachedRiskSets`, `_StreamedRiskSets`, `_resolve_cache`, `_each_interval`) |
| `src/engine/likelihoods.jl` | `_obpm_derivatives`, `_timing_derivatives`: the `θ -> (ll, grad, hess)` closures for `NetworkCore.newton_fit` |
| `src/engine/separation.jl` | margin rows and `_separation_verdict` (the shared NetworkCore verdict on the full risk set) |
| `src/engine/ordinal.jl`, `timing.jl`, `results.jl` | `fit_obpm`, `OrdinalBPMResult`; `is_interval_constant`, `fit_timing`, `TimingModelResult`; their StatsAPI surface and metadata |
| `src/memory.jl` | `AbstractMemory` kernels (`FullMemory`, `HalfLife`, `Window`, `Interval`, `PowerLaw`, `LinearDecay`, `KernelMemory`), `kernel_weight` |
| `src/layers.jl` | `EventLayer` — memory, `types`, `keep`, `weighted`, `clock`, `symmetric` — and its synced state `_LayerState` |
| `src/statistics.jl` | `AbstractRevelStatistic`, the two `compute` methods, `name`, the traits |
| `src/endogenous.jl` | `DyadEffect`, `DegreeEffect`, `DyadDegreeEffect`, `TwoPathEffect`, `FourCycleEffect` and the named constructors; recency, p-shifts, neighbourhood statistics |
| `src/covariates.jl` | `Covariate`, `CovariateEffect`, `TieEffect`, `GlobalEffect`, `TertiusEffect`, `MatchedDegree` |
| `src/interactions.jl` | `Interaction`, `Transformed`, `Standardized`, `split_by_type` |
| `src/design.jl` | `each_risk_set` (the one streaming pass everything consumes), `event_design` |
| `src/fit.jl` | `RevelFit`, `fit_revel`, stratified / moving-window / memory-profile fits |
| `src/simulate.jl` | `simulate_events` |
| `src/gof.jl` | prediction, score processes, score tests, `gof`, sequence metrics, collinearity |
| `src/hyper.jl` | relational hyperevents: `HyperEvent`, statistics, `hyper_design`, `fit_rhem` |
| `src/catalogue.jl` | `effect_catalogue` — the cross-package concordance |

### One internal method, two public interfaces

A statistic subtypes `AbstractRevelStatistic` and implements `_value(stat, events, sender, receiver, t::Float64)` on the raw event vector. `statistics.jl` derives both public methods from it, on the shared `compute` generic (imported by name from REM, which re-exports NetworkCore'):

- `compute(stat, history::InteractionHistory{T}, s, r, t::T)` — the history interface, used by `fit_obpm`/`fit_timing` and by `each_risk_set`;
- `compute(stat, state::REM.EventNetworkState, s, r)` — REM's, so the statistics work in `fit_rem`.

The two cannot drift apart because neither has code of its own. REM's event log holds `(sender, receiver, time, weight)` tuples with **no event type**, so a layer with `types=` throws on that interface. The wrappers (`Interaction`, `Transformed`, `Standardized`) define the two `compute` methods directly instead, because their parts may be foreign statistics that have no `_value`.

Traits: `_uses_history(stat)` feeds `REM.needs_history` (default `true`; covariate effects say `false`), and `_interval_constant(stat)` feeds `Revel.is_interval_constant` (`public`; default `false`; `true` for layers on `FullMemory`/`HalfLife(Inf)` or on the event clock `clock=:order` — `_layer_interval_constant` — static covariates, recency ranks, the p-shifts and `TimeSince(…; clock=:order)`), which is what gates the timing likelihood.

Every entry point that evaluates statistics (`_each_risk_set`, both `fit_revel` routes, `simulate_events`) works on `_fresh(stats)`: a `deepcopy` whose `Base.deepcopy_internal` methods give each `EventLayer` (and `Standardized`'s `_StdCache`) an empty cache and merge layers with identical keywords (the interning table lives in deepcopy's `IdDict`). This is what makes one specification safe to fit from several tasks and keeps a `RevelFit`, which stores the user's originals, from retaining caches. A new mutable cache inside a statistic needs the same `deepcopy_internal` treatment.

### Layers

`_sync!(layer, events, t)` is called at the top of every `_value`. It finds the layer's state for that event vector (by `===`, through a `WeakRef`), absorbs the events at or before `t` that were appended since the last call through a cursor (`_n_before` bisects; order and finiteness are checked as events are absorbed), rebuilds if the vector was reset or rewritten (the signatures — sender, receiver, time, weight, type — of the first and last absorbed events are checked; this is what makes the engine's streamed risk sets, which replay one history in place, work), and for non-accumulating kernels retakes the snapshot when the clock or the cursor moved.

- Per-dyad data (`W`, `L`, `first_t`, `last_t`, `last_i`) live in vectors indexed by a **slot**, one per dyad with a history; `_slot(st, i, j)` finds it through an `Int32` matrix while actor IDs stay ≤ `_DENSE_MAX` (2,048) and through a `Dict` beyond. In the dense range `W` (and `L` on a half-life layer) are also mirrored in `n × n` matrices `Wm`/`Lm`, which `_w` reads — the two-path loops need that speed. Every write to `W`/`L` must update the mirror (`_add_dyad!`, `_snapshot!`, `_reset!`, `_grow!`). Per-actor vectors grow geometrically. Read per-dyad values only through `_w`, `_first_t`, `_last_t`, `_last_i`.
- `FullMemory` and `HalfLife` are accumulated: `W` holds the weight, `L` the clock of the last update, and a read decays lazily. Each event is absorbed once.
- Every other kernel keeps the accepted events (with their slots) and recomputes `W` by walking them backwards up to `_support(memory)`. `touched` lists the slots to zero before the next snapshot.
- Neighbour lists are append-only supersets, so consumers check `w > 0`.
- On a `symmetric` layer every event is recorded in both directions; `_actor_degree` returns the out-degree for `:total` there and `_degree_share_base` does not double the divisor.

`compute` must allocate nothing on a warmed history — pinned in `test/runtests.jl` and `benchmark/regression_tests.jl`. The constructors are keyword-heavy and build type-parameterised structs (`DyadEffect{L,F}`), which is what keeps the hot path free of dynamic dispatch; do not add an abstractly typed field.

### Fitting routes

Revel hosts no Newton loop (the ecosystem's shared-numerics rule: every likelihood goes to `NetworkCore.newton_fit`). `fit_revel` routes:

- ordinal + full directed risk set + every event a case + no sampling + `se=:hessian` → `fit_obpm` (`engine=:stream`);
- any other ordinal model → `event_design` → `REM.fit_rem(::DataFrame, names)` (`engine=:design`);
- `model=:timing` → `fit_timing` (`engine=:stream`).

`fit_obpm`, `fit_timing`, `OrdinalBPMResult`, `TimingModelResult` and `is_interval_constant` are `public`, not exported (`Revel.fit_obpm`): `fit_revel` is the entry point.

### The full-risk-set engine (`src/engine/`)

- **Risk sets stream: `cache=`.** `_RiskSetPlan` is the `O(E + n²)` skeleton (case index, waiting time, read time, which events the history absorbs after each interval, the sparse Efron weights). `_RiskSets` holds the `n(n−1) × p` design matrices under `:all` (every one, sharing one across a frozen tie block), `:chunked` (a bounded cache refilled by replaying the history) or `:none` (`chunk = 1`); `:auto` is `:all` under `cache_bytes`. **The fits are bit-identical across policies — `==`, and the tests assert `==`.** If a cache mode moves a coefficient, the streaming is wrong.
- **The derivative closures allocate O(p²).** `_obpm_derivatives(rs)` and `_timing_derivatives(rs)` are named functions so the `@allocated` pins (suite and `benchmark/regression_tests.jl`) measure the code that runs: ≤ 512 bytes per evaluation, independent of `E` and the risk set. Do not add a temporary in their loops.
- **Ties: two likelihoods, two policy sets** (the shared `NetworkCore.TIE_POLICIES`, both defaulting to `:error`). `fit_obpm` (order): `:ordered`, `:breslow` (history frozen across the block), `:efron` (plus `1 − (j−1)/d` weights; tied cases must be distinct dyads); `:batch` is refused because with the history frozen it *is* Breslow. `fit_timing` (exact time): `:ordered` (zero-length waiting intervals), `:batch` (one exposure interval); `:breslow`/`:efron` are refused — they correct a partial likelihood. `is_exact` is `false` as soon as the data carried a tie.
- **Exact timing admissibility.** `fit_timing` requires `is_interval_constant` of every statistic (default `false`; a custom statistic opts in by a method). Do not re-enable endpoint-frozen finite decay: with events 1→2 at 1 and 2→1 at 3, a half-life-1 inertia and β = [0, 1], the endpoint value gave ℓ = −6.568 where integrating the actual hazard gives −7.517.
- **The observation window.** `t0` is the onset (default 0, as in relevent); `t_end` adds the right-censored tail (`case_idx == 0`). `relevent::rem.dyad` always has that tail (its last edgelist row is the end of observation), and its temporal likelihood has no intercept (R gets a constant `CovSnd` column).
- **Separation.** `_separation_verdict(rs; timing)` streams the risk sets once, deduplicates the margin rows exactly (`_MarginRows`), falls back to row generation above `cache_bytes`, and calls `NetworkCore.separation_from_margins` before Newton runs (whose warnings are then silenced: they are symptoms). Ordinal rows `x_case − x_j` (`:clogit`); timing rows in `(1, x)` (`:poisson`): `−z_j` for every dyad of an interval with exposure, `+z_case` for every event. `SeparationVerdict` has no keyword constructor; it is built positionally in one place.
- **Show headers** are "Ordinal relational event model (full risk set)" and "Interval-timing relational event model (full risk set)".

The "fit_revel: the three estimators agree" testset pins that the routes agree to 1e-8 on a common model. `RevelFit{F,T,R}` stores the inner result plus everything `each_risk_set` needs (sorted events, statistics, risk-set spec — concretely typed as `R` — cases, tie policy), which is how the diagnostics rebuild the risk sets from a fit alone. `RevelFit` and `HyperFit` forward the StatsAPI verbs to the inner result and define `coefnames` (the StatsAPI binding) from the statistics' names, preceded by `"log_baseline"` for a timing fit; the tests pin it on every route with `check_statsapi(...; required=(STATSAPI_VERBS..., :coefnames), strict=true)` and `coefnames(fit) == coeftable(fit).names == coefnames(fit.fit)`.

Three rules live in the design:

- **Efron × sampled controls is refused**: `event_design`'s `draw` calls REM.jl's `public` `REM.check_tie_sampling` for every Efron tie block it would sample, so Revel and REM refuse the same combination with the same words (the forced tied rows biased the estimate toward zero, 0.654 for 1.0). `:breslow` is the correction for sampled designs; the "ties=:efron with sampled controls is refused (REM's guard); :breslow is unbiased" testset simulates it (150 reps, coverage in [0.88, 0.99]).
- **A dyad-dependent `cases` predicate restricts the risk set**: `_case_restricted_riskset` evaluates the predicate on the candidate event `Event(s, r, time; eventtype, weight)` of every dyad of every case's base risk set; if it ever rejects one, the risk set becomes a `_CaseRestricted{C,B} <: Function` (the base spec plus the predicate; `_riskset_provider` filters the base provider per event). `fit_revel` does this before building the mask (on `(min, max)` events when undirected) and stores the restriction as the fit's `riskset`, so the diagnostics rebuild the same strata; `_each_risk_set` does it for direct `event_design`/`each_risk_set` calls. The fit equals `fit_stratified`'s stratum to 1e-10 (pinned). `Standardized` cannot follow a per-case risk set and is refused with its own message.
- **Separation is decided by the estimators, never by `fit_revel`.** `fit_obpm`/`fit_timing` and `REM.fit_rem` each call NetworkCore's shared verdict on the risk sets they fit and follow the shared policy (warn, `converged == false`, names in the inner result's `separated`, NaN z/p/CI). Revel only reads it: `_separated(fit)` (`!isempty(fit.fit.separated)`) makes `_require_converged` refuse with a separation message, and `compare_coefficients` puts NaN in `z` and in every `p_difference` involving a separated fit. Pinned by "Separation: every route warns, flags and withholds inference".

### Hyperevents

`src/hyper.jl` is self-contained: `HyperEvent`, `HyperHistory` (the past events plus lazily built sub-hyperedge indices — a subset order is indexed the first time a statistic asks for it), statistics subtyping `AbstractHyperStatistic` with `_hvalue(stat, history, senders, receivers, t)`, the size-stratified case-control design `hyper_design`, and `fit_rhem`, which hands that design to `REM.fit_rem`. The statistics read `Σ kernel_weight(memory, t − tₑ)·wₑ` off the index lists, so every memory kernel works without an accumulator. Its testsets pin that the (1,1) and two-actor special cases equal the dyadic statistics and that `fit_rhem` with full enumeration equals `fit_revel`.

### Diagnostics

All of `gof.jl` is a consumer of `each_risk_set`. `score_process_test` and `score_test` share `_score_components` (per-event score and information under the fitted probabilities); both are refused for timing fits (the coefficients do not maximise the partial likelihood), for fits with sampled controls and for fits that did not converge or whose standard errors are not finite (`_require_converged`, which every diagnostic calls: a singular information leaves the coefficients undetermined, and a separated fit has no maximum). `score_test` goes through `_score_and_information`: under `ties=:efron` every tie block is taken at its **exact** partial likelihood (`_exact_block!`: a depth-first enumeration of the orderings sharing prefixes, gradient and Hessian checked against finite differences), the null model is re-maximised by Newton steps on that likelihood, and the statistic uses the efficient score — Efron's own information understated its score's variance (7.6 % to 57 % rejections on true models; now 5.7 %, as Breslow and untied). Blocks above `_EXACT_TIE_LIMIT = 7` and Efron fits on `cases` with ties are refused, pointing at `score_process_test`. `gof(...; refit=true)` (`_gof_refit`) refits every simulated sequence and compares residuals from the mean of `n_inner` simulations of each refit; one seed per replicate is drawn up front (`threaded=false` gives the same bits). Per-replicate flags written from the `@threads` loop must be byte-addressed (`_failure_flags` → `Vector{Bool}`), never a `BitVector`, whose 64-flag words make concurrent writes race; a testset pins it under `:greedy` scheduling, and checks that the lowered code of `_gof_refit` calls `_failure_flags` and never `falses`/`BitVector`. `simulate_events(...; times, ties=:efron)` draws a tie block without replacement (an Efron fit refuses a dyad acting twice at one time); `:breslow` draws independently. `gof` simulates an ordinal fit conditional on the observed event times, types and weights (freezing tie blocks under `:breslow`/`:efron`), and its `p_overall` is a leave-one-out Mahalanobis rank test over all auxiliaries. The statistical claims of the diagnostics are pinned by simulation in the tests (size over 40 to 200 data sets, power against an omitted effect); keep them rate-based, never tuned to one seed. Size tests are two-sided. The resampling p-value and score-test calibration testsets use `_size_two_sided` on 200 true-model p-values: 2 to 21 rejections at 5 % (each tail about 4e-4 under an exact test) and a Kolmogorov distance `_ks_uniform` below 0.15. The distance alone misses a moderately conservative test (p-values uniform on (0.06, 1) never reject, D ≈ 0.06); the lower rejection bound catches it.

## Release engineering

- **Precompile workload** (`src/Revel.jl`, bottom): simulate, the exact fit (the streamed engine) and a sampled one (design engine → `REM.fit_rem`), `coeftable`/`show`, a score test, `prediction_summary` and a two-replicate `gof` on a 5-actor toy. It must stay silent (every fit converges). PrecompileTools is a dependency: any environment that sources Revel by path (the root workspace, the site's `.snippet-env`) needs `Pkg.resolve()` after pulling it.
- An `Aqua.test_all(Revel)` testset runs; Aqua and Test have `[compat]` entries.

## Validation

- `test/fixtures/revel_remstats.toml` (script `test/fixtures/r/revel_remstats.R`, remstats 4.1.0): 141 statistic arrays, including typed (`consider_type = "separate"`), `a:b` and `event()` keys. The generator needs remstats/remify 4.1.0 (a private library can be named in `R_LIBS_USER`). Tolerance 1e-10 — everything is a deterministic count or a closed-form decay; do not loosen it. Three known differences are asserted as exact relationships, not as `@test_broken`: decay evaluated at the previous event's time in remstats, the sample standard deviation in `std`, and the share of an empty memory.
- `test/fixtures/relevent_catalogue.toml` (script `test/fixtures/r/relevent_catalogue.R`, relevent 1.2.1): relevent's own design statistics, read off `rem.dyad.lambda` with one unit coefficient — 24 columns on every candidate dyad before each of 14 events. The testset evaluates the Revel call `effect_catalogue()` names for each (the fixed effects as indicator covariates) and asserts the catalogue says so; tolerance 1e-12 (measured: exact), in both compute interfaces.
- `test/fixtures/relevent_rem_dyad.toml` (script `test/fixtures/r/relevent_rem_dyad.R`): a real `rem.dyad` fit of both likelihoods (8 actors, 100 simulated events; CovSnd → `SendEffect`, CovRec → `ReceiveEffect`, four p-shifts). Tolerance 1e-6 is the reference BFGS termination slack (`reltol = 1e-15`), not a Monte-Carlo allowance — a failure means the likelihood differs; do not widen it. It is re-asserted under every cache policy, and the `t_end` bias is pinned (> 0.04 in log λ₀ without the tail).
- The half-life statistics are checked against a direct rescan of the past at every evaluation (1e-10), in both interfaces.
- Every memory kernel is checked against `ref_weight`, a brute-force evaluation of the definition, in the test file's preamble.

## Conventions

- Names must not collide with any export of the ecosystem: REM already owns `Reciprocity`, `Repetition`, `FourCycle`, `TransitiveClosure`, `SenderActivity`, …, hence `Reciprocation`, `FourCycleEffect`, `OTP`, `HyperSenderActivity`; Siena owns `SameEffect`, `DifferenceEffect` and `SimilarityEffect`, hence `MatchEffect`, `DiffEffect`, `SimEffect`. The "Namespace" testset pins that any name shared with REM or NetworkCore is the same binding, and "Namespace: co-loading with ERGM, SNA and Siena" checks in a fresh process that every export survives `using ERGM, SNA, Siena, REM, Revel`. Before adding an export, check it against all fourteen sibling packages, not only those three. (The retired Relevent.jl exports `InteractionHistory`, `PShift`, … as different bindings; it must not be loaded beside Revel.)
- `Event` and `n_simulations` are re-exported unchanged so that `using Revel` is self-sufficient; they are documented where they are defined. `InteractionHistory`, `update_history!`, `PShift` and `pshift_types` are Revel's own. `is_directed` is a method of the shared Graphs/NetworkCore generic.
- Every exported **and `public`** name has a docstring with a runnable example (`names(Revel)` lists both); the engine's examples must also run without a warning ("Engine: the docstring examples run without warnings").
- Every exported name has a docstring with a runnable ```` ```julia ```` example (a testset executes them all), and the strict docs build requires it to be listed in `docs/src/api/*.md`. A trailing `# value` comment that parses as a literal (after an optional `≈`, cut at ` — `, `; ` or two spaces, and at the last ` = `) is a **claim**: `check_block` in the test preamble compares it with the value, to half a unit of its last digit (1 % after `≈`). The README and every page of `docs/src` run through the same checker; write prose comments so they do not parse as a literal, or make them true. The guide pages use `@example` blocks, so the published site shows real output. Docstrings on shared generics are attached to a method without type parameters so that `@docs` can address them (`compute(::AbstractRevelStatistic, ::REM.EventNetworkState, ::Int, ::Int)`).
- A new effect belongs in `effect_catalogue`; the tables of `docs/src/guide/concordance.md` are rendered from it by `julia --project=docs docs/render_concordance.jl`, and a testset keeps the two in step.
- `docs/references.bib` is the review's bibliography (keys `<firstauthor><year><firsttitleword>`, issue year) plus the methodological references the package cites; a testset checks that every key named in `docs/src/guide/literature.md` exists. The PDFs behind it are in the workspace's private `literature/` folder and must never be committed.
- All randomness flows through an `rng::AbstractRNG` keyword.
- Unimplemented features are refused with an `ArgumentError` and listed in the README "Not implemented" section and the CHANGELOG "Known limitations".
- Record any behavioural change in `CHANGELOG.md` under `[0.1.0] - Unreleased`.
