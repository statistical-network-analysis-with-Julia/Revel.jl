# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Revel.jl adds relational event effects, effect × covariate interactions, goodness-of-fit diagnostics and relational hyperevents on top of Relevent.jl and REM.jl. It is not a port of one R package: it implements the effect catalogue of a scoping review of the relational event model literature (the report and notes live in the workspace root under `reports/` and `research_notes/`; the bibliography is `docs/references.bib`).

It depends on the sibling checkouts `../Networks.jl`, `../REM.jl` and `../Relevent.jl` through `[sources]` paths; `../ERGM.jl`, `../SNA.jl` and `../Siena.jl` are test-only (the co-loading test).

The package was created on 2026-09-30 and has not been committed or published: until the directory is a git repository with a commit, the Documenter build fails at `git rev-parse HEAD` (build it from a throwaway clone), and CI cannot clone it as a sibling.

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
```

`test/runtests.jl` is one file of flat `@testset` blocks. To run every testset and see all failures at once (a failing top-level testset otherwise aborts the file), wrap it: `julia --project -e 'using Test; @testset "all" begin include("test/runtests.jl") end'`.

## Architecture

The organising idea comes from the review: the literature contains about five structural configurations crossed with a few measurement choices. The code keeps the two apart.

| File | Holds |
|---|---|
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

A statistic subtypes `AbstractRevelStatistic` and implements `_value(stat, events, sender, receiver, t::Float64)` on the raw event vector. `statistics.jl` derives both public methods from it, on the shared `compute` generic (imported by name from REM, which re-exports Networks'):

- `compute(stat, history::InteractionHistory{T}, s, r, t::T)` — Relevent's interface, used by `fit_obpm`/`fit_timing` and by `each_risk_set`;
- `compute(stat, state::REM.EventNetworkState, s, r)` — REM's, so the statistics work in `fit_rem`.

The two cannot drift apart because neither has code of its own. REM's event log holds `(sender, receiver, time, weight)` tuples with **no event type**, so a layer with `types=` throws on that interface. The wrappers (`Interaction`, `Transformed`, `Standardized`) define the two `compute` methods directly instead, because their parts may be foreign statistics that have no `_value`.

Traits: `_uses_history(stat)` feeds `REM.needs_history` (default `true`; covariate effects say `false`), and `_interval_constant(stat)` feeds `Relevent.is_interval_constant` (default `false`; `true` only for `FullMemory`/`HalfLife(Inf)` layers and static covariates), which is what gates the timing likelihood.

### Layers

`_sync!(layer, events, t)` is called at the top of every `_value`. It finds the layer's state for that event vector (by `===`, through a `WeakRef`), absorbs events appended since the last call through a cursor, rebuilds if the vector was reset or rewritten (the signature of the last absorbed event is checked — this is what makes Relevent's streamed risk sets, which replay one history in place, work), and for non-accumulating kernels retakes the snapshot when the clock or the cursor moved.

- `FullMemory` and `HalfLife` are accumulated: `W` holds the weight, `L` the clock of the last update, and a read decays lazily. Each event is absorbed once.
- Every other kernel keeps the accepted events and recomputes `W` by walking them backwards up to `_support(memory)`. `touched` lists the entries to zero before the next snapshot.
- Storage is dense (`n × n`, grown geometrically to the largest actor ID seen) and neighbour lists are append-only supersets, so consumers check `w > 0`.
- On a `symmetric` layer every event is recorded in both directions; `_actor_degree` returns the out-degree for `:total` there and `_degree_share_base` does not double the divisor.

`compute` must allocate nothing on a warmed history — pinned in `test/runtests.jl` and `benchmark/regression_tests.jl`. The constructors are keyword-heavy and build type-parameterised structs (`DyadEffect{L,F}`), which is what keeps the hot path free of dynamic dispatch; do not add an abstractly typed field.

### Fitting routes, no kernel

`fit_revel` hosts no Newton loop and no likelihood (the ecosystem's shared-numerics rule):

- ordinal + full directed risk set + every event a case + no sampling + `se=:hessian` → `Relevent.fit_obpm`;
- any other ordinal model → `event_design` → `REM.fit_rem(::DataFrame, names)`;
- `model=:timing` → `Relevent.fit_timing`.

The "fit_revel: the three estimators agree" testset pins that the routes agree to 1e-8 on a common model. `RevelFit` stores the inner result plus everything `each_risk_set` needs (sorted events, statistics, risk-set spec, cases, tie policy), which is how the diagnostics rebuild the risk sets from a fit alone.

### Hyperevents

`src/hyper.jl` is self-contained: `HyperEvent`, `HyperHistory` (the past events plus lazily built sub-hyperedge indices — a subset order is indexed the first time a statistic asks for it), statistics subtyping `AbstractHyperStatistic` with `_hvalue(stat, history, senders, receivers, t)`, the size-stratified case-control design `hyper_design`, and `fit_rhem`, which hands that design to `REM.fit_rem`. The statistics read `Σ kernel_weight(memory, t − tₑ)·wₑ` off the index lists, so every memory kernel works without an accumulator. Its testsets pin that the (1,1) and two-actor special cases equal the dyadic statistics and that `fit_rhem` with full enumeration equals `fit_revel`.

### Diagnostics

All of `gof.jl` is a consumer of `each_risk_set`. `score_process_test` and `score_test` share `_score_components` (per-event score and information under the fitted probabilities); both are refused for timing fits (the coefficients do not maximise the partial likelihood) and for fits with sampled controls. `gof` simulates an ordinal fit conditional on the observed event times, types and weights.

## Validation

- `test/fixtures/revel_remstats.toml` (script `test/fixtures/r/revel_remstats.R`, remstats 4.1.0): 131 statistic arrays. Tolerance 1e-10 — everything is a deterministic count or a closed-form decay; do not loosen it. Three known differences are asserted as exact relationships, not as `@test_broken`: decay evaluated at the previous event's time in remstats, the sample standard deviation in `std`, and the share of an empty memory.
- Parity with Relevent.jl's relevent-validated catalogue statistics is exact (`==`) in both interfaces.
- Every memory kernel is checked against `ref_weight`, a brute-force evaluation of the definition, in the test file's preamble.

## Conventions

- Names must not collide with any export of the ecosystem: REM already owns `Reciprocity`, `Repetition`, `FourCycle`, `TransitiveClosure`, `SenderActivity`, …, hence `Reciprocation`, `FourCycleEffect`, `OTP`, `HyperSenderActivity`; Siena owns `SameEffect`, `DifferenceEffect` and `SimilarityEffect`, hence `MatchEffect`, `DiffEffect`, `SimEffect`. The "Namespace" testset pins that any name shared with REM, Relevent or Networks is the same binding, and "Namespace: co-loading with ERGM, SNA and Siena" checks in a fresh process that every export survives `using ERGM, SNA, Siena, REM, Relevent, Revel`. Before adding an export, check it against all fifteen sibling packages, not only those three.
- `Event`, `InteractionHistory`, `update_history!`, `PShift`, `pshift_types` and `n_simulations` are re-exported unchanged so that `using Revel` is self-sufficient; they are documented where they are defined. `is_directed` is a method of the shared Graphs/Networks generic.
- Every exported name has a docstring with a runnable ```` ```julia ```` example (a testset executes them all), and the strict docs build requires it to be listed in `docs/src/api/*.md`. Docstrings on shared generics are attached to a method without type parameters so that `@docs` can address them (`compute(::AbstractRevelStatistic, ::REM.EventNetworkState, ::Int, ::Int)`).
- A new effect belongs in `effect_catalogue`; the tables of `docs/src/guide/concordance.md` are rendered from it by `julia --project=docs docs/render_concordance.jl`, and a testset keeps the two in step.
- `docs/references.bib` is the review's bibliography (keys `<firstauthor><year><firsttitleword>`, issue year) plus the methodological references the package cites; a testset checks that every key named in `docs/src/guide/literature.md` exists. The PDFs behind it are in the workspace's private `literature/` folder and must never be committed.
- All randomness flows through an `rng::AbstractRNG` keyword.
- Unimplemented features are refused with an `ArgumentError` and listed in the README "Not implemented" section and the CHANGELOG "Known limitations".
- Record any behavioural change in `CHANGELOG.md` under `[0.1.0] - Unreleased`.
