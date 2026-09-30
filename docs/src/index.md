# Revel.jl

```@raw html
<p>Specify, fit and check relational event models with the <strong>effects, interactions and diagnostics of the relational event literature</strong>. Revel.jl builds on <a href="/Relevent.jl/dev/">Relevent.jl</a> and <a href="/REM.jl/dev/">REM.jl</a>: its statistics plug into their fitters, and it adds what a review of the literature from 2008 to 2026 found missing between the packages.</p>
```

**Start here:** [Getting started](getting_started.md) ·
[Endogenous effects](guide/effects.md) · [Interactions](guide/interactions.md) ·
[Goodness of fit](guide/gof.md) · [Concordance](guide/concordance.md)

## What the package is built around

A review of 210 works on relational event models found that the literature does
not contain hundreds of distinct effects. It contains about **five structural
configurations** — the dyad, the reversed dyad, node degree, the two-path and
the three-path — crossed with a small number of **measurement choices**: how
past events are weighted, which events count, how the statistic is scaled.
The same configuration under a different measurement carries a different name
in each package, which is why "inertia" is a proportion in relevent, a count in
remstats, a decayed weight in rem and an indicator in goldfish.

Revel.jl is organised the same way:

| Layer of the design | What it holds |
|---|---|
| [`EventLayer`](@ref) | the measurement: a memory kernel, event-type and attribute filters, event weights, the clock |
| five parametric statistics | [`DyadEffect`](@ref), [`DegreeEffect`](@ref), [`DyadDegreeEffect`](@ref), [`TwoPathEffect`](@ref), [`FourCycleEffect`](@ref) |
| named constructors | the literature's vocabulary: [`Inertia`](@ref), [`Reciprocation`](@ref), [`OTP`](@ref), [`ITP`](@ref), [`OSP`](@ref), [`ISP`](@ref), … |
| covariates | actor, dyadic and global covariates, constant or time-varying |
| interactions | product terms, filtered statistics, type splits, attribute-weighted statistics, stratified and moving-window fits |
| diagnostics | prediction, score processes, score tests, simulation, closing times, collinearity |

[`effect_catalogue`](@ref) maps every effect onto its name in relevent,
remstats, rem, goldfish and eventnet.

## Fit a model to radio calls

The bundled World Trade Center data record 481 radio calls among 37 officers;
the covariate marks the officers who held an institutional coordinator role.

```@raw html
<p>Use Julia <strong>1.12+</strong> and the <a href="/getting-started/">workspace installation guide</a> for the current <strong>0.1.0 development version, unreleased</strong>. The examples assume that environment is already prepared.</p>
```

```julia
using Networks, Revel

wtc = load_dataset(:wtc_police_calls)
calls = [Event(row[2], row[3], Float64(row[1])) for row in eachrow(wtc.events)]
coordinator = SumEffect(Float64.(wtc.is_icr); name="coordinator")

stats = [coordinator,
         Inertia(transform=:log1p),          # how often has s called r?
         Reciprocation(transform=:log1p)]    # how often has r called s?
fit = fit_revel(calls, stats, wtc.n_actors)
coeftable(fit)
```

The fit is the exact ordinal likelihood over all 1332 directed dyads. Which
effect is the model missing? A score test screens candidates without refitting:

```julia
candidates = [PShift(:AB_BA),                 # an immediate reply
              RecencyRank(:send),             # whom s called most recently
              OTP(transform=:log1p), ISP(transform=:log1p)]
score_test(fit, candidates)
```

The immediate reply dominates, as one expects of radio traffic. Adding it, and
asking how well the model now predicts who calls whom:

```julia
better = fit_revel(calls, [stats; PShift(:AB_BA); RecencyRank(:send)], wtc.n_actors)
prediction_summary(better).recall      # share of calls ranked in the top 1, 5, 10
```

## What can be fitted

| Task | Path |
|---|---|
| Which dyad acts next (event order) | [`fit_revel`](@ref), full or restricted risk set, all or some events as cases |
| Receiver choice given the sender (actor-oriented) | [`fit_receiver_choice`](@ref) |
| Two-mode and undirected events | `riskset=`[`two_mode_dyads`](@ref)`(…)`, `directed=false` |
| Waiting times as well (exponential baseline) | `fit_revel(…; model=:timing)` |
| Large networks | `n_controls=` (sampled controls), or the statistics inside `REM.fit_rem` |
| Effects that differ by period, context or event type | [`fit_stratified`](@ref), [`fit_moving_window`](@ref), [`Interaction`](@ref) |
| Multi-actor events | [`fit_rhem`](@ref) ([relational hyperevents](guide/hyperevents.md)) |

```@raw html
<p>Revel.jl hosts no optimizer and no likelihood of its own: models are fitted by <code>Relevent.fit_obpm</code>, <code>Relevent.fit_timing</code> and <code>REM.fit_rem</code>, on the shared <code>Networks.newton_fit</code>. Random effects, smooth (non-linear) effects, a dyad × type risk set, events with duration and the sender-rate step of actor-oriented models are <strong>not implemented</strong>; the <a href="guide/concordance/">concordance</a> lists what is refused and why.</p>
```

```@docs
Revel
```
