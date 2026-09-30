# Getting Started

## Installation

Revel.jl requires Julia 1.12+ and is not yet registered. It finds its three
sibling packages through `[sources]` paths, so clone them side by side:

```bash
git clone https://github.com/statistical-network-analysis-with-Julia/Networks.jl
git clone https://github.com/statistical-network-analysis-with-Julia/REM.jl
git clone https://github.com/statistical-network-analysis-with-Julia/Relevent.jl
git clone https://github.com/statistical-network-analysis-with-Julia/Revel.jl
cd Revel.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Events

An event is a sender, a receiver and a time, optionally with a type and a
weight. Revel re-exports `Event` from REM.jl. Actors are numbered `1:n`, and the
clock must be numeric.

```julia
using Revel

events = [Event(1, 2, 1.0), Event(2, 1, 2.5), Event(1, 3, 3.0),
          Event(3, 1, 4.2; eventtype=:reply, weight=2.0)]
```

## Statistics

A statistic answers one question about a candidate event `s → r`, given the
events that came before it. `compute` evaluates it against a history:

```julia
history = build_history(events)

compute(Inertia(), history, 1, 2, 5.0)          # past 1 → 2 events: 1.0
compute(Reciprocation(), history, 1, 3, 5.0)    # past 3 → 1 events: 1.0
compute(OutdegreeSender(), history, 1, 2, 5.0)  # events sent by 1: 2.0
```

Every endogenous statistic takes the same measurement keywords, which select
how the past is remembered:

```julia
compute(Inertia(memory=HalfLife(1.0)), history, 1, 2, 5.0)   # 0.5^4 = 0.0625
compute(Inertia(memory=Window(2.0)), history, 1, 2, 5.0)     # 0.0 — too long ago
compute(Reciprocation(types=:reply), history, 1, 3, 5.0)     # 1.0
compute(Reciprocation(weighted=true), history, 1, 3, 5.0)    # 2.0
```

## A first model

Simulate a sequence in which actors repeat themselves and answer one another,
and recover the coefficients:

```julia
using Random

stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
sequence = simulate_events(stats, [0.8, 0.6], 8, 1000; rng=Xoshiro(1))

fit = fit_revel(sequence, stats, 8)
fit
```

`fit_revel` returns a [`RevelFit`](@ref), which answers the StatsAPI verbs:

```julia
coef(fit)
stderror(fit)
confint(fit)
aic(fit), bic(fit)
```

and says exactly what was done:

```julia
using Networks

Networks.is_exact(fit)          # true — the full risk set, no ties
Networks.fit_metadata(fit)
```

## Checking the model

```julia
prediction_summary(fit)                         # how highly were the events ranked?
score_test(fit, [OTP(transform=:log1p)])        # is a closure effect missing?
score_process_test(fit; n_sim=200, rng=Xoshiro(2))   # are the effects constant?
gof(fit; n_sim=50, rng=Xoshiro(3))              # does it reproduce the sequence?
```

## Where next

- [Memory and layers](guide/layers.md) — the measurement choices every effect
  shares.
- [Endogenous effects](guide/effects.md) and [covariates](guide/covariates.md) —
  the catalogue.
- [Interactions](guide/interactions.md) — five ways to let a covariate moderate
  an effect, and why they are not the same model.
- [Fitting](guide/fitting.md) — risk sets, ties, actor-oriented and two-mode
  models.
- [Goodness of fit](guide/gof.md) — prediction, residuals, simulation.
- [Concordance](guide/concordance.md) — relevent, remstats, rem, goldfish and
  eventnet names, and the deliberate differences.
