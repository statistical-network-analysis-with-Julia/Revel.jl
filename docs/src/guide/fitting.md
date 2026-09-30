# Fitting

[`fit_revel`](@ref) fits a relational event model with any statistics of the
ecosystem. Revel.jl contains no optimizer and no likelihood kernel: it routes
the model to the estimator that owns its likelihood and keeps what is needed to
rebuild the risk sets afterwards.

| Model | Estimator |
|---|---|
| ordinal, full directed risk set, every event a case | `Relevent.fit_obpm` — exact, bounded memory |
| ordinal, anything else | [`event_design`](@ref), then `REM.fit_rem` |
| interval timing | `Relevent.fit_timing` |

```julia
using Revel, Random

stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 400; rng=Xoshiro(1))
fit = fit_revel(events, stats, 6)
```

The two ordinal routes maximise the same likelihood and agree to optimiser
tolerance:

```julia
coef(fit) ≈ coef(fit_revel(events, stats, 6; engine=:design))    # true
```

## The risk set is the estimand

The likelihood is conditional on the risk set, so the actor universe is an
argument (`n_actors`), never inferred from who happened to act.

```julia
fit_revel(events, stats, 6; riskset=:sender)      # receiver choice
fit_revel(events, stats, 6; riskset=:active)      # only the dyads that ever occur
```

| `riskset` | Meaning |
|---|---|
| `:full` | every ordered dyad among `1:n_actors` |
| `:active` | the dyads that occur somewhere in the sequence (remify's "active") |
| `:sender` | the observed sender's possible receivers — DyNAM's choice step |
| `:receiver` | the observed receiver's possible senders |
| a vector of dyads | e.g. [`two_mode_dyads`](@ref) |
| `(index, event) -> dyads` | a risk set that changes over time |

### Actor-oriented choice

```julia
choice = fit_receiver_choice(events, stats, 6)
coeftable(choice)
```

The sender-rate step (who acts next, and when) is not implemented.

### Two-mode events

```julia
users, pages = 1:4, 5:9
tm_stats = [Inertia(transform=:log1p), IndegreeReceiver(transform=:log1p),
            FourCycleEffect(transform=:log1p)]
edits = simulate_events(tm_stats, [0.7, 0.3, 0.2], 9, 300;
                        riskset=two_mode_dyads(users, pages), rng=Xoshiro(2))
fit_revel(edits, tm_stats, 9; riskset=two_mode_dyads(users, pages))
```

### Undirected events

```julia
pair_stats = [Inertia(symmetric=true, transform=:log1p), SharedPartners(transform=:log1p)]
fit_revel(events, pair_stats, 6; directed=false)
```

The risk set holds unordered pairs; build the statistics on symmetric layers.

## Some events as cases

`cases` restricts the likelihood to some events while all of them build the
history. It is what [`fit_stratified`](@ref) and [`fit_moving_window`](@ref)
use, and how a model is held out for prediction:

```julia
train = fit_revel(events, stats, 6; cases=1:300)
prediction_summary(train; cases=301:400).recall     # out of sample
```

## Large networks

The full risk set has `n(n−1)` dyads per event. Two routes scale further:

```julia
sampled = fit_revel(events, stats, 6; n_controls=10, rng=Xoshiro(3))
```

keeps the case and ten sampled controls per event (nested case-control
sampling; the inverse Hessian of the sampled likelihood is a consistent variance
estimator). For very large networks, pass the same statistics to `REM.fit_rem`,
which streams the design.

## Tied event times

An ordinal likelihood is a likelihood over the order of events, and a tie says
the order is unobserved. The default refuses; the policies are the ecosystem's
shared vocabulary:

| `ties` | Meaning |
|---|---|
| `:error` | name the tie and refuse (default) |
| `:ordered` | sequence order, no correction |
| `:breslow` | history frozen across the tie; one denominator |
| `:efron` | as Breslow, with the `1 − (j−1)/d` weights |

`Networks.tie_method(fit)` reports what was actually done.

## Waiting times

```julia
timed = simulate_events(stats, [0.8, 0.6], 6, 400; baseline=0.05, rng=Xoshiro(4))
tfit = fit_revel(timed, stats, 6; model=:timing)
coefnames(tfit)        # ["log_baseline", "log1p(inertia)", "log1p(reciprocity)"]
```

The exact-time likelihood multiplies a hazard by a waiting time, so every
statistic must be constant between events: [`FullMemory`](@ref) layers, static
covariates, ranks and participation shifts. A decaying memory, [`TimeSince`](@ref)
and time-varying covariates are refused with an explanation. It needs the full
directed risk set.

## Estimating memory

```julia
decaying = [Inertia(memory=HalfLife(5.0), transform=:log1p)]
seq = simulate_events(decaying, [1.5], 6, 400; rng=Xoshiro(5))
profile_memory(seq, 6, [1.0, 5.0, 25.0, 125.0]) do h
    [Inertia(memory=HalfLife(h), transform=:log1p)]
end
```

## The design itself

[`event_design`](@ref) returns the stratified design as a `DataFrame` — the
frame to take to a mixed model, a GAM or a penalised regression, none of which
Revel fits:

```julia
design = event_design(events, stats, 6; n_controls=5, rng=Xoshiro(6))
first(design, 3)
```

and [`each_risk_set`](@ref) streams the risk sets without materialising them.

## When a fit does not converge

An unconverged fit is loud: a warning, `fit.fit.converged == false`, and an
entry in `Networks.approximations(fit)`. Three causes are worth telling apart:

- **Collinearity** — [`statistic_collinearity`](@ref) before fitting.
- **Separation** — a statistic that perfectly predicts the events; the estimator
  names it.
- **An overshooting first step.** The shared Newton optimizer starts at zero and
  halves a step at most ten times. A statistic that is a rare indicator with a
  large true coefficient — a participation shift in conversational data — can
  need more, in which case the fit stops after two iterations with coefficients
  that are plainly not estimates. Dropping a near-duplicate term (an immediate
  reply and a recency rank measure much the same thing) usually resolves it.
