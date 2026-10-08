# Fitting

[`fit_revel`](@ref) fits a relational event model with Revel's statistics. It
routes the model to the estimator for its likelihood — every one of them
maximised by the ecosystem's shared Newton optimizer, `NetworkCore.newton_fit` —
and keeps what is needed to rebuild the risk sets afterwards.

| Model | Estimator (`engine`) |
|---|---|
| ordinal, full directed risk set, every event a case, `se=:hessian` | [`Revel.fit_obpm`](@ref) (`:stream`) — evaluates the full risk set interval by interval, and can stream it instead of holding the design |
| ordinal, anything else | [`event_design`](@ref), then `REM.fit_rem` (`:design`) |
| interval timing | [`Revel.fit_timing`](@ref) (`:stream`) |

Both ordinal routes maximise the exact likelihood of the risk sets they are
given; neither approximates. The two full-risk-set likelihoods are those of
`relevent::rem.dyad` (Butts 2008), pinned against relevent 1.2.1 by a golden
fixture.

```@example fitting
using Revel, Random

stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 400; rng=Xoshiro(1))
fit = fit_revel(events, stats, 6)
```

The two ordinal routes maximise the same likelihood and agree to optimizer
tolerance:

```@example fitting
coef(fit) ≈ coef(fit_revel(events, stats, 6; engine=:design))    # true
```

## The risk set is the estimand

The likelihood is conditional on the risk set, so the actor universe is an
argument (`n_actors`), never inferred from who happened to act.

```@example fitting
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

```@example fitting
choice = fit_receiver_choice(events, stats, 6)
coeftable(choice)
```

The printed table comes from `REM.fit_rem`, whose note on the standard errors
says "full risk set": here that means the full choice set of each sender, which
is the risk set of this model. The sender-rate step (who acts next, and when)
is not implemented.

### Two-mode events

```@example fitting
users, pages = 1:4, 5:9
tm_stats = [Inertia(transform=:log1p), IndegreeReceiver(transform=:log1p),
            FourCycleEffect(transform=:log1p)]
edits = simulate_events(tm_stats, [0.7, 0.3, 0.2], 9, 300;
                        riskset=two_mode_dyads(users, pages), rng=Xoshiro(2))
fit_revel(edits, tm_stats, 9; riskset=two_mode_dyads(users, pages))
```

### Undirected events

```@example fitting
pair_stats = [Inertia(symmetric=true, transform=:log1p), SharedPartners(transform=:log1p)]
fit_revel(events, pair_stats, 6; directed=false)
```

The risk set holds unordered pairs, and every event is rewritten as the pair
`(min, max)` before the history is built. The events here were simulated as
directed; `directed=false` discards their direction, which is what an
undirected model of directed data does. Build the statistics on symmetric
layers: a directed statistic would read the pair in ID order, which carries no
meaning.

## Some events as cases

`cases` restricts the likelihood to some events while all of them build the
history. It is what [`fit_stratified`](@ref) and [`fit_moving_window`](@ref)
use, and how a model is held out for prediction:

```@example fitting
train = fit_revel(events, stats, 6; cases=1:300)
prediction_summary(train; cases=301:400).recall     # out of sample
```

When the selection depends on **who acts** — the events sent by one group,
say — the likelihood must be conditional on the event being a case: given that
the next event is one the selection admits, which of the dyads that could have
produced such an event acted? Pass the selection as a predicate and each
case's risk set is restricted to the dyads whose candidate event the predicate
accepts (against the full risk set, a sender covariate would "explain" the
selection itself):

```@example fitting
group = [1, 1, 1, 2, 2, 2]
senders_2 = fit_revel(events, stats, 6; cases = e -> group[e.sender] == 2)
senders_2.riskset isa Function                      # the restricted risk set
```

A predicate on the time or the event type keeps the risk set. Indices and
masks cannot be inspected and are taken to be chosen without regard to which
dyad acted; pass a predicate when they are not.

## Large networks

The full risk set has `n(n−1)` dyads per event. Two routes scale further:

```@example fitting
sampled = fit_revel(events, stats, 6; n_controls=10, rng=Xoshiro(3))
```

keeps the case and ten sampled controls per event (nested case-control
sampling; the inverse Hessian of the sampled likelihood is a consistent variance
estimator). The statistics are evaluated on the sampled dyads only, so the cost
is `events × (n_controls + 1)` evaluations however large the risk set, and a
layer's memory grows with the dyads that have a history, not with `n²`. The
diagnostics that need the score of the full likelihood ([`score_test`](@ref),
[`score_process_test`](@ref), [`statistic_collinearity`](@ref) of a fit) are
refused for a sampled fit.

On the full risk set the exact route is [`Revel.fit_obpm`](@ref), which by
default keeps the design matrices in memory up to a quarter of the free memory
and recomputes the statistics on every pass of the optimizer beyond that; pass
`cache=:all` or a larger `cache_bytes` when memory allows, or `cache=:none` to
keep a single design matrix alive. Every policy gives the bit-identical fit:

```@example fitting
coef(fit_revel(events, stats, 6; cache=:none)) == coef(fit)     # true
```

## Tied event times

An ordinal likelihood is a likelihood over the order of events, and a tie says
the order is unobserved. The default refuses; the policies are the ecosystem's
shared vocabulary:

| `ties` | Meaning |
|---|---|
| `:error` | name the tie and refuse (default) |
| `:ordered` | sequence order, no correction |
| `:breslow` | history frozen across the tie; one denominator. The one to use with sampled controls |
| `:efron` | as Breslow, with the `1 − (j−1)/d` weights; full risk set only |

With sampled controls (`n_controls`) `:efron` is refused: the other tied cases
would have to stay in every stratum with probability one while the controls
are sampled, and the tied cases — the dyads that just acted — then dominate the
sampled denominator, biasing the estimate toward zero (0.65 for a truth of 1.0
at 10 controls). The refusal is REM.jl's `REM.check_tie_sampling`, the same
rule `REM.fit_rem` applies. `:breslow` is unbiased with sampled controls; Efron
remains available on the full risk set.

`NetworkCore.tie_method(fit)` reports what was actually done. With a subset of
cases under `:efron`, every event of a tie block stays in the denominator —
including tied events that are not cases, which are part of the risk set — and
`j` is the event's position in its tie block, counting the events that are not
cases as well: the weights are those of the full likelihood, restricted to the
cases.

## Waiting times

```@example fitting
timed = simulate_events(stats, [0.8, 0.6], 6, 400; baseline=0.05, rng=Xoshiro(4))
tfit = fit_revel(timed, stats, 6; model=:timing)
coefnames(tfit)        # ["log_baseline", "log1p(inertia)", "log1p(reciprocity)"]
```

The interval-timing model gives dyad `(i, j)` the hazard `λ₀ exp(θ'x_ij)`
between events, with an exponential baseline, and its likelihood is

```math
\ell(\lambda_0, \theta) = \sum_m \Big[\log\lambda_0 + \theta^\top x_{\text{case}} -
  \lambda_0\, \Delta t_m \sum_{ij} \exp(\theta^\top x_{ij})\Big].
```

It multiplies a hazard by a waiting time, so every statistic must be constant
between events ([`Revel.is_interval_constant`](@ref)): [`FullMemory`](@ref)
layers, any layer on the event clock (`clock=:order`), static covariates, ranks
and participation shifts. A decaying memory on the time clock,
[`TimeSince`](@ref) on the time clock, [`GlobalEffect`](@ref) and time-varying
covariates are refused with an explanation: evaluating a decaying statistic at
the next event and multiplying it by the waiting time is not its integral. It
needs the full directed risk set, and only the exponential baseline is
implemented.

Two keywords set the observation window. `t0` is its start (default `0`, as in
relevent, whose "event times should be relative to onset of observation"): the
first waiting time is `t₁ − t0`. `t_end` is the time recording stopped; it adds
the right-censored final interval `[t_M, t_end]`, exposure with no event.
`relevent::rem.dyad` always has that interval (the last row of its edgelist is
the end of observation), so pass `t_end` to reproduce it; without it the
sequence ends at its last event, and λ₀ is biased upward.

```@example fitting
window = fit_revel(timed, stats, 6; model=:timing, t0=0.0, t_end=timed[end].time + 1.0)
nobs(window) == nobs(tfit)                       # true — the tail is exposure, not an event
loglikelihood(window) < loglikelihood(tfit)      # true — and it has a likelihood of its own
```

Ties mean something different here. Under a continuous-time process two events
at one instant have probability zero, so a tie is the model's assumption
failing rather than an ambiguous order, and the policies differ from the
ordinal ones:

| `ties` | ordinal | timing |
|---|---|---|
| `:error` | default | default |
| `:ordered` | sequence order, no correction | tied events after the first get a zero-length waiting interval |
| `:breslow`, `:efron` | the Cox corrections | **refused**: they correct a partial likelihood, which this is not |
| `:batch` | **refused**: with the history frozen it *is* Breslow | one simultaneous batch: history frozen, one exposure interval |

`NetworkCore.is_exact(fit)` is `false` for a timing fit as soon as the data
carried a tie, under every policy.

## Estimating memory

```@example fitting
decaying = [Inertia(memory=HalfLife(5.0), transform=:log1p)]
seq = simulate_events(decaying, [1.5], 6, 400; rng=Xoshiro(5))
profile = profile_memory(seq, 6, [1.0, 2.5, 5.0, 10.0, 25.0, 125.0]) do h
    [Inertia(memory=HalfLife(h), transform=:log1p)]
end
profile.value[profile.in_ci]     # the half-lives inside the 95 % profile interval
```

The profile marks the maximum (`best`) and the grid values inside the
profile-likelihood confidence set (`in_ci`), and its `aic`/`bic` count the
memory parameter, so they compare with a model without one. The standard errors
of the other coefficients treat the memory as known.

## The design itself

[`event_design`](@ref) returns the stratified design as a `DataFrame` — the
frame to take to a mixed model, a GAM or a penalised regression, none of which
Revel fits:

```@example fitting
design = event_design(events, stats, 6; n_controls=5, rng=Xoshiro(6))
first(design, 3)
```

and [`each_risk_set`](@ref) streams the risk sets without materialising them.

## When a fit does not converge

An unconverged fit is loud: a warning, `fit.fit.converged == false`, and an
entry in `NetworkCore.approximations(fit)`. Three causes are worth telling apart:

- **Collinearity** — [`statistic_collinearity`](@ref) before fitting.
- **Separation** — a statistic that perfectly predicts the events. Every route
  decides it on the risk sets it fits, before the optimizer runs, with the
  ecosystem's one verdict (`NetworkCore.separation_from_margins`, certified in
  exact arithmetic), and follows the one policy: a warning, `converged ==
  false`, the separated coefficients named in `fit.fit.separated` (with
  `"log_baseline"` among them when a timing fit's baseline runs away), and z
  values, p-values and confidence intervals withheld (`NaN`).
- **An overshooting first step.** The shared Newton optimizer starts at zero
  and halves an overshooting step up to thirty times (since NetworkCore.jl commit
  `03aaa03`; earlier checkouts allowed ten, which stopped some fits with a rare
  indicator statistic after two iterations). If a fit stops after two iterations
  with implausible coefficients, update NetworkCore.jl.

The diagnostics refuse a fit that did not converge: they are defined at the
maximum of the likelihood.
