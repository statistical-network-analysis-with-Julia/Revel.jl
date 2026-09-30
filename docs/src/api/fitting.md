# Fitting API Reference

```@meta
CurrentModule = Revel
```

## Risk sets and designs

```@docs
RiskSetView
each_risk_set
event_design
two_mode_dyads
```

## Fitting

```@docs
fit_revel
revel
fit_receiver_choice
RevelFit
```

## Moderation by refitting

```@docs
fit_stratified
fit_moving_window
compare_coefficients
profile_memory
```

## Simulation

```@docs
simulate_events
```

## Result accessors

A [`RevelFit`](@ref) forwards the StatsAPI verbs to the underlying estimator's
result, and the ecosystem's result-metadata protocol
(`Networks.fit_metadata(fit)`: estimand, objective, exactness, standard-error
method, tie method, approximations) likewise.

```@docs
coef(::RevelFit)
stderror(::RevelFit)
vcov(::RevelFit)
confint(::RevelFit)
loglikelihood(::RevelFit)
nobs(::RevelFit)
dof(::RevelFit)
aic(::RevelFit)
bic(::RevelFit)
coeftable(::RevelFit)
coefnames(::RevelFit)
```
