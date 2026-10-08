# =============================================================================
# Fitting
# =============================================================================
#
# Revel hosts no optimizer of its own (the ecosystem's shared-numerics rule:
# every likelihood is maximised by NetworkCore.newton_fit). `fit_revel` routes a
# model to the estimator for its likelihood:
#
#   ordinal, full directed risk set, every event a case
#       → fit_obpm   (src/engine/: exact, streamed risk sets)
#   ordinal, anything else (a restricted or sampled risk set, a subset of
#   cases, undirected events, sandwich standard errors)
#       → the design frame, then REM.fit_rem(::DataFrame, names)
#   interval timing
#       → fit_timing (src/engine/)
#
# and wraps the result together with what is needed to rebuild its risk sets, so
# that the goodness-of-fit diagnostics can be computed from the fit alone.

"""
    RevelFit

A fitted relational event model: the underlying estimator's result (`fit.fit` —
a [`Revel.OrdinalBPMResult`](@ref), a [`Revel.TimingModelResult`](@ref) or a
`REM.REMResult`) together with the time-sorted events, the statistics and the
risk-set specification that produced it.

It answers the StatsAPI verbs (`coef`, `stderror`, `vcov`, `confint`,
`loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `coeftable`, `coefnames`) and the
ecosystem's result-metadata protocol (`NetworkCore.fit_metadata`), and it is what
[`event_diagnostics`](@ref), [`prediction_summary`](@ref),
[`score_process_test`](@ref), [`score_test`](@ref) and `gof` take.

# Example
```julia
using Revel, Random
events = simulate_events([Inertia(transform=:log1p)], [1.0], 5, 60; rng=Xoshiro(1))
fit = fit_revel(events, [Inertia(transform=:log1p), Reciprocation(transform=:log1p)], 5)
fit isa RevelFit             # true
coefnames(fit)               # ["log1p(inertia)", "log1p(reciprocity)"]
length(coef(fit)), nobs(fit) # (2, 60)
```
"""
struct RevelFit{F, T, R}
    fit::F
    events::Vector{Event{T}}
    statistics::Vector{AbstractStatistic}
    n_actors::Int
    model::Symbol
    engine::Symbol
    directed::Bool
    riskset::R                               # concretely typed: no dynamic dispatch
    cases::Union{Nothing, BitVector}
    ties::Symbol
    n_controls::Union{Nothing, Int}
    t0::Union{Nothing, T}                    # timing model: start of the clock
    t_end::Union{Nothing, T}                 # timing model: end of observation
end

const _FIT_MODELS = (:ordinal, :timing)
const _FIT_ENGINES = (:auto, :stream, :design)

_riskset_label(rs::Symbol) = String(rs)
_riskset_label(rs::AbstractVector) = "$(length(rs)) listed dyads"
_riskset_label(rs::_CaseRestricted) =
    "the dyads the `cases` predicate admits (of $(_riskset_label(rs.base)))"
_riskset_label(::Function) = "time-varying"
_riskset_label(rs) = string(rs)

Base.show(io::IO, fit::RevelFit) =
    print(io, "RevelFit(", fit.model, ", ", length(fit.events), " events, ",
          length(fit.statistics), " statistic", length(fit.statistics) == 1 ? "" : "s", ")")

function Base.show(io::IO, ::MIME"text/plain", fit::RevelFit)
    println(io, "Revel relational event model")
    println(io, "  model:     ", fit.model === :ordinal ? "ordinal (event order)" :
                                 "interval timing (exponential baseline)")
    println(io, "  events:    ", length(fit.events),
            fit.cases === nothing ? "" : " ($(count(fit.cases)) modelled as cases)",
            fit.directed ? ", directed" : ", undirected")
    println(io, "  actors:    ", fit.n_actors)
    println(io, "  risk set:  ", _riskset_label(fit.riskset),
            fit.n_controls === nothing ? "" : " ($(fit.n_controls) sampled controls)")
    println(io, "  estimator: ", fit.engine === :stream ?
            (fit.model === :ordinal ? "Revel.fit_obpm (full risk set, streamed)" :
                                      "Revel.fit_timing (full risk set, streamed)") :
            "REM.fit_rem on the event design")
    println(io)
    show(io, fit.fit)
end

"""
    fit_revel(events, statistics, n_actors; model=:ordinal, directed=true,
              riskset=:full, cases=nothing, ties=:error, n_controls=nothing,
              rng=Random.default_rng(), se=:hessian, engine=:auto,
              maxiter=100, tol=1e-8, kwargs...) -> RevelFit

Fit a relational event model with Revel's statistics (or any statistic with the
history interface `compute(stat, ::InteractionHistory, sender, receiver, time)`;
a REM.jl statistic, which has only REM's interface, is refused with an
`ArgumentError` — fit it with `REM.fit_rem`). [`revel`](@ref) is an alias.

`model=:ordinal` (default) models which dyad acts next given the history — the
ordinal likelihood of `relevent::rem.dyad`, a conditional logit with one stratum
per event. `model=:timing` adds the waiting times under an exponential baseline
([`Revel.fit_timing`](@ref); pass `t0`/`t_end` through `kwargs`); it needs the full
directed risk set and statistics that are constant between events:
[`FullMemory`](@ref) layers and any layer with `clock=:order`, static
covariates, recency ranks and participation shifts qualify; a decaying memory
on the time clock, [`TimeSince`](@ref), [`GlobalEffect`](@ref) and time-varying
covariates are refused.

For the ordinal model:

- `directed`, `riskset`, `cases` and `ties` are as in [`each_risk_set`](@ref).
  `riskset=:sender` fits the receiver-choice step of an actor-oriented model
  (see [`fit_receiver_choice`](@ref)); a vector from
  [`two_mode_dyads`](@ref) fits a two-mode model; `cases` restricts the
  likelihood to some events while the rest still build the history. A `cases`
  predicate that depends on who acts (`e -> group[e.sender] == 2`) also
  restricts each case's risk set to the dyads that would have produced a case
  — the conditional likelihood of "which dyad, given that the event is a
  case" — and that restricted risk set is what the fit stores and shows.
- `n_controls` samples that many controls per event (nested case-control
  sampling) instead of enumerating the risk set, with all randomness from `rng`.
- `se` — `:hessian` or `:sandwich` (`REM.fit_rem`'s event-clustered sandwich.
  Each event is its own cluster, so it guards against a misspecified
  functional form but not against dependence between the score contributions
  of successive events).
- `engine` — `:stream` uses [`Revel.fit_obpm`](@ref), which evaluates the full
  risk set interval by interval and can stream it instead of holding the design
  in memory; `:design` builds the [`event_design`](@ref) frame and fits it with
  `REM.fit_rem`. `:auto` picks `:stream` whenever the model is one it can fit
  (full directed risk set, every event a case, no sampling, `se=:hessian`) and
  `:design` otherwise. Both maximise the same likelihood exactly; on a common
  model they agree to optimizer tolerance.

Remaining keywords are forwarded to the underlying fitter (`cache`, `chunk`,
`cache_bytes` for [`Revel.fit_obpm`](@ref)/[`Revel.fit_timing`](@ref); `t0`,
`t_end` for the timing model). The streamed engine keeps the design matrices in
memory up to `cache_bytes` and otherwise recomputes the statistics on every pass
of the optimizer; `fit_revel`'s default `cache_bytes` is a quarter of the free
memory (between 256 MiB and 8 GiB), because the statistics are costly to
recompute.

**Separation.** When no finite maximum exists (a statistic that every case
maximises within its risk set, say), every route follows the ecosystem's
separation policy, decided by the shared verdict of NetworkCore.jl on the risk
sets actually fitted: a warning, `converged == false`, the separated
coefficients named in `fit.fit.separated`, and z values, p-values and
[`confint`](@ref) withheld (`NaN`). The diagnostics refuse such a fit.

Statistic names must be unique; pass `name=` to tell apart two effects that
differ only in a filter. A [`GlobalEffect`](@ref) outside an
[`Interaction`](@ref) is refused under the ordinal model, which cannot identify
it. The statistics are evaluated on private copies (see [`EventLayer`](@ref)),
so one specification can be fitted from several tasks at once.

# Example
```julia
using Revel, Random
truth = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(truth, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, truth, 6)
round.(coef(fit); digits=1)                      # [0.6, 0.5] — the truth is [0.8, 0.6]
round.(stderror(fit); digits=2)                  # [0.13, 0.13]
choice = fit_revel(events, truth, 6; riskset=:sender)   # receiver choice
nobs(choice)                                     # 200
```
"""
fit_revel(events::AbstractVector{<:Event}, statistics::AbstractVector{<:AbstractStatistic},
          n_actors::Int; kwargs...) = _fit_revel(collect(events), statistics, n_actors; kwargs...)

_default_cache_bytes() = clamp(Int(Sys.free_memory() ÷ 4), 1 << 28, 1 << 33)

# Every statistic must have the history interface
function _check_history_interface(stats, ::Type{T}) where T
    for stat in stats
        hasmethod(compute, Tuple{typeof(stat), InteractionHistory{T}, Int, Int, T}) &&
            continue
        throw(ArgumentError(
            "$(name(stat)) ($(typeof(stat))) has no method " *
            "`compute(stat, ::InteractionHistory, sender, receiver, time)`: it is a " *
            "REM.jl statistic, which reads REM's `EventNetworkState` only. Use the " *
            "Revel counterpart (see `effect_catalogue()`), or fit the model with " *
            "`REM.fit_rem`, where Revel's statistics work as well."))
    end
    return nothing
end

# A global covariate is constant within every risk set: the ordinal likelihood
# does not depend on its coefficient
function _check_global_main_effects(stats)
    for stat in stats
        stat isa GlobalEffect && throw(ArgumentError(
            "$(name(stat)) is a GlobalEffect, which takes the same value for every " *
            "dyad of a risk set, so the ordinal likelihood does not depend on its " *
            "coefficient. Use it only inside an Interaction (e.g. " *
            "`Interaction(GlobalEffect(...), Inertia())`), which lets the effect " *
            "of another statistic change with it."))
    end
    return nothing
end

function _fit_revel(events::Vector{Event{T}},
                    statistics::AbstractVector{<:AbstractStatistic}, n_actors::Int;
                   model::Symbol=:ordinal, directed::Bool=true, riskset=:full,
                   cases=nothing, ties::Symbol=:error,
                   n_controls::Union{Nothing,Int}=nothing,
                   rng::AbstractRNG=Random.default_rng(), se::Symbol=:hessian,
                   engine::Symbol=:auto, maxiter::Int=100, tol::Float64=1e-8,
                   kwargs...) where T
    model in _FIT_MODELS || throw(ArgumentError(
        "model must be :ordinal or :timing, got :$model"))
    engine in _FIT_ENGINES || throw(ArgumentError(
        "engine must be :auto, :stream or :design, got :$engine"))
    isempty(events) && throw(ArgumentError("no events to fit"))
    names = _stat_names(statistics)
    stats = collect(AbstractStatistic, statistics)
    _check_history_interface(stats, T)
    model === :ordinal && _check_global_main_effects(stats)
    _check_events(events)
    sorted = sort(events; by=e -> e.time)
    # A `cases` predicate that depends on who acts restricts each case's risk
    # set to the dyads that would have produced a case (see `each_risk_set`). The restriction becomes the fit's risk set, so the
    # diagnostics rebuild the same strata. The predicate is read on the events
    # as the design sees them (an undirected pair as (min, max)).
    if cases isa Function
        seen = directed ? sorted :
               [Event(minmax(e.sender, e.receiver)..., e.time; eventtype=e.eventtype,
                      weight=e.weight) for e in sorted]
        riskset = _case_restricted_riskset(cases, riskset, n_actors, directed, seen)
        mask = _case_mask(cases, seen)
    else
        mask = cases === nothing ? nothing : _case_mask(cases, sorted)
    end
    mask === nothing || any(mask) || throw(ArgumentError(
        "`cases` selects no event; there is nothing to fit"))

    plain = directed && riskset === :full && cases === nothing && n_controls === nothing

    if model === :timing
        plain || throw(ArgumentError(
            "model=:timing is the exact-time likelihood of `Revel.fit_timing`, " *
            "which enumerates the full directed risk set and treats every event as " *
            "a case. A restricted or sampled risk set, undirected events and a " *
            "subset of cases are available for model=:ordinal only."))
        se === :hessian || throw(ArgumentError(
            "model=:timing offers se=:hessian only"))
        engine === :design && throw(ArgumentError(
            "model=:timing is fitted by Revel.fit_timing; engine=:design is for " *
            "the ordinal model"))
        _check_standardized(stats, :full, n_actors, true)
        inner = fit_timing(sorted, _fresh(stats), n_actors; ties=ties,
                           maxiter=maxiter, tol=tol,
                           cache_bytes=_default_cache_bytes(), kwargs...)
        t0 = convert(T, get(kwargs, :t0, zero(T)))
        t_end = get(kwargs, :t_end, nothing)
        return RevelFit{typeof(inner), T, Symbol}(inner, sorted, stats, n_actors, :timing,
                                          :stream, true, :full, nothing, ties, nothing,
                                          t0, t_end === nothing ? nothing : convert(T, t_end))
    end

    use_stream = engine === :stream || (engine === :auto && plain && se === :hessian)
    if use_stream
        plain && se === :hessian || throw(ArgumentError(
            "engine=:stream (Revel.fit_obpm) fits the full directed risk set " *
            "with every event a case and se=:hessian; use engine=:design (or " *
            ":auto) for a restricted or sampled risk set, undirected events, a " *
            "subset of cases or se=:sandwich"))
        _check_standardized(stats, :full, n_actors, true)
        inner = fit_obpm(sorted, _fresh(stats), n_actors; ties=ties,
                         maxiter=maxiter, tol=tol,
                         cache_bytes=_default_cache_bytes(), kwargs...)
        return RevelFit{typeof(inner), T, Symbol}(inner, sorted, stats, n_actors, :ordinal,
                                          :stream, true, :full, nothing, ties, nothing,
                                          nothing, nothing)
    end

    isempty(kwargs) || throw(ArgumentError(
        "unknown keyword$(length(kwargs) == 1 ? "" : "s") for the design engine: " *
        join((":" * string(k) for k in keys(kwargs)), ", ")))
    design = event_design(sorted, stats, n_actors; directed=directed, riskset=riskset,
                          ties=ties, cases=mask, n_controls=n_controls, rng=rng)
    inner = REM.fit_rem(design, names; maxiter=maxiter, tol=tol, se=se)
    return RevelFit{typeof(inner), T, typeof(riskset)}(inner, sorted, stats, n_actors,
                                                       :ordinal, :design, directed, riskset,
                                                       mask, ties, n_controls, nothing, nothing)
end

"""
    revel(events, statistics, n_actors; kwargs...) -> RevelFit

Alias for [`fit_revel`](@ref) (the ecosystem's convention: every model answers
to a `fit_<model>` name and a short one).

# Example
```julia
using Revel, Random
events = simulate_events([Inertia(transform=:log1p)], [1.0], 5, 80; rng=Xoshiro(2))
revel === fit_revel                                           # true
fit = revel(events, [Inertia(transform=:log1p)], 5)
coef(fit)[1] > 0                                              # true
```
"""
const revel = fit_revel

"""
    fit_receiver_choice(events, statistics, n_actors; kwargs...) -> RevelFit

The receiver-choice step of an actor-oriented event model: given the sender of
each event, which receiver did it choose among the other `n−1` actors? This is
the multinomial choice model of DyNAM (Stadtfeld & Block 2017; goldfish
`subModel = "choice"`) and of remstats' actor-oriented model. It is
`fit_revel(…; riskset=:sender)`.

A statistic that depends on the sender alone — [`SendEffect`](@ref),
[`OutdegreeSender`](@ref) — is constant within every choice set and cannot be
estimated here (the fit does not converge, and [`statistic_collinearity`](@ref)
reports it with an infinite VIF); it can enter only as a moderator, through an
[`Interaction`](@ref).

Stadtfeld & Block (2017, p. 340) report estimates of opposite sign for the same
effect ("outdegree × friendship") in a tie-oriented and an actor-oriented model
of the same data, and explain the difference by what the other effects of each
model (transitivity of friendship) absorb. Fitting both is therefore a
specification check. A reason of the same kind, which is ours rather than
theirs: a tie-oriented model carries the sender's activity in its intercept of
dyads, so a sender-level term there mixes "active senders" with "senders who
repeat".

The sender-rate step (who acts next, and when) is not implemented.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(3))
choice = fit_receiver_choice(events, stats, 6)
choice.riskset                 # :sender
length(coef(choice))           # 2
```
"""
function fit_receiver_choice(events, statistics, n_actors::Int; kwargs...)
    haskey(kwargs, :riskset) && throw(ArgumentError(
        "fit_receiver_choice fixes the risk set to the observed sender's possible " *
        "receivers (riskset=:sender); call fit_revel for any other risk set"))
    return fit_revel(events, statistics, n_actors; riskset=:sender, kwargs...)
end

# -----------------------------------------------------------------------------
# StatsAPI and the result-metadata protocol: forwarded to the underlying fit
# -----------------------------------------------------------------------------

"""
    coef(fit::RevelFit) -> Vector{Float64}

The estimated coefficients, in [`coefnames`](@ref) order (StatsAPI). An
interval-timing fit puts the log baseline rate first.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
round.(coef(fit); digits=1)          # [0.6, 0.5] — the truth is [0.8, 0.6]
```
"""
coef(fit::RevelFit) = coef(fit.fit)

"""
    stderror(fit::RevelFit) -> Vector{Float64}

Standard errors of the coefficients (StatsAPI): the square roots of the
diagonal of [`vcov`](@ref), under the estimator `NetworkCore.se_method(fit)` names.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
all(0 .< stderror(fit) .< 0.2)       # true
```
"""
stderror(fit::RevelFit) = stderror(fit.fit)

"""
    vcov(fit::RevelFit) -> Matrix{Float64}

The covariance matrix of the coefficients (StatsAPI): the inverse observed
information, or the event-clustered sandwich for a fit with `se=:sandwich`.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
size(vcov(fit))                      # (2, 2)
sqrt.([vcov(fit)[1, 1], vcov(fit)[2, 2]]) ≈ stderror(fit)   # true
```
"""
vcov(fit::RevelFit) = vcov(fit.fit)

"""
    confint(fit::RevelFit; level=0.95) -> Matrix{Float64}

Wald confidence intervals, one row per coefficient (StatsAPI).

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
ci = confint(fit; level=0.9)
all(ci[:, 1] .< coef(fit) .< ci[:, 2])    # true
```
"""
confint(fit::RevelFit; kwargs...) = confint(fit.fit; kwargs...)

"""
    loglikelihood(fit::RevelFit) -> Float64

The maximised log-likelihood (StatsAPI): the ordinal partial likelihood of the
risk sets the fit used, or the interval-timing likelihood. Likelihoods of fits on
different risk sets, or of the two kinds of model, are not comparable.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
loglikelihood(fit) < 0                # true
-2loglikelihood(fit) ≈ prediction_summary(fit).deviance    # true
```
"""
loglikelihood(fit::RevelFit) = loglikelihood(fit.fit)

"""
    nobs(fit::RevelFit) -> Int

The number of events the likelihood was built from (StatsAPI) — the events
selected by `cases`, not the rows of the design.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
nobs(fit)                                   # 200
nobs(fit_revel(events, stats, 6; cases=101:200))   # 100
```
"""
nobs(fit::RevelFit) = nobs(fit.fit)

"""
    dof(fit::RevelFit) -> Int

The number of estimated coefficients (StatsAPI), including the log baseline of
an interval-timing fit.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
dof(fit)                                           # 2
dof(fit_revel(events, stats, 6; model=:timing))    # 3
```
"""
dof(fit::RevelFit) = dof(fit.fit)

"""
    aic(fit::RevelFit) -> Float64

Akaike's information criterion, `-2 loglikelihood + 2 dof` (StatsAPI). Compare
only fits of the same events on the same risk sets.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
aic(fit) ≈ -2loglikelihood(fit) + 2dof(fit)                  # true
aic(fit) < aic(fit_revel(events, stats[1:1], 6))             # reciprocity earns its place
```
"""
aic(fit::RevelFit) = aic(fit.fit)

"""
    bic(fit::RevelFit) -> Float64

The Bayesian information criterion, `-2 loglikelihood + log(nobs) dof`
(StatsAPI), with the number of events as the sample size.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
bic(fit) ≈ -2loglikelihood(fit) + log(nobs(fit)) * dof(fit)  # true
```
"""
bic(fit::RevelFit) = bic(fit.fit)

"""
    coeftable(fit::RevelFit) -> NetworkCore.CoefficientTable

The coefficient table — estimates, standard errors, z- and p-values — as the
ecosystem's shared `NetworkCore.CoefficientTable` (StatsAPI's `coeftable`).

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
table = coeftable(fit)
table isa Revel.NetworkCore.CoefficientTable    # true
```
"""
coeftable(fit::RevelFit) = coeftable(fit.fit)

"""
    coefnames(fit::RevelFit) -> Vector{String}

The coefficient names in `coef(fit)` order: the statistic names, preceded by
`"log_baseline"` for an interval-timing fit.

# Example
```julia
using Revel, Random
events = simulate_events([Inertia(transform=:log1p)], [1.0], 5, 60; rng=Xoshiro(4))
coefnames(fit_revel(events, [Inertia(transform=:log1p), OTP()], 5))   # ["log1p(inertia)", "otp"]
```
"""
coefnames(fit::RevelFit) =
    fit.model === :timing ? ["log_baseline"; [name(s) for s in fit.statistics]] :
                            [name(s) for s in fit.statistics]

estimand(fit::RevelFit) = estimand(fit.fit)
objective(fit::RevelFit) = objective(fit.fit)
is_exact(fit::RevelFit) = is_exact(fit.fit)
se_method(fit::RevelFit) = se_method(fit.fit)
missing_method(fit::RevelFit) = missing_method(fit.fit)
tie_method(fit::RevelFit) = tie_method(fit.fit)
approximations(fit::RevelFit) = approximations(fit.fit)

# The effect coefficients of the ordinal (choice) part — without the baseline
_effect_coef(fit::RevelFit) = fit.model === :timing ? coef(fit)[2:end] : coef(fit)

# -----------------------------------------------------------------------------
# Moderation by refitting: strata and moving windows
# -----------------------------------------------------------------------------

"""
    fit_stratified(events, statistics, n_actors; by, kwargs...) -> Dict

Fit the model separately within strata of the **events being explained**, while
every event still builds the history: `by` is a function `event -> key`, and the
result maps each key to a [`RevelFit`](@ref) whose cases are the events with
that key.

This is the stratified form of moderation — every coefficient is free to differ
by event type (`by = e -> e.eventtype`; the type-specific intensities of Vu,
Lomi, Mascia & Pallotti 2017), weekday, period or context. It equals a
product-term model only when *all* interactions with the stratifier are
included (the stratum likelihoods are disjoint factors of one partial
likelihood), and it is not the same model as a statistic filtered on the
stratifier. Because the factors are disjoint, the estimates of different strata
are asymptotically independent, and [`compare_coefficients`](@ref) with
`reference=` gives the Wald test of each difference. A difference in
significance is not a significant difference.

**The risk set of a stratum.** When the stratum of an event depends on who acts
— `by = e -> department[e.sender]` — the events of one stratum can only have
been produced by some of the dyads, and fitting them against every dyad would
let the model "explain" the stratum itself (a sender covariate then absorbs the
selection). So each event's risk set is restricted to the dyads that would have
produced an event of the same stratum: `by` is applied to the candidate event
`Event(s, r, time; eventtype, weight)` of every dyad, and the dyads whose key
differs are dropped. A stratifier that does not depend on the dyad (the time,
the event type) keeps the full risk set, as before. The restriction is applied
to `riskset=:full` or a vector of dyads; with another `riskset` a dyad-dependent
stratifier is refused.

Keywords are those of [`fit_revel`](@ref) (the ordinal model), except `cases`,
which the strata define.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p)]
events = simulate_events(stats, [1.0], 5, 120; rng=Xoshiro(5))
fits = fit_stratified(events, stats, 5; by = e -> e.time <= 60 ? :early : :late)
sort(collect(keys(fits)))          # [:early, :late]
nobs(fits[:early]) + nobs(fits[:late])    # 120
```
"""
function fit_stratified(events::AbstractVector{<:Event}, statistics, n_actors::Int; by,
                        kwargs...)
    by isa Function || throw(ArgumentError(
        "`by` must be a function event -> stratum key (e.g. `e -> e.eventtype`)"))
    haskey(kwargs, :cases) && throw(ArgumentError(
        "fit_stratified defines the cases of each fit by its stratum; pass the " *
        "events you want to stratify instead of `cases`"))
    sorted = sort(events; by=e -> e.time)
    keys_ = [by(e) for e in sorted]
    for (m, key) in enumerate(keys_)
        key === missing || key === nothing || continue
        throw(ArgumentError(
            "the stratifier returned `$key` for event $m ($(sorted[m])); every event " *
            "needs a stratum key (drop the events you do not want to stratify)"))
    end
    directed = get(kwargs, :directed, true)
    base = get(kwargs, :riskset, :full)
    candidates = base === :full ? _full_dyads(n_actors, directed) :
                 base isa AbstractVector ? [(Int(a), Int(b)) for (a, b) in base] : nothing
    candidate(s, r, e) = Event(s, r, e.time; eventtype=e.eventtype, weight=e.weight)
    # Does the stratum depend on which dyad acts? Then each stratum gets the
    # dyads that could have produced it as its risk set.
    dyad_dependent = candidates !== nothing &&
        any(any(by(candidate(s, r, e)) != keys_[m] for (s, r) in candidates)
            for (m, e) in enumerate(sorted))
    if candidates === nothing
        # Only checkable against an explicit set of dyads
        any(by(candidate(e.receiver, e.sender, e)) != by(e) for e in sorted) &&
            throw(ArgumentError(
                "the stratifier depends on the acting dyad, and the risk set of each " *
                "stratum can only be restricted for riskset=:full or a vector of " *
                "dyads; got riskset=$(repr(base))"))
    end
    fits = Dict{Any, RevelFit}()
    for key in unique(keys_)
        mask = keys_ .== Ref(key)
        if dyad_dependent
            own = (m, e) -> [(s, r) for (s, r) in candidates if by(candidate(s, r, e)) == key]
            rest = Base.structdiff(values(kwargs), NamedTuple{(:riskset,)})
            fits[key] = fit_revel(sorted, statistics, n_actors; cases=mask, riskset=own,
                                  rest...)
        else
            fits[key] = fit_revel(sorted, statistics, n_actors; cases=mask, kwargs...)
        end
    end
    return fits
end

"""
    fit_moving_window(events, statistics, n_actors; width, step=width,
                      min_events=10, kwargs...) -> Vector{NamedTuple}

Refit the model on the events of successive time windows `[from, from + width)`,
advancing by `step`, with the whole earlier sequence as history. Each element is
`(from, to, n_events, fit)`; a window with fewer than `min_events` events is
skipped.

This is the moving-window route to **time-varying coefficients** (Mulder &
Leenders 2019; Meijerink-Bosman, Leenders & Mulder 2022): it shows whether an
effect such as reciprocity strengthens or fades over the observation period
without committing to a functional form. Overlapping windows (`step < width`)
smooth the path and make neighbouring estimates dependent. Use
[`compare_coefficients`](@ref) for the path as a table; with non-overlapping
windows its `reference=` Wald tests are valid.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p)]
events = simulate_events(stats, [1.0], 5, 150; rng=Xoshiro(6))
path = fit_moving_window(events, stats, 5; width=50.0)
length(path)                       # 3
[w.n_events for w in path]         # [50, 50, 50]
```
"""
function fit_moving_window(events::AbstractVector{<:Event}, statistics, n_actors::Int;
                           width::Real, step::Real=width, min_events::Int=10,
                           kwargs...)
    width > 0 && step > 0 || throw(ArgumentError("width and step must be positive"))
    isempty(events) && throw(ArgumentError("no events to fit"))
    haskey(kwargs, :cases) && throw(ArgumentError(
        "fit_moving_window defines the cases of each fit by its window; pass the " *
        "events you want to window instead of `cases`"))
    sorted = sort(events; by=e -> e.time)
    times = [_tfloat(e.time) for e in sorted]
    t_first, t_last = first(times), last(times)
    out = NamedTuple{(:from, :to, :n_events, :fit), Tuple{Float64, Float64, Int, RevelFit}}[]
    k = 0
    from = t_first
    while from <= t_last
        to = from + width
        mask = (times .>= from) .& (times .< to)
        n = count(mask)
        if n >= min_events
            fit = fit_revel(sorted, statistics, n_actors; cases=mask, kwargs...)
            push!(out, (from=from, to=to, n_events=n, fit=fit))
        end
        k += 1
        from = t_first + k * step           # no drift from repeated addition
    end
    isempty(out) && throw(ArgumentError(
        "no window holds at least min_events=$min_events events; widen the " *
        "windows or lower min_events"))
    return out
end

"""
    compare_coefficients(fits; reference=nothing) -> DataFrame

A long table — one row per (fit, coefficient) with `group`, `term`, `estimate`,
`std_error`, `z`, `n_events` — of several fits of the same specification: the
`Dict` returned by [`fit_stratified`](@ref), the vector returned by
[`fit_moving_window`](@ref) (grouped by window start), or any collection of
`key => fit` pairs. It lays the stratified, windowed and pooled estimates side
by side, which is how the literature reads a moderation by refitting.

With `reference=key` the table also holds, for every other group, the Wald test
of the difference from the reference group's coefficient: `difference`,
`se_difference = √(se² + se_ref²)` and its two-sided `p_difference` (`missing`
in the reference rows). A fit with no finite maximum (separation) has `z`, and
every `p_difference` that involves it, `NaN`. The test assumes the fits are independent, which holds
for the strata of [`fit_stratified`](@ref) and for non-overlapping windows —
their likelihoods are disjoint factors of one partial likelihood — and is
refused for overlapping windows.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p)]
events = simulate_events(stats, [1.0], 5, 120; rng=Xoshiro(5))
fits = fit_stratified(events, stats, 5; by = e -> e.time <= 60 ? :early : :late)
table = compare_coefficients(fits)
size(table)                # (2, 6)
names(table)               # ["group", "term", "estimate", "std_error", "z", "n_events"]
tested = compare_coefficients(fits; reference=:early)
tested.p_difference[2] > 0.05     # true — one process throughout
```
"""
function compare_coefficients(fits; reference=nothing)
    pairs_ = _fit_pairs(fits)
    group = Any[]; term = String[]; est = Float64[]; se = Float64[]; z = Float64[]
    n = Int[]
    for (key, fit) in pairs_
        names = coefnames(fit)
        c, s = coef(fit), stderror(fit)
        withheld = _separated(fit)
        for k in eachindex(names)
            push!(group, key); push!(term, names[k]); push!(est, c[k]); push!(se, s[k])
            push!(z, withheld ? NaN : c[k] / s[k]); push!(n, nobs(fit))
        end
    end
    table = DataFrame(group=group, term=term, estimate=est, std_error=se, z=z, n_events=n)
    reference === nothing && return table
    _check_disjoint(fits)
    any(isequal(reference), group) || throw(ArgumentError(
        "reference=$(repr(reference)) is not one of the groups " *
        "$(join(unique(repr.(group)), ", "))"))
    ref = Dict(t => (e, s) for (g, t, e, s) in zip(group, term, est, se)
               if isequal(g, reference))
    separated_groups = Set(key for (key, fit) in pairs_ if _separated(fit))
    diff = Union{Missing,Float64}[]; sed = Union{Missing,Float64}[]
    pd = Union{Missing,Float64}[]
    for (g, t, e, s) in zip(group, term, est, se)
        if isequal(g, reference) || !haskey(ref, t)
            push!(diff, missing); push!(sed, missing); push!(pd, missing)
        else
            e0, s0 = ref[t]
            d = e - e0; sd = sqrt(s^2 + s0^2)
            push!(diff, d); push!(sed, sd)
            # no Wald test against a fit without a finite maximum
            withheld = g in separated_groups || reference in separated_groups
            push!(pd, withheld ? NaN : only(z_pvalues([d / sd])))
        end
    end
    table.difference = diff
    table.se_difference = sed
    table.p_difference = pd
    return table
end

_check_disjoint(fits) = nothing
function _check_disjoint(fits::AbstractVector{<:NamedTuple})
    for k in 2:length(fits)
        fits[k].from < fits[k - 1].to && throw(ArgumentError(
            "the windows overlap (step < width), so their estimates are dependent and " *
            "the Wald test of a difference would be invalid; refit with step >= width"))
    end
    return nothing
end

_fit_pairs(fits::AbstractDict) = sort!(collect(pairs(fits)); by=p -> string(first(p)))
_fit_pairs(fits::AbstractVector{<:NamedTuple}) = [w.from => w.fit for w in fits]
_fit_pairs(fits::AbstractVector{<:Pair}) = fits
_fit_pairs(fits) = throw(ArgumentError(
    "compare_coefficients takes the result of fit_stratified or " *
    "fit_moving_window, or a vector of `key => fit` pairs"))

"""
    profile_memory(build, events, n_actors, grid; level=0.95, kwargs...) -> DataFrame

Estimate a memory parameter by profile likelihood: for each `value` in `grid`,
fit the model whose statistics are `build(value)` and record the maximised
log-likelihood. The table has one row per value — `value`, `loglik`, `aic`,
`bic`, `converged`, `in_ci`, `best`. `best` marks the maximum among the fits
that converged to a finite likelihood; `in_ci` marks the values inside the
profile-likelihood confidence set `{h : 2(ℓ_max − ℓ(h)) ≤ χ²₁(level)}`, which
is as fine as the grid. `aic` and `bic` count the memory parameter as estimated,
so they compare directly with a model that has no memory parameter. A value
whose fit has no finite maximum (separation) is kept with `converged = false`
and left out of `best` and `in_ci`; a value whose fit throws is kept with
`loglik = NaN`, `converged = false`, and the error is reported as a warning.

The literature's guidance is that memory should be estimated, not assumed:
transitivity estimates move from about zero to above one across plausible
half-lives (Arena, Mulder & Leenders 2023), and half-lives and windows in
applied work span orders of magnitude. The same function profiles a window
width, a power-law exponent (the grid search of the Lomi–Vu line) or any other
scalar the statistics depend on. Keywords are those of [`fit_revel`](@ref); with
`n_controls`, every grid value is fitted on the same draw of controls (the `rng`
is copied before each fit), so the profile compares models, not samples.

The profile treats the memory parameter as fixed when the standard errors of the
other coefficients are computed; they do not reflect its uncertainty.

# Example
```julia
using Revel, Random
truth = [Inertia(memory=HalfLife(5.0), transform=:log1p)]
events = simulate_events(truth, [1.5], 6, 300; rng=Xoshiro(8))
profile = profile_memory(events, 6, [1.0, 5.0, 25.0, 125.0]) do h
    [Inertia(memory=HalfLife(h), transform=:log1p)]
end
profile.value[profile.best]      # [5.0] — the half-life the data prefer
profile.value[profile.in_ci]     # the half-lives the data do not reject
```
"""
function profile_memory(build, events::AbstractVector{<:Event}, n_actors::Int, grid;
                        level::Real=0.95, rng::AbstractRNG=Random.default_rng(),
                        kwargs...)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))
    values = collect(grid)
    isempty(values) && throw(ArgumentError("the grid is empty"))
    loglik = Float64[]; aics = Float64[]; bics = Float64[]; converged = Bool[]
    start = copy(rng)
    for v in values
        fit = try
            fit_revel(events, build(v), n_actors; rng=copy(start), kwargs...)
        catch err
            err isa InterruptException && rethrow()
            @warn "profile_memory: the fit at value $v failed" exception = err
            nothing
        end
        if fit === nothing
            push!(loglik, NaN); push!(aics, NaN); push!(bics, NaN); push!(converged, false)
        else
            ll = loglikelihood(fit)
            push!(loglik, ll)
            push!(aics, aic(fit) + 2)                       # + the memory parameter
            push!(bics, bic(fit) + log(nobs(fit)))
            push!(converged, _converged(fit))
        end
    end
    usable = [converged[k] && isfinite(loglik[k]) for k in eachindex(values)]
    any(usable) || throw(ArgumentError(
        "no value of the grid gave a converged fit with a finite likelihood"))
    lmax = maximum(loglik[usable])
    best = [usable[k] && loglik[k] == lmax for k in eachindex(values)]
    # only the first maximiser is `best`
    first_best = findfirst(best)
    best .= false; best[first_best] = true
    cut = quantile(Chisq(1), level) / 2
    in_ci = [usable[k] && lmax - loglik[k] <= cut for k in eachindex(values)]
    return DataFrame(value=values, loglik=loglik, aic=aics, bic=bics,
                     converged=converged, in_ci=in_ci, best=best)
end

_converged(fit::RevelFit) = fit.fit.converged

# Whether the fit has no finite maximum (the shared separation verdict found a
# direction of recession). REM's result and the engine's both store the
# separated coefficient names as `separated`.
_separated(fit::RevelFit) = !isempty(fit.fit.separated)
