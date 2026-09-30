# =============================================================================
# Fitting
# =============================================================================
#
# Revel hosts no optimizer and no likelihood kernel of its own (the ecosystem's
# shared-numerics rule). `fit_revel` routes a model to the estimator that
# already owns its likelihood:
#
#   ordinal, full directed risk set, every event a case
#       → Relevent.fit_obpm   (exact, streamed risk sets, separation certificate)
#   ordinal, anything else (a restricted or sampled risk set, a subset of
#   cases, undirected events, sandwich standard errors)
#       → the design frame, then REM.fit_rem(::DataFrame, names)
#   interval timing
#       → Relevent.fit_timing
#
# and wraps the result together with what is needed to rebuild its risk sets, so
# that the goodness-of-fit diagnostics can be computed from the fit alone.

"""
    RevelFit

A fitted relational event model: the underlying estimator's result (`fit.fit` —
a `Relevent.OrdinalBPMResult`, a `Relevent.TimingModelResult` or a
`REM.REMResult`) together with the time-sorted events, the statistics and the
risk-set specification that produced it.

It answers the StatsAPI verbs (`coef`, `stderror`, `vcov`, `confint`,
`loglikelihood`, `nobs`, `dof`, `aic`, `bic`, `coeftable`, `coefnames`) and the
ecosystem's result-metadata protocol (`Networks.fit_metadata`), and it is what
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
struct RevelFit{F, T}
    fit::F
    events::Vector{Event{T}}
    statistics::Vector{AbstractStatistic}
    n_actors::Int
    model::Symbol
    engine::Symbol
    directed::Bool
    riskset::Any
    cases::Union{Nothing, BitVector}
    ties::Symbol
    n_controls::Union{Nothing, Int}
end

const _FIT_MODELS = (:ordinal, :timing)
const _FIT_ENGINES = (:auto, :relevent, :design)

_riskset_label(rs::Symbol) = String(rs)
_riskset_label(rs::AbstractVector) = "$(length(rs)) listed dyads"
_riskset_label(::Function) = "time-varying"
_riskset_label(rs) = string(rs)

function Base.show(io::IO, fit::RevelFit)
    println(io, "Revel relational event model")
    println(io, "  model:     ", fit.model === :ordinal ? "ordinal (event order)" :
                                 "interval timing (exponential baseline)")
    println(io, "  events:    ", length(fit.events),
            fit.cases === nothing ? "" : " ($(count(fit.cases)) modelled as cases)",
            fit.directed ? ", directed" : ", undirected")
    println(io, "  actors:    ", fit.n_actors)
    println(io, "  risk set:  ", _riskset_label(fit.riskset),
            fit.n_controls === nothing ? "" : " ($(fit.n_controls) sampled controls)")
    println(io, "  estimator: ", fit.engine === :relevent ?
            (fit.model === :ordinal ? "Relevent.fit_obpm" : "Relevent.fit_timing") :
            "REM.fit_rem on the event design")
    println(io)
    show(io, fit.fit)
end

"""
    fit_revel(events, statistics, n_actors; model=:ordinal, directed=true,
              riskset=:full, cases=nothing, ties=:error, n_controls=nothing,
              rng=Random.default_rng(), se=:hessian, engine=:auto,
              maxiter=100, tol=1e-8, kwargs...) -> RevelFit

Fit a relational event model with any statistics of the ecosystem — Revel's,
Relevent's and REM's alike. [`revel`](@ref) is an alias.

`model=:ordinal` (default) models which dyad acts next given the history — the
ordinal likelihood of `relevent::rem.dyad`, a conditional logit with one stratum
per event. `model=:timing` adds the waiting times under an exponential baseline
(`Relevent.fit_timing`; pass `t0`/`t_end` through `kwargs`); it needs the full
directed risk set and statistics that are constant between events, so
[`FullMemory`](@ref) layers only.

For the ordinal model:

- `directed`, `riskset`, `cases` and `ties` are as in [`each_risk_set`](@ref).
  `riskset=:sender` fits the receiver-choice step of an actor-oriented model
  (see [`fit_receiver_choice`](@ref)); a vector from
  [`two_mode_dyads`](@ref) fits a two-mode model; `cases` restricts the
  likelihood to some events while the rest still build the history.
- `n_controls` samples that many controls per event (nested case-control
  sampling) instead of enumerating the risk set, with all randomness from `rng`.
- `se` — `:hessian` or `:sandwich` (event-clustered, `REM.fit_rem`'s).
- `engine` — `:relevent` uses `Relevent.fit_obpm` (exact, bounded memory, and it
  refuses a model with a verified separating direction); `:design` builds the
  [`event_design`](@ref) frame and fits it with `REM.fit_rem`. `:auto` picks
  `:relevent` whenever the model is one it can fit (full directed risk set, every
  event a case, no sampling, `se=:hessian`) and `:design` otherwise. Both
  maximise the same likelihood; on a common model they agree to optimiser
  tolerance.

Remaining keywords are forwarded to the underlying fitter (`cache`, `chunk`,
`cache_bytes` for `Relevent.fit_obpm`/`fit_timing`; `t0`, `t_end` for the timing
model).

Statistic names must be unique; pass `name=` to tell apart two effects that
differ only in a filter.

# Example
```julia
using Revel, Random
truth = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(truth, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, truth, 6)
round.(coef(fit); digits=1)                      # close to [0.8, 0.6]
choice = fit_revel(events, truth, 6; riskset=:sender)   # receiver choice
nobs(choice)                                     # 200
```
"""
function fit_revel(events::Vector{Event{T}},
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
        "engine must be :auto, :relevent or :design, got :$engine"))
    isempty(events) && throw(ArgumentError("no events to fit"))
    names = _stat_names(statistics)
    stats = collect(AbstractStatistic, statistics)
    sorted = sort(events; by=e -> e.time)
    mask = cases === nothing ? nothing : _case_mask(cases, sorted)
    mask === nothing || any(mask) || throw(ArgumentError(
        "`cases` selects no event; there is nothing to fit"))

    plain = directed && riskset === :full && cases === nothing && n_controls === nothing

    if model === :timing
        plain || throw(ArgumentError(
            "model=:timing is the exact-time likelihood of `Relevent.fit_timing`, " *
            "which enumerates the full directed risk set and treats every event as " *
            "a case. A restricted or sampled risk set, undirected events and a " *
            "subset of cases are available for model=:ordinal only."))
        se === :hessian || throw(ArgumentError(
            "model=:timing offers se=:hessian only"))
        engine === :design && throw(ArgumentError(
            "model=:timing is fitted by Relevent.fit_timing; engine=:design is for " *
            "the ordinal model"))
        inner = Relevent.fit_timing(sorted, stats, n_actors; ties=ties, maxiter=maxiter,
                                    tol=tol, kwargs...)
        return RevelFit{typeof(inner), T}(inner, sorted, stats, n_actors, :timing,
                                          :relevent, true, :full, nothing, ties, nothing)
    end

    use_relevent = engine === :relevent || (engine === :auto && plain && se === :hessian)
    if use_relevent
        plain && se === :hessian || throw(ArgumentError(
            "engine=:relevent (Relevent.fit_obpm) fits the full directed risk set " *
            "with every event a case and se=:hessian; use engine=:design (or " *
            ":auto) for a restricted or sampled risk set, undirected events, a " *
            "subset of cases or se=:sandwich"))
        inner = Relevent.fit_obpm(sorted, stats, n_actors; ties=ties, maxiter=maxiter,
                                  tol=tol, kwargs...)
        return RevelFit{typeof(inner), T}(inner, sorted, stats, n_actors, :ordinal,
                                          :relevent, true, :full, nothing, ties, nothing)
    end

    isempty(kwargs) || throw(ArgumentError(
        "unknown keyword$(length(kwargs) == 1 ? "" : "s") for the design engine: " *
        join((":" * string(k) for k in keys(kwargs)), ", ")))
    design = event_design(sorted, stats, n_actors; directed=directed, riskset=riskset,
                          ties=ties, cases=mask, n_controls=n_controls, rng=rng)
    inner = REM.fit_rem(design, names; maxiter=maxiter, tol=tol, se=se)
    return RevelFit{typeof(inner), T}(inner, sorted, stats, n_actors, :ordinal, :design,
                                      directed, riskset, mask, ties, n_controls)
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
estimated here; it can enter only as a moderator, through an
[`Interaction`](@ref). Stadtfeld & Block report that tie-oriented and
actor-oriented estimates of the same effect can differ in sign, because a
tie-oriented inertia × sender term conflates "active senders" with "repetitive
senders"; fitting both is a specification check.

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
fit_receiver_choice(events, statistics, n_actors::Int; kwargs...) =
    fit_revel(events, statistics, n_actors; riskset=:sender, kwargs...)

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
round.(coef(fit); digits=1)          # close to [0.8, 0.6]
```
"""
coef(fit::RevelFit) = coef(fit.fit)

"""
    stderror(fit::RevelFit) -> Vector{Float64}

Standard errors of the coefficients (StatsAPI): the square roots of the
diagonal of [`vcov`](@ref), under the estimator `Networks.se_method(fit)` names.

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
    coeftable(fit::RevelFit) -> Networks.CoefficientTable

The coefficient table — estimates, standard errors, z- and p-values — as the
ecosystem's shared `Networks.CoefficientTable` (StatsAPI's `coeftable`).

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = fit_revel(events, stats, 6)
table = coeftable(fit)
table isa Revel.Networks.CoefficientTable    # true
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
by event type (`by = e -> e.eventtype`), weekday, period or context (Amati, Lomi
& Mascia 2019; Vu, Lomi, Mascia & Pallotti 2017). It equals a product-term model
only when *all* interactions with the stratifier are included, and it is not the
same model as a statistic filtered on the stratifier. Coefficient magnitudes
should not be compared across strata of a logit-type model; compare signs and
significance, or use [`compare_coefficients`](@ref) for the side-by-side table.

Keywords are those of [`fit_revel`](@ref) (the ordinal model).

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
function fit_stratified(events::Vector{Event{T}}, statistics, n_actors::Int; by,
                        kwargs...) where T
    by isa Function || throw(ArgumentError(
        "`by` must be a function event -> stratum key (e.g. `e -> e.eventtype`)"))
    sorted = sort(events; by=e -> e.time)
    keys_ = [by(e) for e in sorted]
    fits = Dict{Any, RevelFit}()
    for key in unique(keys_)
        fits[key] = fit_revel(sorted, statistics, n_actors; cases=keys_ .== Ref(key),
                              kwargs...)
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
Leenders 2019; Meijerink-Bosman, Leenders & Mulder 2022; remstimate
`remwindow`): it shows whether an effect such as reciprocity strengthens or
fades over the observation period without committing to a functional form.
Overlapping windows (`step < width`) smooth the path and make neighbouring
estimates dependent. Use [`compare_coefficients`](@ref) for the path as a table.

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
function fit_moving_window(events::Vector{Event{T}}, statistics, n_actors::Int;
                           width::Real, step::Real=width, min_events::Int=10,
                           kwargs...) where T
    width > 0 && step > 0 || throw(ArgumentError("width and step must be positive"))
    isempty(events) && throw(ArgumentError("no events to fit"))
    sorted = sort(events; by=e -> e.time)
    times = [_tfloat(e.time) for e in sorted]
    t_first, t_last = first(times), last(times)
    out = NamedTuple{(:from, :to, :n_events, :fit), Tuple{Float64, Float64, Int, RevelFit}}[]
    from = t_first
    while from <= t_last
        to = from + width
        mask = (times .>= from) .& (times .< to)
        n = count(mask)
        if n >= min_events
            fit = fit_revel(sorted, statistics, n_actors; cases=mask, kwargs...)
            push!(out, (from=from, to=to, n_events=n, fit=fit))
        end
        from += step
    end
    isempty(out) && throw(ArgumentError(
        "no window holds at least min_events=$min_events events; widen the " *
        "windows or lower min_events"))
    return out
end

"""
    compare_coefficients(fits) -> DataFrame

A long table — one row per (fit, coefficient) with `group`, `term`, `estimate`,
`std_error`, `z`, `n_events` — of several fits of the same specification: the
`Dict` returned by [`fit_stratified`](@ref), the vector returned by
[`fit_moving_window`](@ref) (grouped by window start), or any collection of
`key => fit` pairs. It lays the stratified, windowed and pooled estimates side
by side, which is how the literature reads a moderation by refitting.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p)]
events = simulate_events(stats, [1.0], 5, 120; rng=Xoshiro(5))
fits = fit_stratified(events, stats, 5; by = e -> e.time <= 60 ? :early : :late)
table = compare_coefficients(fits)
size(table)                # (2, 6)
names(table)               # ["group", "term", "estimate", "std_error", "z", "n_events"]
```
"""
function compare_coefficients(fits)
    pairs_ = _fit_pairs(fits)
    group = Any[]; term = String[]; est = Float64[]; se = Float64[]; z = Float64[]
    n = Int[]
    for (key, fit) in pairs_
        names = coefnames(fit)
        c, s = coef(fit), stderror(fit)
        for k in eachindex(names)
            push!(group, key); push!(term, names[k]); push!(est, c[k]); push!(se, s[k])
            push!(z, c[k] / s[k]); push!(n, nobs(fit))
        end
    end
    return DataFrame(group=group, term=term, estimate=est, std_error=se, z=z, n_events=n)
end

_fit_pairs(fits::AbstractDict) = sort!(collect(pairs(fits)); by=p -> string(first(p)))
_fit_pairs(fits::AbstractVector{<:NamedTuple}) = [w.from => w.fit for w in fits]
_fit_pairs(fits::AbstractVector{<:Pair}) = fits
_fit_pairs(fits) = throw(ArgumentError(
    "compare_coefficients takes the result of fit_stratified or " *
    "fit_moving_window, or a vector of `key => fit` pairs"))

"""
    profile_memory(build, events, n_actors, grid; kwargs...) -> DataFrame

Estimate a memory parameter by profile likelihood: for each `value` in `grid`,
fit the model whose statistics are `build(value)` and record the maximised
log-likelihood. The table has one row per value — `value`, `loglik`, `aic`,
`bic`, `converged`, `best` — and `best` marks the maximum.

The literature's guidance is that memory should be estimated, not assumed:
transitivity estimates move from about zero to above one across plausible
half-lives (Arena, Mulder & Leenders 2023), and half-lives and windows in
applied work span orders of magnitude. The same function profiles a window
width, a power-law exponent (the grid search of the Lomi–Vu line) or any other
scalar the statistics depend on. Keywords are those of [`fit_revel`](@ref).

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
profile.value[profile.best]      # the half-life the data prefer
```
"""
function profile_memory(build, events::Vector{<:Event}, n_actors::Int, grid; kwargs...)
    values = collect(grid)
    isempty(values) && throw(ArgumentError("the grid is empty"))
    loglik = Float64[]; aics = Float64[]; bics = Float64[]; converged = Bool[]
    for v in values
        fit = fit_revel(events, build(v), n_actors; kwargs...)
        push!(loglik, loglikelihood(fit)); push!(aics, aic(fit)); push!(bics, bic(fit))
        push!(converged, _converged(fit))
    end
    best = falses(length(values))
    best[argmax(loglik)] = true
    return DataFrame(value=values, loglik=loglik, aic=aics, bic=bics,
                     converged=converged, best=collect(best))
end

_converged(fit::RevelFit) = fit.fit.converged
