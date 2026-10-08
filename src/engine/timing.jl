# =============================================================================
# The interval-timing likelihood on the full risk set
# =============================================================================

"""
    Revel.is_interval_constant(stat) -> Bool

Whether `compute(stat, history, sender, receiver, t)` is constant as `t`
advances between events with the history and candidate dyad fixed. The
interval-timing likelihood ([`Revel.fit_timing`](@ref), `fit_revel(…;
model=:timing)`) requires it, because its exposure is the waiting time times the
hazard. The default is `false`: a custom statistic must explicitly extend this
function (`Revel.is_interval_constant(::MyStat) = true`) to declare that the
identity holds for every admissible history. The trait is `public`, not exported.

Revel's statistics answer it themselves: `true` for layers on
[`FullMemory`](@ref) (or `HalfLife(Inf)`) and for any layer on the event clock
(`clock=:order`), for static covariates, recency ranks and participation shifts
([`PShift`](@ref), [`PShiftABAB`](@ref), [`UndirectedPShift`](@ref)), and for
`TimeSince(…; clock=:order)`; `false` for a decaying memory on the time clock,
[`TimeSince`](@ref) on the time clock, [`GlobalEffect`](@ref) and time-varying
covariates. A wrapper ([`Interaction`](@ref), [`Transformed`](@ref),
[`Standardized`](@ref)) is constant when its parts are. Finite-half-life
statistics remain available to the ordinal likelihood; their continuously
varying hazard is not integrated by the timing fitter.

# Example
```julia
using Revel
Revel.is_interval_constant(PShift(:AB_BA))                  # true
Revel.is_interval_constant(Inertia())                       # true
Revel.is_interval_constant(Inertia(memory=HalfLife(2.0)))   # false
```
"""
is_interval_constant(::AbstractStatistic) = false
is_interval_constant(::PShift) = true

# The model specification behind a `TimingModelResult`. Only the exponential
# baseline has a likelihood here.
struct TimingModel
    statistics::Vector{AbstractStatistic}
    baseline::Symbol
end

"""
    Revel.TimingModelResult

The result of [`Revel.fit_timing`](@ref), the interval-timing relational event
model with an exponential baseline, fitted by exact maximum likelihood over the
full risk set. A [`RevelFit`](@ref) with `model=:timing` holds one in `fit.fit`.

The coefficient vector is `[log(λ₀); θ]`: `coef`, `coefnames` (which starts with
`"log_baseline"`), `stderror`, `vcov`, `confint` and `coeftable` all use that
order, and `dof` counts the baseline. The effect-only vectors remain in the
fields `coefficients` and `std_errors`, and `baseline_params[1]` is the fitted
baseline rate λ₀. `nobs` counts events (a right-censored tail is exposure, not
an event). The metadata protocol (`NetworkCore.fit_metadata`) reports the exact
likelihood, `is_exact == false` as soon as the data carried tied event times
(under a continuous-time model a tie has probability zero, so no policy can make
the likelihood exact for such data), and `tie_method` (`:none`, `:ordered` or
`:batch`). `separation` and `separated` are as in
[`Revel.OrdinalBPMResult`](@ref), with `"log_baseline"` among the names that can
be flagged.

# Example
```julia
using Revel
events = [Event(1, 2, 1.0), Event(2, 3, 2.0), Event(3, 1, 3.0),
          Event(1, 3, 4.0), Event(2, 1, 5.0), Event(3, 2, 6.0)]
fit = Revel.fit_timing(events, [SendEffect([0.0, 1.0, 2.0])], 3; t_end=7.0)
fit isa Revel.TimingModelResult     # true
coefnames(fit)                      # ["log_baseline", "send.x"]
abs(coef(fit)[2]) < 1e-8            # true — every sender acts twice
exp(coef(fit)[1]) == fit.baseline_params[1]   # true
```
"""
struct TimingModelResult
    model::TimingModel
    coefficients::Vector{Float64}
    baseline_params::Vector{Float64}
    std_errors::Vector{Float64}
    loglik::Float64
    converged::Bool
    # What was ACTUALLY done with tied event times: `:none` (the data had none —
    # which is what a continuous-time model expects), `:ordered` or `:batch`.
    tie_type::Symbol
    log_baseline::Float64
    log_baseline_se::Float64
    var_cov::Matrix{Float64}
    n_events::Int
    iterations::Int
    # The shared separation verdict (coefficients in `coef` order, the log
    # baseline first) and the names of the coefficients it flags.
    separation::SeparationVerdict
    separated::Vector{String}
end

function Base.show(io::IO, result::TimingModelResult)
    println(io, "Interval-timing relational event model (full risk set)")
    println(io, "======================================================")
    println(io, "Baseline: $(result.model.baseline)")
    println(io, "Baseline rate λ₀: $(round(result.baseline_params[1], digits=6))")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
    result.tie_type === :none ||
        println(io, "Tied event times: $(result.tie_type) " *
                    "(a continuous-time model gives a tie probability zero)")
    println(io)
    show(io, coeftable(result))
    _show_separation(io, result)
end

estimand(::TimingModelResult) = :relational_event_timing
objective(::TimingModelResult) = :likelihood
# Exact without ties; with ties the data contradicts the continuous-time process
# rather than merely under-determining it.
is_exact(result::TimingModelResult) = result.tie_type === :none
se_method(::TimingModelResult) = :hessian
missing_method(::TimingModelResult) = :none
tie_method(result::TimingModelResult) = result.tie_type

function approximations(result::TimingModelResult)
    out = String[]
    caveat = separation_caveat(result.separated)
    caveat === nothing || push!(out, caveat)
    result.converged || caveat !== nothing ||
        push!(out, "the Newton-Raphson maximization did NOT converge: the reported " *
                   "optimizer did not certify an identified optimum")
    if result.tie_type === :ordered
        push!(out, "tied event times were ordered arbitrarily with NO correction " *
                   "(`ties=:ordered`): each tied event after the first enters as a " *
                   "ZERO-LENGTH waiting interval — it contributes an event term with " *
                   "no exposure — while still updating the statistics of the next, " *
                   "i.e. the fit claims one event caused another in no time at all. " *
                   "Under the continuous-time model being fitted, a tie has " *
                   "probability zero")
    elseif result.tie_type === :batch
        push!(out, "tied event times were read as a simultaneous BATCH " *
                   "(`ties=:batch`): the tied events could not have influenced one " *
                   "another (statistics frozen across the block) and the block " *
                   "consumes ONE exposure interval. This is the likelihood of a " *
                   "COARSENED observation process, not of the continuous-time model " *
                   "the exponential likelihood otherwise assumes — under which a tie " *
                   "has probability zero")
    end
    return out
end

"""
    Revel.fit_timing(events, statistics, n_actors; t0=zero(T), t_end=nothing,
                     ties=:error, cache=:auto, chunk=nothing, cache_bytes=2^28,
                     maxiter=100, tol=1e-8) -> Revel.TimingModelResult

Fit the interval-timing relational event model with an exponential baseline by
exact maximum likelihood over the full risk set: with waiting time `Δt_m` before
event m and per-dyad hazards `λ₀·exp(θ'x_ij)` (statistics constant between
events),

    ℓ(λ₀, θ) = Σ_m [ log λ₀ + θ'x_case − λ₀ Δt_m Σ_{ij} exp(θ'x_ij) ].

`(log λ₀, θ)` are estimated jointly by `NetworkCore.newton_fit`; standard errors
come from the observed information. This is the estimator behind
`fit_revel(…; model=:timing)`, which wraps the result for the diagnostics:
prefer it. Only the exponential baseline is implemented (`baseline=:exponential`,
the default; any other value is refused).

Every statistic must satisfy [`Revel.is_interval_constant`](@ref); a statistic
that is not (a decaying memory on the time clock, an uncertified custom
statistic) is refused with an `ArgumentError`, because multiplying the hazard
evaluated at an event endpoint by the preceding interval is not its integral.
Fit such statistics with the ordinal likelihood, or define an interval-constant
model deliberately (a `FullMemory` layer, or `clock=:order`) — changing the
memory changes the model.

# Observation window

`t0` is the observation onset: the first event's waiting time is
`Δt_1 = t_1 − t0`. The default `t0 = zero(T)` matches `relevent`, where "event
times should be relative to onset of observation". For a process already running
when recording started, pass the window start as `t0`; it must not exceed the
first event time.

`t_end` is the time recording stopped. It adds the right-censored final interval
`[t_M, t_end]` — exposure with no event — to the likelihood,
`ℓ += −λ₀ (t_end − t_M) Σ_{ij} exp(θ'x_ij)`. `relevent::rem.dyad` always has this
term (its last edgelist row is the termination time, and any event on it is
ignored), so `t_end` is required to reproduce it: without it the sequence is
treated as ending at its last event, and λ₀ is biased upward. (`rem.dyad`'s
temporal likelihood also has no intercept: give it a constant `CovSnd` column,
whose coefficient is then `log λ₀`.)

# Separation

Decided before the optimizer runs, by the shared verdict, and handled as in
[`Revel.fit_obpm`](@ref): a warning, `converged == false`, `fit.separated`
naming the coefficients (`"log_baseline"` among them when the baseline runs
away), and NaN z values, p-values and confidence intervals. The likelihood is a
Poisson log-linear likelihood with offset `log Δt` over the dyads of every
interval of positive length, so a direction separates when no dyad's log-rate
rises in an interval with exposure, no event's log-rate falls, and some exposure
strictly falls. The censored tail counts as exposure. An event in a zero-length
interval (a tie under `ties=:ordered` or `:batch`) is required to rise on its
own; this can miss a direction along which such events trade off against each
other, but never reports a separation that does not exist.

# Risk-set caching

`cache=:auto|:all|:chunked|:none` (with `chunk` and `cache_bytes`) bounds the
memory the risk-set design matrices take, exactly as in [`Revel.fit_obpm`](@ref);
the fit is bit-identical under every policy.

# Tied event times

This is an **exact-time** likelihood. Under the continuous-time process it fits,
two events at one instant have probability **zero**: a tie is the model's own
assumption failing — a coarse clock or a genuinely batched observation — and the
policies (`NetworkCore.TIE_POLICIES`) say which:

- `ties=:error` (default) — name the tie and refuse.
- `ties=:ordered` — each tied event after the first enters with a
  **zero-length waiting interval** (an event term with no exposure) while still
  updating the statistics of the next: the fit then claims one event caused
  another in no time at all.
- `ties=:batch` — one simultaneous batch: the statistics are frozen across the
  block and the block consumes **one** exposure interval. The
  coarsened-observation reading, and the one to prefer when the clock is coarse.
- `ties=:breslow` / `ties=:efron` — **refused**: Breslow and Efron correct a
  *partial* likelihood, in which the baseline hazard is profiled out and only the
  order of events survives. Fit the ordinal model with `ties=:efron` if the order
  is what matters.

On tie-free data every policy gives the identical fit, and `is_exact(fit)` turns
`false` as soon as a tie was in the data.

# Example
```julia
using Revel, Random
truth = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(truth, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
fit = Revel.fit_timing(events, truth, 6; t_end=events[end].time + 1.0)
coefnames(fit)                  # ["log_baseline", "log1p(inertia)", "log1p(reciprocity)"]
fit.converged                   # true
decaying = try Revel.fit_timing(events, [Inertia(memory=HalfLife(5.0))], 6) catch e; e end
decaying isa ArgumentError      # true — a decaying memory is not interval-constant
```
"""
function fit_timing(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
                    n_actors::Int; baseline::Symbol=:exponential,
                    t0::T=zero(T), t_end::Union{Nothing,T}=nothing,
                    ties::Symbol=:error, cache::Symbol=:auto,
                    chunk::Union{Nothing,Int}=nothing,
                    cache_bytes::Int=_DEFAULT_CACHE_BYTES,
                    maxiter::Int=100, tol::Float64=1e-8) where T
    check_tie_policy(ties, _TIMING_TIES_SUPPORTED; model=_TIMING_TIES_MODEL,
                     reasons=_TIMING_TIES_REASONS)
    baseline === :exponential || throw(ArgumentError(
        "fit_timing implements the exponential-baseline likelihood only; " *
        "baseline=:$baseline is not implemented"))
    isempty(statistics) && throw(ArgumentError("need at least one statistic"))
    n_actors >= 2 || throw(ArgumentError("need at least two actors"))
    model = TimingModel(collect(AbstractStatistic, statistics), baseline)
    isempty(events) && throw(ArgumentError("no events to fit"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))

    for stat in model.statistics
        is_interval_constant(stat) || throw(ArgumentError(
            "fit_timing requires statistics constant between events; " *
            "$(name(stat)) ($(typeof(stat))) does not satisfy " *
            "Revel.is_interval_constant. Continuously varying hazards require " *
            "exposure integration, which this fitter does not implement. Use the " *
            "ordinal model for conditional event choice, or specify an " *
            "interval-constant statistic (a FullMemory layer, or clock=:order — " *
            "which changes the model)."))
    end

    sorted = sort(events, by=e -> e.time)
    blocks = _tie_blocks(sorted)
    has_ties = any(b -> length(b) > 1, blocks)
    ties === :error && has_ties && _reject_ties(sorted, blocks,
        "This is an EXACT-TIME likelihood: under the continuous-time process it " *
        "fits, two events at one instant have probability ZERO, so a tie is not " *
        "an ambiguous ordering but the model's own assumption failing — the clock " *
        "is coarse, or the events are genuinely simultaneous.",
        "Say which: `ties=:batch` (a coarsened simultaneous batch — the tied " *
        "events cannot influence one another and the block consumes one exposure " *
        "interval) or `ties=:ordered` (each tied event after the first enters " *
        "with a zero-length waiting interval). Breslow and Efron do not apply to " *
        "an exact-time likelihood; use `fit_obpm(...; ties=:efron)` if only the " *
        "order matters.")
    tie_applied = has_ties ? ties : :none

    p = length(statistics)
    plan = _risk_set_plan(events, model.statistics, n_actors;
                          t0=t0, t_end=t_end, ties=ties)
    rs = _risk_sets(plan; cache=cache, chunk=chunk, cache_bytes=cache_bytes)
    derivatives = _timing_derivatives(rs)

    verdict = _separation_verdict(rs; timing=true, budget=cache_bytes)
    names = ["log_baseline"; [name(s) for s in model.statistics]]
    opt = _quietly(verdict.separated) do
        newton_fit(derivatives, zeros(p + 1); maxiter=maxiter, tol=tol)
    end
    warn_separation("fit_timing", verdict, names)
    opt.converged || verdict.separated || @warn "fit_timing did not converge; inspect the model for non-identification, or increase maxiter" iterations=opt.iterations
    β = opt.θ
    return TimingModelResult(model, β[2:end], [exp(β[1])], opt.se[2:end],
                             opt.loglik, opt.converged && !verdict.separated,
                             tie_applied, β[1], opt.se[1], opt.vcov, length(events),
                             opt.iterations, verdict, names[verdict.terms])
end
