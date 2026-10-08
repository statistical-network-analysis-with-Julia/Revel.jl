# =============================================================================
# The ordinal likelihood on the full risk set
# =============================================================================

# The model specification behind an `OrdinalBPMResult`: the statistics and the
# actor universe. Each event contributes a multinomial-logit term over the full
# risk set, P(event m is (s, r)) = exp(θ'x_sr) / Σ_{(i,j)} exp(θ'x_ij) — the
# ordinal likelihood of `relevent::rem.dyad`.
struct OrdinalBPM
    statistics::Vector{AbstractStatistic}
    n_actors::Int

    function OrdinalBPM(statistics::Vector{<:AbstractStatistic}, n::Int)
        isempty(statistics) && throw(ArgumentError("need at least one statistic"))
        n >= 2 || throw(ArgumentError("need at least two actors"))
        new(collect(AbstractStatistic, statistics), n)
    end
end

"""
    Revel.OrdinalBPMResult

The result of [`Revel.fit_obpm`](@ref), the ordinal relational event model
(Butts 2008) fitted by exact maximum likelihood over the full risk set, with
observed-information standard errors. A [`RevelFit`](@ref) of the ordinal model
on the full directed risk set holds one in `fit.fit`.

It answers the StatsAPI verbs — `coef`, `coefnames`, `stderror`, `vcov`,
`confint`, `loglikelihood`, `nobs` (events), `dof`, `aic`, `aicc`, `bic` and
`coeftable` (a `NetworkCore.CoefficientTable`) — and the ecosystem's
result-metadata protocol (`NetworkCore.fit_metadata`): the objective is the
ordinal likelihood itself (nothing is sampled), `is_exact` is `true` on data
without tied event times, and `tie_method` reports what was done with ties
(`:none`, `:ordered`, `:breslow` or `:efron`; `:error` never appears, since
under it a tie throws). Fields include `coefficients`, `std_errors`, `loglik`,
`converged`, `iterations`, `tie_type`, `separation` (the shared
`NetworkCore.SeparationVerdict` on the risk sets) and `separated` (the names of
the coefficients it flags, empty when a finite maximum exists).

# Example
```julia
using Revel
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 3, 3.0), Event(3, 1, 4.0),
          Event(2, 3, 5.0), Event(1, 2, 6.0), Event(3, 2, 7.0), Event(2, 1, 8.0)]
fit = Revel.fit_obpm(events, [PShift(:AB_BA)], 3)
fit isa Revel.OrdinalBPMResult      # true
coefnames(fit)                      # ["PSAB-BA"]
nobs(fit), dof(fit)                 # (8, 1)
Revel.NetworkCore.tie_method(fit)   # :none
```
"""
struct OrdinalBPMResult
    model::OrdinalBPM
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    loglik::Float64
    converged::Bool
    n_events::Int
    # What was ACTUALLY done with tied event times: `:none` (the data had no
    # ties), or the policy that bit — `:ordered`, `:breslow`, `:efron`. `:error`
    # can never appear: under it a tie throws instead of fitting.
    tie_type::Symbol
    var_cov::Matrix{Float64}
    iterations::Int
    # The shared separation verdict on the risk sets, and the names of the
    # coefficients it flags (empty when a finite maximum exists).
    separation::SeparationVerdict
    separated::Vector{String}
end

function Base.show(io::IO, result::OrdinalBPMResult)
    println(io, "Ordinal relational event model (full risk set)")
    println(io, "==============================================")
    println(io, "N actors: $(result.model.n_actors)")
    println(io, "N events: $(result.n_events)")
    println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
    println(io, "Converged: $(result.converged)")
    result.tie_type === :none ||
        println(io, "Tied event times: $(result.tie_type)")
    println(io)
    show(io, coeftable(result))
    _show_separation(io, result)
end

function _show_separation(io::IO, result)
    caveat = separation_caveat(result.separated)
    caveat === nothing || print(io, "\nWarning: ", caveat, ".")
    return nothing
end

# -----------------------------------------------------------------------------
# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`)
# -----------------------------------------------------------------------------
#
# Both full-risk-set models enumerate every dyad — the distinguishing property
# against sampled-control `REM.fit_rem` fits — so their objectives ARE the exact
# likelihoods. `fit_metadata(fit)` makes that claim inspectable, and names what
# stands in the way of exactness: tied timestamps, separation, non-convergence.

estimand(::OrdinalBPMResult) = :relational_event
objective(::OrdinalBPMResult) = :likelihood
# Exact on strictly ordered data; with ties the order the likelihood is over is
# not in the data, whichever policy stands in for it.
is_exact(result::OrdinalBPMResult) = result.tie_type === :none
se_method(::OrdinalBPMResult) = :hessian
missing_method(::OrdinalBPMResult) = :none
tie_method(result::OrdinalBPMResult) = result.tie_type

# Prose for the tie policy that actually bit; `nothing` when the data had no
# ties, since a correction on tie-free data corrected nothing.
function _tie_approximation(tie_type::Symbol)
    if tie_type === :ordered
        return "tied event times were ordered arbitrarily with NO tie correction " *
               "(`ties=:ordered`): the ordinal likelihood is a likelihood over the " *
               "ORDER of events, and the event placed first also enters the " *
               "statistics of the events placed after it, so the estimate depends " *
               "on a sort the data does not determine"
    elseif tie_type === :breslow
        return "tied event times were handled by the BRESLOW correction " *
               "(`ties=:breslow`): the tied events share one risk set (statistics " *
               "frozen across the tie) and each contributes the same denominator. " *
               "An approximation to the average over the d! orderings, and the " *
               "cruder of the two — it biases coefficients toward zero as ties get " *
               "heavier (`ties=:efron` is the better approximation)"
    elseif tie_type === :efron
        return "tied event times were handled by the EFRON correction " *
               "(`ties=:efron`): the tied cases enter the j-th denominator with " *
               "weight 1 − (j−1)/d. A close approximation to the average over the " *
               "d! orderings — what `survival::coxph` defaults to — but the order " *
               "of simultaneous events remains unobserved"
    end
    return nothing
end

function approximations(result::OrdinalBPMResult)
    out = String[]
    caveat = separation_caveat(result.separated)
    caveat === nothing || push!(out, caveat)
    result.converged || caveat !== nothing ||
        push!(out, "the Newton-Raphson maximization did NOT converge: the reported " *
                   "optimizer did not certify an identified optimum")
    tie_note = _tie_approximation(result.tie_type)
    isnothing(tie_note) || push!(out, tie_note)
    return out
end

"""
    Revel.fit_obpm(events, statistics, n_actors; ties=:error, cache=:auto,
                   chunk=nothing, cache_bytes=2^28, maxiter=100, tol=1e-8)
        -> Revel.OrdinalBPMResult

Fit the ordinal relational event model (Butts 2008) by exact maximum likelihood
over the full risk set of all `n(n−1)` ordered dyads: each event contributes the
multinomial-logit term `exp(θ'x_case) / Σ_{(i,j)} exp(θ'x_ij)`, with the
statistics read off the history before the event. This is the likelihood of
`relevent::rem.dyad(…, ordinal = TRUE)`, and the estimator behind
[`fit_revel`](@ref) for the ordinal model on the full directed risk set (which
also sorts the events, checks the statistics and wraps the result for the
diagnostics: prefer it). The likelihood is maximised by the ecosystem's shared
Newton optimizer, `NetworkCore.newton_fit`.

`statistics` may be any statistic with the history interface
`compute(stat, ::InteractionHistory, sender, receiver, time)`. The likelihood
models event choice conditional on the history, without a waiting-time
likelihood; finite-half-life statistics still use the observed time differences.

# Separation

Whether a finite maximum exists is decided on the risk sets, before the
optimizer runs, by the ecosystem's shared verdict: a direction along which every
case scores at least as high as every alternative of its risk set, strictly
somewhere, is searched for by a linear programme and certified in exact
arithmetic (`NetworkCore.separation_from_margins`; the risk set is a stratum of
a conditional logit). Complete and quasi-complete separation are both found, and
tiny genuine overlap is never rounded into one. A separated fit warns, returns
`converged == false`, names the separated coefficients in `fit.separated` (the
verdict is `fit.separation`), and withholds inference: z values, p-values and
`confint` are `NaN`. The coefficients and standard errors are kept for
diagnosis; they are where the optimizer stopped, not estimates.

# Risk-set caching

The full risk set means an `n(n−1) × p` design matrix per event. Materializing
them all costs `O(E · n² · p)` doubles — 906 MiB at `(n, E, p) = (100, 2000, 6)` —
so `cache=` decides how many are alive at once:

- `:auto` (default) — `:all` while the projected footprint fits in `cache_bytes`
  (256 MiB here; `fit_revel` passes a quarter of the free memory), `:chunked`
  above it.
- `:all` — every design matrix materialized once. Fastest; `O(E · n² · p)`.
- `:chunked` — a bounded cache of `chunk` matrices (default: as many as fit in
  `cache_bytes`), refilled by replaying the event history on each pass.
- `:none` — `:chunked` with `chunk = 1`: one matrix alive at a time, least
  memory, most recomputation.

Every policy visits the intervals in the same order and reads the same
statistics off the same histories, so the fits are bit-identical; only memory
and time differ.

# Tied event times

A tied timestamp means the event order is unobserved, and sorting the tie
invents it (worse: the statistics are read off the pre-event history, so the
event sorted first enters the *statistics* of the one sorted second). `ties=`
(the shared `NetworkCore.TIE_POLICIES` vocabulary) says what to do:

- `:error` (default) — name the tie and refuse.
- `:ordered` — sequence order, no correction.
- `:breslow` — the Breslow correction: the tied events share one risk set (the
  history is frozen across the tie block, so they cannot enter each other's
  statistics) and each contributes the same denominator.
- `:efron` — the Efron correction: as `:breslow`, plus the `1 − (j−1)/d`
  denominator weights on the tied cases; the better approximation, and what
  `survival::coxph` defaults to. Requires the tied events to be distinct dyads.
- `:batch` — refused: with the history frozen, a simultaneous batch in an ordinal
  likelihood IS the Breslow correction. (It is [`Revel.fit_timing`](@ref)'s
  policy, where there is an exposure interval for a batch to consume.)

On tie-free data all four give the identical fit. `tie_method(fit)` reports what
actually happened and `approximations(fit)` carries the caveat.

# Example
```julia
using Revel, Random
truth = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(truth, [0.8, 0.6], 6, 200; rng=Xoshiro(7))
direct = Revel.fit_obpm(events, truth, 6)
direct.converged                                 # true
coef(direct) == coef(fit_revel(events, truth, 6))  # true — the same estimator
small = Revel.fit_obpm(events, truth, 6; cache=:none)
coef(small) == coef(direct)                      # true — one design matrix in memory
```
"""
function fit_obpm(events::Vector{Event{T}}, statistics::Vector{<:AbstractStatistic},
                  n_actors::Int; ties::Symbol=:error, cache::Symbol=:auto,
                  chunk::Union{Nothing,Int}=nothing,
                  cache_bytes::Int=_DEFAULT_CACHE_BYTES, maxiter::Int=100,
                  tol::Float64=1e-8) where T
    check_tie_policy(ties, _OBPM_TIES_SUPPORTED; model=_OBPM_TIES_MODEL,
                     reasons=_OBPM_TIES_REASONS)
    model = OrdinalBPM(collect(AbstractStatistic, statistics), n_actors)
    p = length(statistics)
    isempty(events) && throw(ArgumentError("no events to fit"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be finite and positive"))

    sorted = sort(events, by=e -> e.time)
    blocks = _tie_blocks(sorted)
    has_ties = any(b -> length(b) > 1, blocks)
    ties === :error && has_ties && _reject_ties(sorted, blocks,
        "The ordinal likelihood is a likelihood over the ORDER of the events, " *
        "and a tie is precisely the statement that the order is unobserved: " *
        "sorting it would invent the very thing being modelled (and would let " *
        "whichever event is placed first enter the statistics of the ones placed " *
        "after it).",
        "Choose a policy explicitly: `ties=:efron` (the Efron correction, the " *
        "best approximation and `survival::coxph`'s default), `ties=:breslow` " *
        "(the Breslow correction), or `ties=:ordered` (arbitrary order, no " *
        "correction).")
    tie_applied = has_ties ? ties : :none

    plan = _risk_set_plan(events, model.statistics, n_actors; ties=ties)
    rs = _risk_sets(plan; cache=cache, chunk=chunk, cache_bytes=cache_bytes)

    # Whether a finite maximum exists is decided on the risk sets themselves,
    # before the optimizer runs (see `_separation_verdict`).
    verdict = _separation_verdict(rs; budget=cache_bytes)
    names = [name(s) for s in model.statistics]
    derivatives = _obpm_derivatives(rs)
    opt = _quietly(verdict.separated) do
        newton_fit(derivatives, zeros(p); maxiter=maxiter, tol=tol)
    end
    warn_separation("fit_obpm", verdict, names)
    opt.converged || verdict.separated || @warn "fit_obpm did not converge; inspect the model for non-identification, or increase maxiter" iterations=opt.iterations
    return OrdinalBPMResult(model, opt.θ, opt.se, opt.loglik,
                           opt.converged && !verdict.separated,
                           length(events), tie_applied, opt.vcov, opt.iterations,
                           verdict, names[verdict.terms])
end
