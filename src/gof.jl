# =============================================================================
# Goodness of fit and diagnostics
# =============================================================================
#
# The reviews agree that REM goodness of fit is unsettled — "there exists no
# general consensus regarding a formal testing paradigm" (Bianchi, Filippi-
# Mazzola, Lomi & Wit 2024) — and the literature offers three families of
# proposals, all of which are here:
#
#   prediction     how highly did the model rank what happened? (Butts 2008's
#                  deviance residuals and classification; the top-k recall of
#                  Meijerink-Bosman, Back, Geukes, Leenders & Mulder 2023)
#                  → event_diagnostics, prediction_summary
#   residuals      cumulative score (martingale-residual) processes (after
#                  Boschi & Wit 2026 and Lin, Wei & Ying 1993) and the score
#                  test for an omitted effect
#                  → score_process_test, score_test
#   simulation     auxiliary statistics of sequences simulated from the fit
#                  (in the spirit of Amati, Lomi & Snijders 2024)
#                  → gof, mechanism_shares, closing_times
#
# plus the specification diagnostic the reviews ask for and no source supplies:
# collinearity among the statistics (`statistic_collinearity`).

function _require_ordinal(fit::RevelFit, what::AbstractString)
    fit.model === :ordinal || throw(ArgumentError(
        "$what is defined for the ordinal (partial-likelihood) model. The " *
        "coefficients of an interval-timing fit do not maximise the partial " *
        "likelihood, so its score process is not centred; refit with " *
        "model=:ordinal to use this diagnostic."))
    return nothing
end

# A diagnostic is defined at the maximum of the likelihood
function _require_converged(fit::RevelFit, what::AbstractString)
    _converged(fit) || throw(ArgumentError(
        "$what needs the maximum of the likelihood, but this fit did not converge " *
        "(`fit.fit.converged == false`), so its coefficients are not estimates and " *
        "the diagnostic's reference distribution does not apply. Look for a " *
        "statistic the data cannot identify (`statistic_collinearity`), or raise " *
        "`maxiter`, and refit."))
    return nothing
end

# A finite test statistic never has a p-value of exactly 0 (the ecosystem's
# `z_pvalues` convention)
_pfloor(p::Float64) = isnan(p) ? p : max(p, floatmin(Float64))

# The tie policy to REBUILD the fit's risk sets with (`:batch` freezes the
# history across a tie block exactly as `:breslow` does)
_rebuild_ties(fit::RevelFit) = fit.ties === :batch ? :breslow : fit.ties

_fit_risk_sets(f, fit::RevelFit; statistics=fit.statistics, cases=fit.cases) =
    each_risk_set(f, fit.events, statistics, fit.n_actors; directed=fit.directed,
                  riskset=fit.riskset, ties=_rebuild_ties(fit), cases=cases)

# Fitted probabilities over a risk set (Efron-weighted where a tie correction
# applies); returns the log of the normalising constant
function _risk_set_probs!(probs::Vector{Float64}, η::Vector{Float64}, v::RiskSetView,
                          θ::Vector{Float64})
    D = length(v.dyads)
    ηmax = -Inf
    @inbounds for d in 1:D
        acc = 0.0
        for k in eachindex(θ)
            acc += θ[k] * v.X[d, k]
        end
        η[d] = acc
        ηmax = max(ηmax, acc)
    end
    @inbounds for d in 1:D
        probs[d] = exp(η[d] - ηmax)
    end
    if v.tie_weight != 1.0
        @inbounds for d in v.tied
            probs[d] *= v.tie_weight
        end
    end
    Z = 0.0
    @inbounds for d in 1:D
        Z += probs[d]
    end
    @inbounds for d in 1:D
        probs[d] /= Z
    end
    return ηmax + log(Z)
end

"""
    event_diagnostics(fit::RevelFit; cases=fit.cases) -> DataFrame

How the fitted model scored each event: one row per event with

- `event_index`, `time`, `sender`, `receiver`, `risk_set_size`;
- `probability` — the fitted probability of the observed dyad among its risk set;
- `rank` — its rank by fitted rate (1 = the model's first choice; ties share the
  average rank);
- `n_above`, `n_tied` — how many dyads the model rated above the observed one,
  and how many it rated equal to it (itself included): the rank under a random
  breaking of ties is uniform on `n_above + 1 … n_above + n_tied`;
- `reciprocal_rank` — the expected `1/rank` under that random tie-breaking;
- `rank_fraction` — `(rank − 1)/(risk set size − 1)`, from 0 (top) to 1;
- `deviance_residual` — `−2 log(probability)`, the deviance residual of Butts
  (2008) and Butts & Marcum (2017);
- `null_residual` — `2 log(risk set size)`, the residual of the model with no
  effects, under which every dyad is equally likely;
- `surprise` — `−log₂(probability)` in bits.

Events with a large deviance residual are the ones the model did not see coming;
read against the sequence they show *where* a specification fails (a phase of
the process, a group of actors), which a single fit index cannot.

`cases` selects the events to score (indices into `fit.events`, a mask or a
predicate). Passing events that were held out of the fit gives **out-of-sample**
prediction: fit on `cases=1:k`, score `cases=(k+1):n`. Each held-out event is
predicted one step ahead from the true history before it — not by simulating
the held-out segment, which is the procedure of Brandenberger (2019).

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [1.0, 0.5], 6, 200; rng=Xoshiro(1))
train = fit_revel(events, stats, 6; cases=1:150)
held_out = event_diagnostics(train; cases=151:200)
size(held_out, 1)                          # 50
all(0 .< held_out.probability .<= 1)       # true
```
"""
function event_diagnostics(fit::RevelFit{F, T}; cases=fit.cases) where {F, T}
    θ = collect(Float64, _effect_coef(fit))
    index = Int[]; time = T[]
    sender = Int[]; receiver = Int[]; size_ = Int[]
    prob = Float64[]; rank = Float64[]; frac = Float64[]; null_size = Float64[]
    above = Int[]; tied_ = Int[]; rr = Float64[]
    η = Float64[]; probs = Float64[]
    _require_converged(fit, "event_diagnostics")
    _fit_risk_sets(fit; cases=cases) do v
        D = length(v.dyads)
        length(η) < D && (resize!(η, D); resize!(probs, D))
        logZ = _risk_set_probs!(probs, η, v, θ)
        ηc = η[v.case]
        greater = 0; equal = 0
        @inbounds for d in 1:D
            greater += η[d] > ηc
            equal += η[d] == ηc
        end
        rk = greater + (equal + 1) / 2
        push!(above, greater); push!(tied_, equal)
        # E[1/rank] with the rank uniform on greater+1 … greater+equal
        push!(rr, sum(1 / r for r in (greater + 1):(greater + equal)) / equal)
        push!(index, v.index); push!(time, v.event.time)
        push!(sender, v.event.sender); push!(receiver, v.event.receiver)
        # The case's own probability. Under the Efron correction the tie weights
        # sit in the denominator only (`logZ`), not on the case's numerator —
        # `probs[v.case]` would carry its weight too.
        push!(size_, D); push!(prob, exp(ηc - logZ)); push!(rank, rk)
        push!(frac, (rk - 1) / (D - 1))
        # The model with no effects gives every dyad the same rate: its
        # probability is 1 over the (Efron-weighted) size of the risk set
        push!(null_size, D - length(v.tied) * (1.0 - v.tie_weight))
    end
    return DataFrame(event_index=index, time=time, sender=sender, receiver=receiver,
                     risk_set_size=size_, probability=prob, rank=rank,
                     n_above=above, n_tied=tied_, reciprocal_rank=rr,
                     rank_fraction=frac, deviance_residual=-2 .* log.(prob),
                     null_residual=2 .* log.(null_size), surprise=-log2.(prob))
end

"""
    prediction_summary(fit::RevelFit; ks=(1, 5, 10), cases=fit.cases) -> NamedTuple

Predictive fit of a relational event model, summarised over events:

- `n_events`;
- `recall` — for each `k` in `ks`, the share of events whose observed dyad was
  among the model's top `k` dyads (an integer `k`) or its top fraction `k` of
  the risk set (a `k` in `(0, 1)`: `0.05` is the top 5 %). The **default
  `ks=(1, 5, 10)` are ranks**, not percentages. `recall` at 1 is the
  classification accuracy of Butts (2008); the recall of the top 5 % is the
  prediction measure of Meijerink-Bosman, Back, Geukes, Leenders & Mulder
  (2023), used in the remstats tutorials. Dyads the model rates equal to the
  observed one are credited as a random breaking of the tie would credit them
  (the expected recall), so an event tied with the whole risk set scores
  `k/D`, not 0. A fraction of a small risk set can hold fewer dyads than a
  larger integer `k`, and so give a lower recall;
- `mean_rank`, `median_rank`, `mean_reciprocal_rank` (the mean expected
  `1/rank` under random tie-breaking), `mean_rank_fraction`;
- `deviance` and `null_deviance` — sums of the residuals of
  [`event_diagnostics`](@ref);
- `pseudo_r2` — `1 − deviance/null_deviance`, the share of the null deviance the
  effects account for (Perry & Wolfe 2013 report their models this way);
- `perplexity` — `2^(mean surprise)`, the effective number of equally likely
  dyads the model leaves.

Computed on the fitted events these are in-sample; pass held-out `cases` for an
honest comparison between specifications.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [1.0, 0.5], 6, 200; rng=Xoshiro(1))
summary = prediction_summary(fit_revel(events, stats, 6); ks=(1, 3, 0.2))
summary.n_events                 # 200
0 < summary.pseudo_r2 < 1        # true
summary.recall                   # share of events in the top 1, top 3, top 20 %
```
"""
function prediction_summary(fit::RevelFit; ks=(1, 5, 10), cases=fit.cases)
    d = event_diagnostics(fit; cases=cases)
    n = size(d, 1)
    n > 0 || throw(ArgumentError("no events to summarise"))
    # the chance that the observed dyad is among the top `cut` when the dyads
    # rated equal to it are ordered at random
    credit(i, cut) = clamp((cut - d.n_above[i]) / d.n_tied[i], 0.0, 1.0)
    recall = Float64[]
    for k in ks
        if k isa Integer
            k >= 1 || throw(ArgumentError("an integer k must be at least 1"))
            push!(recall, sum(i -> credit(i, k), 1:n) / n)
        else
            0 < k < 1 || throw(ArgumentError(
                "a fractional k must lie strictly between 0 and 1, got $k"))
            push!(recall, sum(i -> credit(i, max(1.0, k * d.risk_set_size[i])), 1:n) / n)
        end
    end
    deviance = sum(d.deviance_residual)
    null_deviance = sum(d.null_residual)
    return (n_events=n, ks=collect(ks), recall=recall, mean_rank=mean(d.rank),
            median_rank=median(d.rank), mean_reciprocal_rank=mean(d.reciprocal_rank),
            mean_rank_fraction=mean(d.rank_fraction), deviance=deviance,
            null_deviance=null_deviance, pseudo_r2=1 - deviance / null_deviance,
            perplexity=2.0^mean(d.surprise))
end

# -----------------------------------------------------------------------------
# Score (martingale-residual) processes
# -----------------------------------------------------------------------------

# Per-event score contributions u_m = x_case − E[x] and information matrices
# V_m = Cov(x) under the fitted probabilities
function _score_components(fit::RevelFit, statistics, θ::Vector{Float64})
    p = length(θ)
    U = Vector{Vector{Float64}}()
    V = Vector{Matrix{Float64}}()
    η = Float64[]; probs = Float64[]
    xbar = zeros(p)
    _fit_risk_sets(fit; statistics=statistics) do v
        D = length(v.dyads)
        length(η) < D && (resize!(η, D); resize!(probs, D))
        _risk_set_probs!(probs, η, v, θ)
        fill!(xbar, 0.0)
        @inbounds for k in 1:p, d in 1:D
            xbar[k] += probs[d] * v.X[d, k]
        end
        u = [v.X[v.case, k] - xbar[k] for k in 1:p]
        Vm = zeros(p, p)
        @inbounds for l in 1:p, k in 1:l
            acc = 0.0
            for d in 1:D
                acc += probs[d] * (v.X[d, k] - xbar[k]) * (v.X[d, l] - xbar[l])
            end
            Vm[k, l] = acc
            Vm[l, k] = acc
        end
        push!(U, u); push!(V, Vm)
    end
    return U, V
end

function _require_full_design(fit::RevelFit, what::AbstractString)
    fit.n_controls === nothing || throw(ArgumentError(
        "$what needs the score of the likelihood that was maximised, which for a " *
        "fit with sampled controls (n_controls=$(fit.n_controls)) is the sampled " *
        "one. Refit without `n_controls` to use it."))
    return nothing
end

"""
    score_process_test(fit::RevelFit; n_sim=1000, rng=Random.default_rng(),
                       return_process=false)

Check whether the model describes the whole event sequence equally well, with
the cumulative score process of each effect — the cumulative sum, over events,
of the difference between the observed statistic and its expectation under the
fitted model (the martingale residuals of the model). Under a correctly
specified model each process wanders around zero and returns to it; an effect
that strengthens or fades over the sequence, a wrong memory kernel or
functional form, or an omitted effect makes it drift.

This is the residual-based goodness of fit of Boschi & Wit (2026), in the form
of the score-process test of Lin, Wei & Ying (1993) for the Cox model that a
relational event model is: the supremum of the absolute process is compared
with `n_sim` realisations of its null distribution, generated by the multiplier
(Gaussian) resampling that accounts for the coefficients being estimated. (Of
Boschi & Wit's proposals it implements the processes of the model's own
statistics; the processes of auxiliary statistics and their Cauchy combination
are not implemented.)

Returns a `DataFrame` with one row per effect — `term`, `statistic` (the
supremum of the process standardised by `1/√I_kk`, `I` being the observed
information), `p_value` (resampling), `p_kolmogorov`, `at_event` (where the
supremum occurs) — and a final `GLOBAL` row: the largest of the standardised
suprema, with its p-value from the same resampling (the maximum over effects in
every draw), which accounts for the correlation between the effects. Read
`p_value`. `p_kolmogorov` refers the statistic to the supremum of a Brownian
bridge, which holds in large samples only when information accrues at a
constant rate along the sequence — it does not for cumulative statistics, whose
information grows with the history — so treat it as a rough guide; its GLOBAL
entry is the Bonferroni bound. With `return_process=true` the result is
`(table, process)`, `process` being the `events × effects` matrix of
standardised processes for plotting.

Under tied event times each tied event enters the resampling as its own
increment; with the Efron correction that is an approximation whose calibration
has not been checked by simulation.

**A rejection says the specification is not adequate over the sequence; it does
not say which effect is at fault.** The rows are indexed by effect, but drift
in one effect, an omitted effect or a wrong memory leaks into the process of
every effect correlated with it, so several rows reject together. `at_event`
locates the drift, not its cause; [`fit_moving_window`](@ref) and
[`score_test`](@ref) are the follow-ups. Defined for converged ordinal fits
whose risk sets were enumerated (any `riskset`, `cases` or `directed`); refused
for `model=:timing` and for fits with sampled controls.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [1.0, 0.5], 6, 200; rng=Xoshiro(1))
test = score_process_test(fit_revel(events, stats, 6); n_sim=200, rng=Xoshiro(2))
test.term                          # ["log1p(inertia)", "log1p(reciprocity)", "GLOBAL"]
all(0 .<= test.p_value .<= 1)      # true
```
"""
function score_process_test(fit::RevelFit; n_sim::Int=1000,
                            rng::AbstractRNG=Random.default_rng(),
                            return_process::Bool=false)
    _require_ordinal(fit, "score_process_test")
    _require_full_design(fit, "score_process_test")
    _require_converged(fit, "score_process_test")
    n_sim >= 1 || throw(ArgumentError("n_sim must be at least 1"))
    θ = collect(Float64, coef(fit))
    p = length(θ)
    U, V = _score_components(fit, fit.statistics, θ)
    E = length(U)
    info = sum(V)
    info_inv = _checked_inverse(info,
        "the information matrix of the fit is singular; the score process cannot " *
        "be standardised (check the model with `statistic_collinearity`)")
    # Coordinate k of the score process has variance I_kk(t) under the null, so
    # 1/√I_kk puts it on the Brownian-bridge scale. (√(I⁻¹)_kk is not the same
    # number once the statistics are correlated, and inflates the statistic.)
    scale = [info[k, k] > 0 ? 1 / sqrt(info[k, k]) : 0.0 for k in 1:p]

    # Observed process and the cumulative information it is projected with
    process = Matrix{Float64}(undef, E, p)
    cum = zeros(p)
    for m in 1:E
        cum .+= U[m]
        process[m, :] .= cum .* scale
    end
    observed = [maximum(abs, view(process, :, k)) for k in 1:p]
    at = [_fit_index(fit, argmax(abs.(view(process, :, k)))) for k in 1:p]

    # Lin–Wei–Ying multipliers: Ŵ(m) = Σ_{l≤m} u_l G_l − I(m) I⁻¹ Σ_l u_l G_l
    exceed = zeros(Int, p)
    exceed_max = 0
    max_observed = maximum(observed)
    G = Vector{Float64}(undef, E)
    total = zeros(p); partial = zeros(p); cumV = zeros(p, p); shift = zeros(p)
    for _ in 1:n_sim
        randn!(rng, G)
        fill!(total, 0.0)
        for m in 1:E
            total .+= U[m] .* G[m]
        end
        correction = info_inv * total
        fill!(partial, 0.0); fill!(cumV, 0.0)
        sup = zeros(p)
        for m in 1:E
            partial .+= U[m] .* G[m]
            cumV .+= V[m]
            mul!(shift, cumV, correction)
            for k in 1:p
                sup[k] = max(sup[k], abs((partial[k] - shift[k]) * scale[k]))
            end
        end
        for k in 1:p
            exceed[k] += sup[k] >= observed[k]
        end
        exceed_max += maximum(sup) >= max_observed
    end
    p_value = [(1 + exceed[k]) / (n_sim + 1) for k in 1:p]
    p_kolmogorov = [_pfloor(ccdf(Kolmogorov(), observed[k])) for k in 1:p]

    names = [name(s) for s in fit.statistics]
    table = DataFrame(term=[names; "GLOBAL"],
                      statistic=[observed; max_observed],
                      p_value=[p_value; (1 + exceed_max) / (n_sim + 1)],
                      p_kolmogorov=[p_kolmogorov; min(1.0, p * minimum(p_kolmogorov))],
                      at_event=[at; at[argmax(observed)]])
    return return_process ? (table, process) : table
end

# The inverse of a symmetric positive-definite information matrix, refused when
# it is singular to working precision
function _checked_inverse(A::AbstractMatrix, message::AbstractString)
    vals = eigvals(Symmetric(A))
    (isempty(vals) || minimum(vals) > 1e3 * eps() * max(maximum(vals), floatmin())) ||
        throw(ArgumentError(message))
    return inv(cholesky(Symmetric(A)))
end

# Position in the time-sorted sequence of the fit's `k`-th case
function _fit_index(fit::RevelFit, k::Int)
    fit.cases === nothing && return k
    return findall(fit.cases)[k]
end

"""
    score_test(fit::RevelFit, candidates) -> DataFrame

Rao score tests for effects that are **not** in the model: for each candidate
statistic, would adding it improve the fit? The test needs only the fitted
model — nothing is refitted — so a whole catalogue of candidate effects can be
screened in one pass over the risk sets.

Each row gives the candidate's `term`, its `score` (the sum over events of
observed minus expected statistic, at the fitted coefficients), the score's
`variance` after adjusting for the effects already in the model, the `chisq`
statistic (one degree of freedom), its `p_value`, `direction` — the sign the
candidate's coefficient would take — and `residual_share`, the share of the
candidate's information left after the fitted effects (`1 − R²`). A candidate
that the fitted effects span to working precision has nothing left to test and
gets `NaN`.

This is goodness-of-fit-driven effect discovery, and a way to apply the
hierarchy principle: fit the lower-order terms (degree, repetition) and ask
whether a triadic term still has something to explain, rather than reading a
closure coefficient from a model that omits them. It does not settle the
question: Juozaitienė & Wit (2024) show that "ghost" triadic effects can persist
even next to degree terms when actor heterogeneity is not modelled, which this
package cannot do (no random effects). Candidates are tested one at a time, not
jointly; screening many of them calls for a multiplicity correction. Defined
for converged ordinal fits whose risk sets were enumerated (any `riskset`,
`cases` or `directed`); refused for `model=:timing` and for fits with sampled
controls.

# Example
```julia
using Revel, Random
truth = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(truth, [1.0, 1.0], 6, 300; rng=Xoshiro(1))
small = fit_revel(events, [Inertia(transform=:log1p)], 6)
screen = score_test(small, [Reciprocation(transform=:log1p), OTP(transform=:log1p)])
screen.term                        # ["log1p(reciprocity)", "log1p(otp)"]
screen.p_value[1] < 0.05           # true — reciprocity was left out
```
"""
function score_test(fit::RevelFit, candidates)
    _require_ordinal(fit, "score_test")
    _require_full_design(fit, "score_test")
    _require_converged(fit, "score_test")
    cands = candidates isa AbstractStatistic ? AbstractStatistic[candidates] :
            collect(AbstractStatistic, candidates)
    isempty(cands) && throw(ArgumentError("no candidate statistics to test"))
    all_stats = AbstractStatistic[fit.statistics; cands]
    _stat_names(all_stats)
    p = length(fit.statistics); q = length(cands)
    θ = [collect(Float64, coef(fit)); zeros(q)]
    U, V = _score_components(fit, all_stats, θ)
    score = sum(U)
    info = sum(V)
    Ixx = info[1:p, 1:p]
    Ixx_inv = _checked_inverse(Ixx,
        "the information matrix of the fitted model is singular; the score test " *
        "cannot be adjusted for the effects already in the model")
    # The residual variance is a difference of two numbers of size I_zz; it is
    # zero to working precision when it falls below the rounding error of that
    # difference, which the conditioning of I_xx amplifies
    κ = cond(Symmetric(Ixx))
    term = String[]; s = Float64[]; variance = Float64[]; chisq = Float64[]
    pv = Float64[]; direction = Int[]; share = Float64[]
    for j in 1:q
        z = p + j
        Izx = info[z, 1:p]
        v = info[z, z] - dot(Izx, Ixx_inv * Izx)
        testable = info[z, z] > 0 && v > 1e3 * eps() * κ * info[z, z]
        stat = testable ? score[z]^2 / v : NaN
        push!(term, name(cands[j])); push!(s, score[z]); push!(variance, v)
        push!(chisq, stat); push!(pv, isnan(stat) ? NaN : _pfloor(ccdf(Chisq(1), stat)))
        push!(direction, Int(sign(score[z])))
        push!(share, info[z, z] > 0 ? max(v, 0.0) / info[z, z] : NaN)
    end
    return DataFrame(term=term, score=s, variance=variance, chisq=chisq, p_value=pv,
                     direction=direction, residual_share=share)
end

# -----------------------------------------------------------------------------
# Descriptive metrics of an event sequence
# -----------------------------------------------------------------------------

const _MECHANISMS = (:repetition, :reciprocation, :transitive, :cyclic, :shared_out,
                     :shared_in)

# For event s → r against the dyad clocks `ref` (first or last event time per
# dyad): when was the mechanism's triggering configuration in place? NaN: never.
function _trigger_time(mechanism::Symbol, ref::Dict{Tuple{Int,Int},Float64},
                       out_nb::Dict{Int,Vector{Int}}, in_nb::Dict{Int,Vector{Int}},
                       s::Int, r::Int, latest::Bool)
    mechanism === :repetition && return get(ref, (s, r), NaN)
    mechanism === :reciprocation && return get(ref, (r, s), NaN)
    best = NaN
    # The third actors reachable from the sender's side of the configuration
    thirds = mechanism in (:transitive, :shared_out) ? get(out_nb, s, _NO_NEIGHBORS) :
                                                       get(in_nb, s, _NO_NEIGHBORS)
    for k in thirds
        (k == s || k == r) && continue
        a, b = mechanism === :transitive ? ((s, k), (k, r)) :
               mechanism === :cyclic     ? ((k, s), (r, k)) :
               mechanism === :shared_out ? ((s, k), (r, k)) :
                                           ((k, s), (k, r))
        ta = get(ref, a, NaN); tb = get(ref, b, NaN)
        (isnan(ta) || isnan(tb)) && continue
        done = max(ta, tb)               # the two-path exists once both legs do
        best = isnan(best) ? done : latest ? max(best, done) : min(best, done)
    end
    return best
end

function _walk_mechanisms(f, events::AbstractVector{<:Event}, reference::Symbol,
                          clock::Symbol)
    reference in (:first, :last) || throw(ArgumentError(
        "reference must be :first or :last, got :$reference"))
    clock in (:time, :order) || throw(ArgumentError(
        "clock must be :time or :order, got :$clock"))
    sorted = sort(collect(events); by=e -> e.time)
    ref = Dict{Tuple{Int,Int},Float64}()
    out_nb = Dict{Int,Vector{Int}}(); in_nb = Dict{Int,Vector{Int}}()
    latest = reference === :last
    for (m, e) in enumerate(sorted)
        t = clock === :order ? Float64(m) : _tfloat(e.time)
        f(e, t, ref, out_nb, in_nb, latest)
        key = (e.sender, e.receiver)
        if !haskey(ref, key)
            push!(get!(() -> Int[], out_nb, e.sender), e.receiver)
            push!(get!(() -> Int[], in_nb, e.receiver), e.sender)
            ref[key] = t
        elseif latest
            ref[key] = t
        end
    end
    return nothing
end

"""
    mechanism_shares(events) -> NamedTuple

The share of events in a sequence that, when they happened, **closed** each of
six configurations: `repetition` (the dyad had acted before), `reciprocation`
(the reverse dyad had), `transitive` (an outgoing two-path `s → k → r` existed),
`cyclic` (an incoming two-path `r → k → s`), `shared_out` (both had sent to a
common third) and `shared_in` (both had received from one).

These are descriptive counts, not effects — every share rises with the density
of the accumulated network — but they are the natural auxiliary statistics for
simulation-based goodness of fit: a fitted model should reproduce them
([`gof`](@ref) uses them by default).

# Example
```julia
using Revel
events = [Event(1, 2, 1.0), Event(2, 1, 2.0), Event(1, 2, 3.0), Event(2, 3, 4.0),
          Event(1, 3, 5.0)]
shares = mechanism_shares(events)
shares.repetition       # 0.2 — the third event repeats 1 → 2
shares.reciprocation    # 0.4 — the second and third events answer an earlier one
shares.transitive       # 0.2 — 1 → 3 closes 1 → 2 → 3
```
"""
function mechanism_shares(events::AbstractVector{<:Event})
    isempty(events) && throw(ArgumentError("no events"))
    counts = zeros(Int, length(_MECHANISMS))
    _walk_mechanisms(events, :first, :order) do e, t, ref, out_nb, in_nb, latest
        for (k, mech) in enumerate(_MECHANISMS)
            trigger = _trigger_time(mech, ref, out_nb, in_nb, e.sender, e.receiver, latest)
            isnan(trigger) || (counts[k] += 1)
        end
    end
    return NamedTuple{_MECHANISMS}(Tuple(counts ./ length(events)))
end

"""
    closing_times(events; mechanism=:reciprocation, reference=:last, clock=:time)
        -> Vector{Float64}

For every event that closes `mechanism`, the time elapsed since the
configuration it closes came into being. `mechanism` is one of `:repetition`,
`:reciprocation`, `:transitive`, `:cyclic`, `:shared_out`, `:shared_in`.

`reference=:last` measures from the most recent triggering event (how long did
the answer take?); `reference=:first` from the earliest one. For the triadic
mechanisms a two-path exists from the moment its second leg does, and the
reference is taken over the third actors. `clock=:order` measures the gap in
events instead of clock time.

This is in the spirit of the "internal time" of Amati, Lomi & Snijders (2024)
but is not their definition: they consume an antecedent once it has been
closed, count only the antecedents after the previous `s → r` event, and time a
triad from the earlier of its two legs. (Their internal times are not
implemented.)

A closure effect that operates within minutes and one that operates over months
are different mechanisms, and the statistic alone does not tell them apart; the
distribution of closing times does, and it also guides the choice of a memory
kernel before any model is fitted.

# Example
```julia
using Revel
events = [Event(1, 2, 1.0), Event(2, 1, 4.0), Event(1, 2, 5.0), Event(2, 1, 5.5)]
closing_times(events)                              # [3.0, 1.0, 0.5]
closing_times(events; mechanism=:repetition)       # [4.0, 1.5]
closing_times(events; clock=:order)                # [1.0, 1.0, 1.0]
```
"""
function closing_times(events::AbstractVector{<:Event}; mechanism::Symbol=:reciprocation,
                       reference::Symbol=:last, clock::Symbol=:time)
    mechanism in _MECHANISMS || throw(ArgumentError(
        "mechanism must be one of $(_MECHANISMS), got :$mechanism"))
    gaps = Float64[]
    _walk_mechanisms(events, reference, clock) do e, t, ref, out_nb, in_nb, latest
        trigger = _trigger_time(mechanism, ref, out_nb, in_nb, e.sender, e.receiver, latest)
        isnan(trigger) || push!(gaps, t - trigger)
    end
    return gaps
end

# Gini coefficient of a non-negative vector (0 when it sums to zero)
function _gini(x::Vector{Float64})
    n = length(x)
    total = sum(x)
    (n == 0 || total <= 0) && return 0.0
    sorted = sort(x)
    acc = 0.0
    for (i, v) in enumerate(sorted)
        acc += (2i - n - 1) * v
    end
    return acc / (n * total)
end

# Quartiles of closing times; a sequence in which nothing closes is censored at
# `never` (the length of the sequence in events, beyond any gap it can hold), so
# "never" ranks as the slowest possible, not as instantaneous
function _quartiles(x::Vector{Float64}, never::Float64)
    isempty(x) && return (never, never, never)
    q = quantile(x, (0.25, 0.5, 0.75))
    return (q[1], q[2], q[3])
end

# The shares of undirected events that repeat a pair, and that close a shared
# partner (the pair had a common partner), when they happened
function _undirected_shares(events)
    seen = Set{Tuple{Int,Int}}()
    nb = Dict{Int, Set{Int}}()
    repeat = 0; closure = 0
    for e in sort(collect(events); by=x -> x.time)
        a, b = minmax(e.sender, e.receiver)
        (a, b) in seen && (repeat += 1)
        na = get(nb, a, nothing); nbb = get(nb, b, nothing)
        if na !== nothing && nbb !== nothing &&
           any(k -> k != a && k != b && k in nbb, na)
            closure += 1
        end
        push!(seen, (a, b))
        push!(get!(() -> Set{Int}(), nb, a), b)
        push!(get!(() -> Set{Int}(), nb, b), a)
    end
    return [repeat, closure] ./ length(events)
end

# The default auxiliary statistics of `gof` for undirected events: every
# directed mechanism (reciprocation, cycles) is meaningless there
function _undirected_auxiliary(n_actors::Int)
    shares = ("mechanism shares (undirected)", ["repetition", "shared partner"],
              ev -> _undirected_shares(ev))
    concentration = ("degree concentration (undirected)",
        ["gini(degree)", "distinct pairs / events"],
        function (ev)
            deg = zeros(n_actors)
            pairs = Set{Tuple{Int,Int}}()
            for e in ev
                1 <= e.sender <= n_actors && (deg[e.sender] += 1)
                1 <= e.receiver <= n_actors && (deg[e.receiver] += 1)
                push!(pairs, minmax(e.sender, e.receiver))
            end
            return [_gini(deg), length(pairs) / length(ev)]
        end)
    timing = ("closing times (events)",
        ["repetition q25", "repetition q50", "repetition q75"],
        ev -> collect(_quartiles(closing_times(ev; mechanism=:repetition, clock=:order),
                                 Float64(length(ev)))))
    return [shares, concentration, timing]
end

# The default auxiliary statistics of `gof`: (name, labels, events -> values)
function _default_auxiliary(n_actors::Int)
    shares = ("mechanism shares", [String(m) for m in _MECHANISMS],
              ev -> collect(Float64, values(mechanism_shares(ev))))
    concentration = ("degree concentration",
        ["gini(outdegree)", "gini(indegree)", "distinct dyads / events"],
        function (ev)
            out = zeros(n_actors); inn = zeros(n_actors)
            dyads = Set{Tuple{Int,Int}}()
            for e in ev
                1 <= e.sender <= n_actors && (out[e.sender] += 1)
                1 <= e.receiver <= n_actors && (inn[e.receiver] += 1)
                push!(dyads, (e.sender, e.receiver))
            end
            return [_gini(out), _gini(inn), length(dyads) / length(ev)]
        end)
    timing = ("closing times (events)",
        ["reciprocation q25", "reciprocation q50", "reciprocation q75",
         "repetition q25", "repetition q50", "repetition q75"],
        function (ev)
            E = Float64(length(ev))
            a = _quartiles(closing_times(ev; mechanism=:reciprocation, clock=:order), E)
            b = _quartiles(closing_times(ev; mechanism=:repetition, clock=:order), E)
            return [a..., b...]
        end)
    return [shares, concentration, timing]
end

# For an interval-timing fit: the waiting times, which only that model describes
_waiting_auxiliary() = ("waiting times", ["q25", "q50", "q75", "mean"],
    function (ev)
        gaps = diff([_tfloat(e.time) for e in ev])
        isempty(gaps) && return [0.0, 0.0, 0.0, 0.0]
        q = quantile(gaps, (0.25, 0.5, 0.75))
        return [q[1], q[2], q[3], mean(gaps)]
    end)

# The Monte-Carlo p-value of the Mahalanobis distance of the observed vector
# from the simulations (RSiena's sienaGOF). Every point of the pool — the
# observed vector and the `n` simulated ones — is measured against the mean and
# covariance of the OTHER `n`, so under the model the `n + 1` distances are
# exchangeable and the rank of the observed one is a valid p-value (a distance
# measured against a covariance its own point helped to estimate would be
# systematically smaller, and the test would reject too often). The auxiliaries
# live on different scales (shares, gaps in events), so each is first divided
# by its spread over the pool; one that never varies carries no distance and is
# left out. The covariance is regularised as in Siena.jl's `siena_gof`.
function _mahalanobis_pvalue(observed::Vector{Float64}, sims::Matrix{Float64})
    n = size(sims, 1)
    n >= 2 || return nothing
    pool = vcat(permutedims(observed), sims)
    sd = vec(std(pool; dims=1))
    keep = findall(>(0), sd)
    isempty(keep) && return nothing
    pool = pool[:, keep] ./ permutedims(sd[keep])
    dist = map(1:(n + 1)) do i
        rest = pool[[j for j in 1:(n + 1) if j != i], :]
        center = vec(mean(rest; dims=1))
        C = cov(rest)
        C += (1e-6 + 0.01 * mean(diag(C))) * I
        x = pool[i, :] .- center
        sqrt(max(0.0, dot(x, C \ x)))
    end
    return (1 + count(>=(dist[1]), view(dist, 2:(n + 1)))) / (n + 1)
end

"""
    gof(fit::RevelFit; n_sim=100, rng=Random.default_rng(), auxiliary=nothing)
        -> Networks.GOFResult

Simulation-based goodness of fit: simulate `n_sim` event sequences from the
fitted model and compare auxiliary statistics of the observed sequence with
their simulated distribution. The result is the ecosystem's shared
`Networks.GOFResult` — observed value, simulation envelope and Monte-Carlo
p-value per statistic, and an overall p-value (`p_overall`) from the Mahalanobis
distance of all the auxiliary statistics together, as in RSiena's `sienaGOF` —
so it prints like every other model's `gof`.

An ordinal fit is simulated **conditional on the observed event times** (and
event types and weights), so a memory kernel stated in clock units keeps its
meaning; under `ties=:breslow`/`:efron` the events of one timestamp do not see
each other, as in the fit. An interval-timing fit simulates its own waiting
times from the fitted baseline, starting at the fit's `t0` and conditional on
the number of events (the censored tail after the last event is not
simulated). A receiver-choice fit (`riskset=:sender`) is simulated conditional
on the observed senders.

The default auxiliary statistics are statistics the model was *not* fitted to
match in general: the share of events closing each of six configurations
([`mechanism_shares`](@ref)), the concentration of activity and popularity (Gini
coefficients, distinct dyads per event), and the quartiles of the closing times
of reciprocation and repetition measured in events ([`closing_times`](@ref); a
sequence in which nothing closes is censored at its length). An
interval-timing fit adds the quartiles and mean of the waiting times. For an
undirected fit they are the undirected counterparts — the shares of events
repeating a pair and closing a shared partner, the concentration of degree,
the quartiles of repetition closing times — computed with every pair written
as `(min, max)`, as the simulator writes it. Supply your own as
`auxiliary = [(name, labels, events -> values), …]`.

How to read it:

- It is a **plug-in** parametric bootstrap: the sequences are simulated from
  the estimates, which were fitted to the observed sequence, and are not
  refitted. For auxiliaries close to what the model's statistics already fix,
  the observed value sits near the centre by construction, so the per-statistic
  p-values are conservative and the check has little power against a missing
  effect — [`score_test`](@ref) is the sharper tool for that question.
- The per-statistic p-values are pointwise: with fifteen of them, a few small
  ones are expected by chance. `p_overall` is the one joint test.
- Amati, Lomi & Snijders (2024) use deciles of internal times and a
  Mahalanobis test per set of statistics; the auxiliaries here are different
  statistics, and the joint test spans all of them.

Refused for a fit on a subset of cases (the model does not describe the other
events), a time-varying risk set, a receiver-oriented risk set
(`riskset=:receiver`; the simulator draws receivers given senders only) and a
fit that did not converge.

# Example
```julia
using Revel, Random
stats = [Inertia(transform=:log1p), Reciprocation(transform=:log1p)]
events = simulate_events(stats, [1.0, 0.5], 6, 150; rng=Xoshiro(1))
result = gof(fit_revel(events, stats, 6); n_sim=20, rng=Xoshiro(2))
n_simulations(result)                      # 20
[s.name for s in result.statistics]        # the three auxiliary families
```
"""
function gof(fit::RevelFit; n_sim::Int=100, rng::AbstractRNG=Random.default_rng(),
             auxiliary=nothing)
    n_sim >= 1 || throw(ArgumentError("n_sim must be at least 1"))
    _require_converged(fit, "gof")
    fit.riskset === :receiver && throw(ArgumentError(
        "gof cannot simulate a fit with riskset=:receiver: the simulator draws the " *
        "receiver of each event given its sender, not the sender given the receiver"))
    fit.cases === nothing || throw(ArgumentError(
        "gof simulates whole sequences from the fitted model, but this fit " *
        "describes only a subset of the events (`cases`). Assess it with " *
        "event_diagnostics / prediction_summary instead."))
    fit.riskset isa Function && throw(ArgumentError(
        "gof cannot simulate from a time-varying risk set"))
    aux = auxiliary !== nothing ? auxiliary :
          !fit.directed ? _undirected_auxiliary(fit.n_actors) :
          fit.model === :timing ? [_default_auxiliary(fit.n_actors); _waiting_auxiliary()] :
                                  _default_auxiliary(fit.n_actors)
    E = length(fit.events)
    θ = collect(Float64, _effect_coef(fit))

    riskset = fit.riskset === :active ?
        unique!([_norm_dyad(e.sender, e.receiver, fit.directed) for e in fit.events]) :
        fit.riskset
    common = (rng=rng, directed=fit.directed, riskset=riskset,
              eventtype=[e.eventtype for e in fit.events],
              weights=[e.weight for e in fit.events],
              senders=riskset === :sender ? [e.sender for e in fit.events] : nothing)
    simulate = if fit.model === :timing
        λ0 = exp(coef(fit)[1])
        t0 = fit.t0 === nothing ? 0.0 : _tfloat(fit.t0)
        () -> simulate_events(fit.statistics, θ, fit.n_actors, E; baseline=λ0, t0=t0,
                              common...)
    else
        times = [_tfloat(e.time) for e in fit.events]
        sim_ties = fit.ties in (:breslow, :efron) ? fit.ties : :ordered
        () -> simulate_events(fit.statistics, θ, fit.n_actors, E; times=times,
                              ties=sim_ties, common...)
    end

    # The simulator writes an undirected pair as (min, max); the observed events
    # are put the same way, so no auxiliary can tell them apart by orientation
    observed_events = fit.directed ? fit.events :
        [Event(minmax(e.sender, e.receiver)..., e.time; eventtype=e.eventtype,
               weight=e.weight) for e in fit.events]
    observed = [collect(Float64, f(observed_events)) for (_, _, f) in aux]
    sims = [Matrix{Float64}(undef, n_sim, length(o)) for o in observed]
    for b in 1:n_sim
        ev = simulate()
        for (a, (_, _, f)) in enumerate(aux)
            sims[a][b, :] .= f(ev)
        end
    end
    stats = [GOFStatistic(aux[a][1], aux[a][2], observed[a], sims[a])
             for a in eachindex(aux)]
    label = fit.model === :timing ? "relational event model (interval timing)" :
                                    "relational event model (ordinal)"
    p_overall = _mahalanobis_pvalue(reduce(vcat, observed), reduce(hcat, sims))
    return GOFResult(stats; model=label, p_overall=p_overall)
end

# -----------------------------------------------------------------------------
# Collinearity among statistics
# -----------------------------------------------------------------------------

"""
    statistic_collinearity(events, statistics, n_actors; kwargs...) -> NamedTuple
    statistic_collinearity(fit::RevelFit) -> NamedTuple

How collinear the statistics of a specification are, **within risk sets** —
only differences between the dyads of one risk set carry information:

- `names`;
- `correlation` — the within-risk-set correlation matrix;
- `vif` — variance inflation factors, `1/(1 − R²)` of each statistic regressed
  on the others; `Inf` for a statistic the others span exactly (and only for
  those: a duplicated statistic does not make every other VIF infinite);
- `condition_number` — of the correlation matrix.

The first form weights every dyad of a risk set equally, which is how the
likelihood sees the statistics at θ = 0 and needs no fit — the check to run
while choosing a specification. The second weights them by the fitted
probabilities, which is how the likelihood sees them at the estimate: its VIFs
are `I_jj (I⁻¹)_jj` of the observed information `I`, the factor by which
collinearity inflates each variance. Strong effects concentrate the
probabilities on dyads where the statistics co-vary, so the second is usually
the larger.

Endogenous statistics are built from the same history and overlap by
construction (inertia and out-degree, the four triadic statistics, an
interaction and its main effects), yet the literature offers no diagnostic for
it. A VIF above about 10, or a condition number in the hundreds, says a
coefficient is being estimated from little independent variation and will move
when neighbouring terms enter or leave. A statistic that is constant within
every risk set — a [`GlobalEffect`](@ref), a sender covariate in a
receiver-choice model — has no variation at all and is reported with `NaN`
correlations and an infinite VIF.

Keywords are those of [`each_risk_set`](@ref).

# Example
```julia
using Revel, Random
events = simulate_events([Inertia(transform=:log1p)], [1.0], 6, 200; rng=Xoshiro(1))
check = statistic_collinearity(events, [Inertia(), OutdegreeSender(), OTP()], 6)
check.names                        # ["inertia", "outdegreeSender", "otp"]
size(check.correlation)            # (3, 3)
all(check.vif .>= 1)               # true
dup = statistic_collinearity(events, [Inertia(), Inertia(name="copy"), OTP()], 6)
dup.vif[3] < Inf                   # true — only the duplicated pair is infinite
fitted = statistic_collinearity(fit_revel(events, [Inertia(transform=:log1p), OTP()], 6))
fitted.names                       # ["log1p(inertia)", "otp"]
```
"""
function statistic_collinearity(events::AbstractVector{<:Event}, statistics,
                                n_actors::Int; kwargs...)
    names = _stat_names(statistics)
    p = length(names)
    S = zeros(p, p)
    mean_ = zeros(p)
    each_risk_set(events, statistics, n_actors; kwargs...) do v
        D = length(v.dyads)
        fill!(mean_, 0.0)
        @inbounds for k in 1:p, d in 1:D
            mean_[k] += v.X[d, k]
        end
        mean_ ./= D
        @inbounds for l in 1:p, k in 1:l
            acc = 0.0
            for d in 1:D
                acc += (v.X[d, k] - mean_[k]) * (v.X[d, l] - mean_[l])
            end
            S[k, l] += acc
        end
    end
    for l in 1:p, k in 1:(l - 1)
        S[l, k] = S[k, l]
    end
    return _collinearity(names, S)
end

function statistic_collinearity(fit::RevelFit)
    _require_ordinal(fit, "statistic_collinearity(fit)")
    _require_full_design(fit, "statistic_collinearity(fit)")
    _, V = _score_components(fit, fit.statistics, collect(Float64, coef(fit)))
    return _collinearity([name(s) for s in fit.statistics], sum(V))
end

# Correlations, VIFs and condition number from a (weighted) scatter matrix
function _collinearity(names::Vector{String}, S::Matrix{Float64})
    p = length(names)
    sd = sqrt.(max.(diag(S), 0.0))
    R = [sd[k] > 0 && sd[l] > 0 ? S[k, l] / (sd[k] * sd[l]) : (k == l ? 1.0 : NaN)
         for k in 1:p, l in 1:p]
    varying = findall(>(0), sd)
    vif = fill(Inf, p)
    cond_number = Inf
    if !isempty(varying)
        Rv = Symmetric(R[varying, varying])
        vals = eigvals(Rv)
        # R² of each statistic on the others, by least squares that tolerate an
        # exactly collinear set among the others
        for (a, k) in enumerate(varying)
            others = [b for b in eachindex(varying) if b != a]
            if isempty(others)
                vif[k] = 1.0
                continue
            end
            r = Rv[others, a]
            r2 = dot(r, pinv(Matrix(Rv[others, others]); rtol=sqrt(eps())) * r)
            vif[k] = 1 - r2 > 1e3 * eps() ? 1 / (1 - r2) : Inf
        end
        length(varying) == p &&
            minimum(vals) > 1e3 * eps() * maximum(vals) * p &&
            (cond_number = maximum(vals) / minimum(vals))
    end
    return (names=names, correlation=R, vif=vif, condition_number=cond_number)
end
