# =============================================================================
# The StatsAPI surface of the two full-risk-set results
# =============================================================================
#
# One coefficient order everywhere: the timing baseline is an estimated
# coefficient, first in every vector, matrix, table and parameter count. The
# methods are on the StatsAPI generics (imported by name), so they are the
# `coef`, `coefnames`, … of REM, StatsBase and every other package of the
# ecosystem. `RevelFit` forwards to them.

const _EngineResult = Union{OrdinalBPMResult,TimingModelResult}

coef(result::OrdinalBPMResult) = result.coefficients
coef(result::TimingModelResult) = [result.log_baseline; result.coefficients]

stderror(result::OrdinalBPMResult) = result.std_errors
stderror(result::TimingModelResult) = [result.log_baseline_se; result.std_errors]

# R's `names(coef(fit))`, the labels `coeftable(result).names` holds; a new
# vector on every call
coefnames(result::OrdinalBPMResult) = [name(s) for s in result.model.statistics]
coefnames(result::TimingModelResult) =
    ["log_baseline"; [name(s) for s in result.model.statistics]]

# Full observed-information covariance, in `coef` order. A non-negative-definite
# information yields NaN throughout (with the optimizer's warning), not a
# pseudoinverse.
vcov(result::_EngineResult) = result.var_cov

loglikelihood(result::_EngineResult) = result.loglik

# Observed events; a right-censored tail is exposure, not another event
nobs(result::_EngineResult) = result.n_events

dof(result::_EngineResult) = length(coef(result))

# Compare fits on the same events, window and likelihood only (ordinal and
# interval likelihoods are not comparable)
aic(result::_EngineResult) = -2loglikelihood(result) + 2dof(result)

function aicc(result::_EngineResult)
    n, k = nobs(result), dof(result)
    return n > k + 1 ? aic(result) + 2k * (k + 1) / (n - k - 1) : Inf
end

bic(result::_EngineResult) = -2loglikelihood(result) + log(nobs(result)) * dof(result)

# Wald intervals in `coef` order, on the log-rate scale; all NaN when the fit is
# separated (no finite maximum supports a Wald interval)
function confint(result::_EngineResult; level::Real=0.95)
    0 < level < 1 || throw(ArgumentError("confint: level must lie strictly between 0 and 1"))
    q = quantile(Normal(), 1 - (1 - level) / 2)
    c, se = coef(result), stderror(result)
    result.separation.separated && return fill(NaN, length(c), 2)
    return hcat(c .- q .* se, c .+ q .* se)
end

# Wald p-values from the shared, NaN-aware, floored `z_pvalues`; NaN z and p on a
# separated fit
function coeftable(result::_EngineResult)
    c, se = coef(result), stderror(result)
    z, p = z_pvalues(c, se, result.separation)
    return CoefficientTable(coefnames(result), c, se; z_values=z, p_values=p)
end
