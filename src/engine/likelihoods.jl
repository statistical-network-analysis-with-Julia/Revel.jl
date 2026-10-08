# =============================================================================
# The two likelihoods: derivative closures for NetworkCore.newton_fit
# =============================================================================
#
# Both are `θ -> (ll, grad, hess)` for the ecosystem's shared optimizer, and both
# run on workspaces allocated ONCE here rather than per interval per Newton
# evaluation. What a single evaluation allocates is the O(p) gradient and O(p²)
# Hessian handed back to the optimizer — and nothing else: no `Xm * θ`, no
# `exp.(η)`, no `Xm' * (w .* Xm)` weighted copy of the risk-set design matrix, no
# per-event outer product. Pinned by an `@allocated` regression test (in the
# suite and in benchmark/regression_tests.jl), which is why they are named
# functions and not closures buried inside the fitters: the test measures the
# code that runs, not a copy of it.

# Ordinal (multinomial partial) likelihood over the full risk set.
function _obpm_derivatives(rs::_RiskSets)
    plan = rs.plan
    p = plan.p
    D = _n_risk_dyads(plan)
    case_idx = plan.case_idx
    η = Vector{Float64}(undef, D)
    probs = Vector{Float64}(undef, D)
    W = Matrix{Float64}(undef, D, p)
    x_exp = Vector{Float64}(undef, p)

    return function (θ)
        ll = Ref(0.0)
        grad = zeros(p)
        hess = zeros(p, p)
        # `twm === nothing` is the unweighted likelihood (every policy but a
        # biting `:efron`). Under Efron the DENOMINATOR carries the tied cases'
        # weights 1 − (j−1)/d; the numerator is always the case's own exp(η).
        _each_interval(rs) do m, Xm, twm
            # Remove a common row before computing moments. In an ordinal
            # likelihood a constant statistic is exactly unidentifiable;
            # E[XX′]-E[X]E[X′] can fabricate information through cancellation.
            @inbounds for k in 1:p, d in 1:D
                W[d,k] = Xm[d,k] - Xm[1,k]
            end
            mul!(η, W, θ)
            ηmax = maximum(η)
            Z = 0.0
            if twm === nothing
                @inbounds for d in 1:D
                    probs[d] = exp(η[d] - ηmax)
                    Z += probs[d]
                end
            else
                @inbounds for d in 1:D
                    probs[d] = twm[d] * exp(η[d] - ηmax)
                    Z += probs[d]
                end
            end
            probs ./= Z

            ll[] += η[case_idx[m]] - ηmax - log(Z)

            mul!(x_exp, transpose(W), probs)
            @inbounds for k in 1:p
                grad[k] += W[case_idx[m], k] - x_exp[k]
            end
            @inbounds for k in 1:p, d in 1:D
                W[d,k] = sqrt(probs[d]) * (W[d,k] - x_exp[k])
            end
            mul!(hess, transpose(W), W, -1.0, 1.0)
        end
        return ll[], grad, hess
    end
end

# Exponential-baseline interval (exact-time) likelihood; β = (log λ₀, θ).
function _timing_derivatives(rs::_RiskSets)
    plan = rs.plan
    p = plan.p
    D = _n_risk_dyads(plan)
    case_idx, waiting = plan.case_idx, plan.waiting
    θbuf = Vector{Float64}(undef, p)
    η = Vector{Float64}(undef, D)
    w = Vector{Float64}(undef, D)
    WX = Matrix{Float64}(undef, D, p)
    Sx = Vector{Float64}(undef, p)
    Sxx = Matrix{Float64}(undef, p, p)

    return function (β)
        logλ = β[1]
        copyto!(θbuf, 1, β, 2, p)

        ll = Ref(0.0)
        grad = zeros(p + 1)
        hess = zeros(p + 1, p + 1)

        _each_interval(rs) do m, Xm, _
            mul!(η, Xm, θbuf)
            Δt = waiting[m]
            # Integrate the JOINT log hazard before exponentiating. Separate
            # exp(logλ)*exp(η) loses finite hazards to Inf*0; including log Δt
            # also preserves exposures when a very short interval cancels a
            # large hazard. A zero interval contributes no exposure at all.
            if Δt == 0
                fill!(w, 0.0)
            else
                logΔt = log(Δt)
                @. w = exp((logλ + η) + logΔt)
            end
            S = sum(w)
            mul!(Sx, transpose(Xm), w)
            WX .= w .* Xm
            mul!(Sxx, transpose(Xm), WX)
            ci = case_idx[m]                # 0 for the right-censored tail
            observed = ci > 0

            # Every interval contributes exposure; only an interval that
            # ends in an event contributes an event term.
            ll[] += (observed ? logλ + η[ci] : 0.0) - S

            grad[1] += (observed ? 1.0 : 0.0) - S
            hess[1, 1] += -S
            @inbounds for k in 1:p
                observed && (grad[k + 1] += Xm[ci, k])
                grad[k + 1] -= Sx[k]
                hess[1, k + 1] += -Sx[k]
                hess[k + 1, 1] += -Sx[k]
                for l in 1:p
                    hess[l + 1, k + 1] -= Sxx[l, k]
                end
            end
        end

        return ll[], grad, hess
    end
end
