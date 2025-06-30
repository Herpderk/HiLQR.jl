"""
"""
function expand_Lterm!(
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    fwd::ForwardTerms,
    params::Parameters
)::Nothing
    # Get terminal x error
    BLAS.copy!(tmp.x, fwd.xs[end])
    BLAS.axpy!(-1.0, params.xrefs[end], tmp.x)

    # Get terminal cost hessian wrt x
    tmp.xx_hess = ForwardDiff.hessian!(
        tmp.xx_hess, params.bwd_cost.terminal, tmp.x
    )

    # Reference terminal value expansion
    V = bwd.Vs[end]

    # Save terminal cost gradient and hessian
    BLAS.copy!(V.x, DiffResults.gradient(tmp.xx_hess))
    BLAS.copy!(V.xx, DiffResults.hessian(tmp.xx_hess))
    return
end

"""
"""
function expand_L!(
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    fwd::ForwardTerms,
    params::Parameters,
    k::Int
)::Nothing
    # Get k-th x and u errors
    BLAS.copy!(tmp.x, fwd.xs[k])
    BLAS.axpy!(-1.0, params.xrefs[k], tmp.x)

    BLAS.copy!(tmp.u, fwd.us[k])
    BLAS.axpy!(-1.0, params.urefs[k], tmp.u)

    # Get gradients and hessians of stage cost wrt x and u
    tmp.xx_hess = ForwardDiff.hessian!(
        tmp.xx_hess, δx -> params.bwd_cost.stage(δx, tmp.u), tmp.x
    )
    tmp.uu_hess = ForwardDiff.hessian!(
        tmp.uu_hess, δu -> params.bwd_cost.stage(tmp.x, δu), tmp.u
    )

    # Reference k-th stage cost expansion
    L = bwd.Ls[k]

    # Save stage cost gradients and hessians wrt x and u
    BLAS.copy!(L.x, DiffResults.gradient(tmp.xx_hess))
    BLAS.copy!(L.xx, DiffResults.hessian(tmp.xx_hess))

    BLAS.copy!(L.u, DiffResults.gradient(tmp.uu_hess))
    BLAS.copy!(L.uu, DiffResults.hessian(tmp.uu_hess))
    return
end

"""
"""
function expand_F!(
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    fwd::ForwardTerms,
    params::Parameters,
    k::Int
)::Nothing
    # Reference k-th flow expansion, state, input, and mode
    F = bwd.Fs[k]
    x = fwd.xs[k]
    u = fwd.us[k]
    mode = fwd.modes[k]

    # Perform salted update if transition is detected
    trn_sym = fwd.trn_syms[k]
    if trn_sym != NULL_TRANSITION
        trn = params.bwd_sys.transitions[trn_sym]
        BLAS.copy!(tmp.xx1, trn.saltation(x, u))

        # Hybrid dynamics jacobian wrt x: salt * Fx
        ForwardDiff.jacobian!(
            tmp.xx2, δx -> params.igtr(mode.flow, δx, u, params.Δt), x
        )
        mul!(F.x, tmp.xx1, tmp.xx2)

        # Hybrid dynamics jacobian wrt u: salt * Fu
        ForwardDiff.jacobian!(
            tmp.xu, δu -> params.igtr(mode.flow, x, δu, params.Δt), u
        )
        mul!(F.u, tmp.xx1, tmp.xu)
    else
        # Dynamics jacobian wrt x
        ForwardDiff.jacobian!(
            F.x, δx -> params.igtr(mode.flow, δx, u, params.Δt), x
        )
        # Dynamics jacobian wrt u
        ForwardDiff.jacobian!(
            F.u, δu -> params.igtr(mode.flow, x, δu, params.Δt), u
        )
    end
    return
end

"""
"""
function expand_Q!(
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    k::Int
)::Nothing
    # Reference k-th expansions
    V = bwd.Vs[k+1]
    L = bwd.Ls[k]
    F = bwd.Fs[k]
    Q = bwd.Qs[k]

    # Action-value gradients
    # Q.x = L.x + F.x'*V.x
    mul!(Q.x, F.x', V.x)
    BLAS.axpy!(1.0, L.x, Q.x)

    # Q.u = L.u + F.u'*V.x
    mul!(Q.u, F.u', V.x)
    BLAS.axpy!(1.0, L.u, Q.u)

    # Action-value hessians
    # Q.xx = L.xx + F.x'*V.xx*F.x
    mul!(tmp.xx1, F.x', V.xx)
    mul!(Q.xx, tmp.xx1, F.x)
    BLAS.axpy!(1.0, L.xx, Q.xx)

    # Q.uu = L.uu + F.u'*V.xx*F.u + μ*I
    mul!(tmp.ux, F.u', V.xx)
    mul!(Q.uu, tmp.ux, F.u)
    BLAS.axpy!(1.0, L.uu, Q.uu)
    BLAS.axpy!(1.0, bwd.μ, Q.uu)

    # Q.xu = F.x'*V.xx*F.u
    mul!(tmp.xx1, F.x', V.xx)
    mul!(Q.xu, tmp.xx1, F.u)

    # Q.ux = F.u'*V.xx*F.x
    mul!(tmp.ux, F.u', V.xx)
    mul!(Q.ux, tmp.ux, F.x)
    return
end

"""
"""
function expand_V!(
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    fwd::ForwardTerms,
    k::Int
)::Nothing
    # Reference k-th value and action-value expansion
    V = bwd.Vs[k]
    Q = bwd.Qs[k]

    # Reference k-th gains and defect
    K = bwd.Ks[k]
    d = bwd.ds[k]
    f̃ = fwd.f̃s[k]

    # Cost-to-go hessian
    # V.xx = Q.xx - K'*Q.ux + K'*Q.uu*K - Q.xu*K
    BLAS.copy!(V.xx, Q.xx)
    mul!(tmp.xx1, K', Q.ux)
    BLAS.axpy!(-1.0, tmp.xx1, V.xx)
    mul!(tmp.xu, K', Q.uu)
    mul!(tmp.xx1, tmp.xu, K)
    BLAS.axpy!(1.0, tmp.xx1, V.xx)
    mul!(tmp.xx1, Q.xu, K)
    BLAS.axpy!(-1.0, tmp.xx1, V.xx)

    # Cost-to-go gradient with defects
    # V.x = V.xx*f̃ + Q.x - K'*u + K'*uu*d - xu*d
    mul!(V.x, V.xx, f̃)
    BLAS.axpy!(1.0, Q.x, V.x)
    mul!(tmp.x, K', Q.u)
    BLAS.axpy!(-1.0, tmp.x, V.x)
    mul!(tmp.xu, K', Q.uu)
    mul!(tmp.x, tmp.xu, d)
    BLAS.axpy!(1.0, tmp.x, V.x)
    mul!(tmp.x, Q.xu, d)
    BLAS.axpy!(-1.0, tmp.x, V.x)
    return
end

"""
"""
function update_gains!(
    bwd::BackwardTerms,
    k::Int
)::Nothing
    # Reference k-th action-value expansion
    Q = bwd.Qs[k]

    # Get sparse LU factorization
    lu!(Q.uu_lu, sparse(Q.uu))

    # Feedforward gains: d = Q.uu \ Q.u
    ldiv!(bwd.ds[k], Q.uu_lu, Q.u)

    # Feedback gains: K = Q.uu \ Q.ux
    ldiv!(bwd.Ks[k], Q.uu_lu, Q.ux)
    return
end

"""
"""
function update_cost_prediction!(
    bwd::BackwardTerms,
    fwd::ForwardTerms,
    k::Int
)::Nothing
    # Reference k-th action-value and value expansion
    Q = bwd.Qs[k]
    V = bwd.Vs[k]

    # Predicted change in cost
    # ΔJ1 += d'*Qu + f̃'*(Vx - Vxx*x)
    bwd.ΔJ1 += bwd.ds[k]'*Q.u + fwd.f̃s[k]'*(V.x - V.xx*fwd.xs[k])

    # ΔJ2 += d'*Qu*d + f̃'*(2*Vxx*x - Vxx*f̃)
    bwd.ΔJ2 += (
        bwd.ds[k]'*Q.uu*bwd.ds[k]
        + fwd.f̃s[k]'*(2*V.xx*fwd.xs[k] - V.xx*fwd.f̃s[k])
    )
    return
end

"""
"""
function backward_pass!(
    cache::Cache,
    params::Parameters
)::Nothing
    # Get references to Cache structs
    fwd = cache.fwd
    bwd = cache.bwd
    tmp = cache.tmp

    # Reset predicted change in cost
    #bwd.ΔJ1 = 0.0
    #bwd.ΔJ2 = 0.0

    # Initialize value expansion
    expand_Lterm!(bwd, tmp, fwd, params)

    # Backward Riccati
    @inbounds for k = (params.N-1) : -1 : 1
        expand_L!(bwd, tmp, fwd, params, k) # Stage cost expansion
        expand_F!(bwd, tmp, fwd, params, k) # Dynamics expansion
        expand_Q!(bwd, tmp, k)              # Action-value expansion
        update_gains!(bwd, k)               # Update feedback and feedforward
        expand_V!(bwd, tmp, fwd, k)         # Value expansion
        #update_cost_prediction!(bwd, fwd, Qs, Vs, k)
    end
    return
end
