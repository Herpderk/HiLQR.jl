"""
"""
function expand_Lterm!(
    V::ValueExpansion,
    tmp::TemporaryArrays,
    fwd::ForwardTerms,
    params::Parameters
)::Nothing
    # Get terminal x error
    BLAS.copy!(tmp.x, fwd.xs[end])
    BLAS.axpy!(-1.0, params.xrefs[end], tmp.x)

    # Get gradient and hessian of terminal cost wrt x
    tmp.xx_hess = ForwardDiff.hessian!(tmp.xx_hess, params.cost.terminal, tmp.x)
    BLAS.copy!(V.x, DiffResults.gradient(tmp.xx_hess))
    BLAS.copy!(V.xx, DiffResults.hessian(tmp.xx_hess))
    return nothing
end

"""
"""
function expand_L!(
    L::CostExpansion,
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

    # Get gradient and hessian of stage cost wrt x
    tmp.xx_hess = ForwardDiff.hessian!(
        tmp.xx_hess, δx -> params.cost.stage(δx, tmp.u), tmp.x
    )
    BLAS.copy!(L.x, DiffResults.gradient(tmp.xx_hess))
    BLAS.copy!(L.xx, DiffResults.hessian(tmp.xx_hess))

    # Get gradient and hessian of stage cost wrt u
    tmp.uu_hess = ForwardDiff.hessian!(
        tmp.uu_hess, δu -> params.cost.stage(tmp.x, δu), tmp.u
    )
    BLAS.copy!(L.u, DiffResults.gradient(tmp.uu_hess))
    BLAS.copy!(L.uu, DiffResults.hessian(tmp.uu_hess))
    return nothing
end

"""
"""
function expand_F!(
    F::FlowExpansion,
    tmp::TemporaryArrays,
    fwd::ForwardTerms,
    params::Parameters,
    k::Int
)::Nothing
    # Reference k-th state, input, and mode
    x = fwd.xs[k]
    u = fwd.us[k]
    mode = fwd.modes[k]

    # Perform salted update if transition is detected
    trn = fwd.trns[k].val
    if typeof(trn) === Transition
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
    return nothing
end

"""
"""
function expand_Q!(
    Q::ActionValueExpansion,
    tmp::TemporaryArrays,
    V::ValueExpansion,
    L::CostExpansion,
    F::FlowExpansion
)::Nothing
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
    BLAS.axpy!(1.0, Q.uu_μ, Q.uu)

    # Q.xu = F.x'*V.xx*F.u
    mul!(tmp.xx1, F.x', V.xx)
    mul!(Q.xu, tmp.xx1, F.u)

    # Q.ux = F.u'*V.xx*F.x
    mul!(tmp.ux, F.u', V.xx)
    mul!(Q.ux, tmp.ux, F.x)
    return nothing
end

"""
"""
function expand_V!(
    V::ValueExpansion,
    tmp::TemporaryArrays,
    Q::ActionValueExpansion,
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    k::Int
)::Nothing
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
    return nothing
end

"""
"""
function update_gains!(
    bwd::BackwardTerms,
    Q::ActionValueExpansion,
    k::Int
)::Nothing
    # Get sparse LU factorization
    lu!(Q.uu_lu, sparse(Q.uu))

    # Feedforward gains: d = Q.uu \ Q.u
    ldiv!(bwd.ds[k], Q.uu_lu, Q.u)

    # Feedback gains: K = Q.uu \ Q.ux
    ldiv!(bwd.Ks[k], Q.uu_lu, Q.ux)
    return nothing
end

"""
"""
function update_cost_prediction!(
    bwd::BackwardTerms,
    fwd::ForwardTerms,
    Q::ActionValueExpansion,
    V::ValueExpansion,
    k::Int
)::Nothing
    # Predicted change in cost
    # ΔJ1 += d'*Qu + f̃'*(Vx - Vxx*x)
    bwd.ΔJ1 += bwd.ds[k]'*Q.u + fwd.f̃s[k]'*(V.x - V.xx*fwd.xs[k])

    # ΔJ2 += d'*Qu*d + f̃'*(2*Vxx*x - Vxx*f̃)
    bwd.ΔJ2 += (
        bwd.ds[k]'*Q.uu*bwd.ds[k]
        + fwd.f̃s[k]'*(2*V.xx*fwd.xs[k] - V.xx*fwd.f̃s[k])
    )
    return nothing
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
    F = bwd.F
    L = bwd.L
    V = bwd.V
    Q = bwd.Q

    # Reset predicted change in cost
    #bwd.ΔJ1 = 0.0
    #bwd.ΔJ2 = 0.0

    # Initialize value expansion
    expand_Lterm!(V, tmp, fwd, params)

    # Backward Riccati
    @inbounds for k = (params.N-1) : -1 : 1
        expand_L!(L, tmp, fwd, params, k)   # Stage cost expansion
        expand_F!(F, tmp, fwd, params, k)   # Dynamics expansion
        expand_Q!(Q, tmp, V, L, F)          # Action-value expansion
        update_gains!(bwd, Q, k)            # Update feedback and feedforward
        expand_V!(V, tmp, Q, fwd, bwd, k)   # Value expansion
        #update_cost_prediction!(bwd, fwd, Q, V, k)
    end
    return nothing
end
