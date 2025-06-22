"""
"""
function expand_Lterm!(
    V::ValueExpansion,
    tmp::TemporaryArrays,
    cost::TrajectoryCost,
    xerr::Vector{Float64}
)::Nothing
    # Get gradient and hessian of terminal cost wrt x
    tmp.xx_hess = ForwardDiff.hessian!(tmp.xx_hess, cost.terminal, xerr)
    copy!(V.x, DiffResults.gradient(tmp.xx_hess))
    copy!(V.xx, DiffResults.hessian(tmp.xx_hess))
    return nothing
end

"""
"""
function expand_L!(
    L::CostExpansion,
    tmp::TemporaryArrays,
    cost::TrajectoryCost,
    xerr::Vector{Float64},
    uerr::Vector{Float64}
)::Nothing
    # Get gradient and hessian of stage cost wrt x
    tmp.xx_hess = ForwardDiff.hessian!(
        tmp.xx_hess, δx -> cost.stage(δx, uerr), xerr
    )
    copy!(L.x, DiffResults.gradient(tmp.xx_hess))
    copy!(L.xx, DiffResults.hessian(tmp.xx_hess))

    # Get gradient and hessian of stage cost wrt u
    tmp.uu_hess = ForwardDiff.hessian!(
        tmp.uu_hess, δu -> cost.stage(xerr, δu), uerr
    )
    copy!(L.u, DiffResults.gradient(tmp.uu_hess))
    copy!(L.uu, DiffResults.hessian(tmp.uu_hess))
    return nothing
end

"""
"""
function expand_F!(
    F::FlowExpansion,
    tmp::TemporaryArrays,
    params::Parameters,
    trn::Union{Transition, Nothing},
    mode::HybridMode,
    x::Vector{Float64},
    u::Vector{Float64}
)::Nothing
    # Perform salted update if transition is detected
    if typeof(trn) == Transition
        copy!(tmp.xx1, trn.saltation(x, u))

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
    axpy!(1.0, L.x, Q.x)

    # Q.u = L.u + F.u'*V.x
    mul!(Q.u, F.u', V.x)
    axpy!(1.0, L.u, Q.u)

    # Action-value hessians
    # Q.xx = L.xx + F.x'*V.xx*F.x
    mul!(tmp.xx1, F.x', V.xx)
    mul!(Q.xx, tmp.xx1, F.x)
    axpy!(1.0, L.xx, Q.xx)

    # Q.uu = L.uu + F.u'*V.xx*F.u + μ*I
    mul!(tmp.ux, F.u', V.xx)
    mul!(Q.uu, tmp.ux, F.u)
    axpy!(1.0, L.uu, Q.uu)
    axpy!(1.0, Q.uu_μ, Q.uu)

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
    K::Matrix{Float64},
    d::Vector{Float64},
    f̃::Vector{Float64}
)::Nothing
    # Cost-to-go hessian
    # V.xx = Q.xx - K'*Q.ux + K'*Q.uu*K - Q.xu*K
    copy!(V.xx, Q.xx)
    mul!(tmp.xx1, K', Q.ux)
    axpy!(-1.0, tmp.xx1, V.xx)
    mul!(tmp.xu, K', Q.uu)
    mul!(tmp.xx1, tmp.xu, K)
    axpy!(1.0, tmp.xx1, V.xx)
    mul!(tmp.xx1, Q.xu, K)
    axpy!(-1.0, tmp.xx1, V.xx)

    # Cost-to-go gradient with defects
    # V.x = V.xx*f̃ + Q.x - K'*u + K'*uu*d - xu*d
    mul!(V.x, V.xx, f̃)
    axpy!(1.0, Q.x, V.x)
    mul!(tmp.x, K', Q.u)
    axpy!(-1.0, tmp.x, V.x)
    mul!(tmp.xu, K', Q.uu)
    mul!(tmp.x, tmp.xu, d)
    axpy!(1.0, tmp.x, V.x)
    mul!(tmp.x, Q.xu, d)
    axpy!(-1.0, tmp.x, V.x)
    return nothing
end

"""
"""
function update_gains!(
    K::VecOrMat{Float64},
    d::Vector{Float64},
    Q::ActionValueExpansion
)::Nothing
    # Get sparse LU factorization
    lu!(Q.uu_lu, sparse(Q.uu))

    # Feedback gains: K = Q.uu \ Q.ux
    ldiv!(K, Q.uu_lu, Q.ux)

    # Feedforward gains: d = Q.uu \ Q.u
    ldiv!(d, Q.uu_lu, Q.u)
    return nothing
end

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
    copy!(tmp.x, fwd.xs[end])
    axpy!(-1.0, params.xrefs[end], tmp.x)
    expand_Lterm!(V, tmp, params.cost, tmp.x)

    for k = (params.N-1) : -1 : 1
        # Get stage x and u errors
        copy!(tmp.x, fwd.xs[k])
        axpy!(-1.0, params.xrefs[k], tmp.x)
        copy!(tmp.u, fwd.us[k])
        axpy!(-1.0, params.urefs[k], tmp.u)

        # Stage cost expansion
        expand_L!(L, tmp, params.cost, tmp.x, tmp.u)

        # Dynamics expansion
        expand_F!(
            F, tmp, params, fwd.trns[k].val, fwd.modes[k], fwd.xs[k], fwd.us[k]
        )

        # Action-value expansion
        expand_Q!(Q, tmp, V, L, F)

        # Get feedback and feedforward
        update_gains!(bwd.Ks[k], bwd.ds[k], Q)

        # Value expansion
        expand_V!(V, tmp, Q, bwd.Ks[k], bwd.ds[k], fwd.f̃s[k])
        #update_cost_prediction!(bwd, fwd, Q, V, k)
    end
    return nothing
end
