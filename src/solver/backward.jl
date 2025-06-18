"""
"""
function expand_terminal_cost!(
    V::ValueExpansion,
    tmp::TemporaryArrays,
    cost::TrajectoryCost,
    xerr::Vector{Float64}
)::Nothing
    tmp.xx_hess = ForwardDiff.hessian!(tmp.xx_hess, cost.terminal, xerr)
    V.x .= DiffResults.gradient(tmp.xx_hess)
    V.xx .= DiffResults.hessian(tmp.xx_hess)
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
    tmp.xx_hess = ForwardDiff.hessian!(
        tmp.xx_hess, δx -> cost.stage(δx, uerr), xerr
    )
    tmp.uu_hess = ForwardDiff.hessian!(
        tmp.uu_hess, δu -> cost.stage(xerr, δu), uerr
    )
    L.x .= DiffResults.gradient(tmp.xx_hess)
    L.u .= DiffResults.gradient(tmp.uu_hess)
    L.xx .= DiffResults.hessian(tmp.xx_hess)
    L.uu .= DiffResults.hessian(tmp.uu_hess)
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
    if typeof(trn) == Transition
        tmp.xx1 .= trn.saltation(x, u)

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
        ForwardDiff.jacobian!(
            F.x, δx -> params.igtr(mode.flow, δx, u, params.Δt), x
        )
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
    Q.x .+= L.x

    # Q.u = L.u + F.u'*V.x
    mul!(Q.u, F.u', V.x)
    Q.u .+= L.u

    # Action-value hessians
    # Q.xx = L.xx + F.x'*V.xx*F.x
    mul!(tmp.xx1, F.x', V.xx)
    mul!(Q.xx, tmp.xx1, F.x)
    Q.xx .+= L.xx

    # Q.uu = L.uu + F.u'*V.xx*F.u + μ*I
    mul!(tmp.ux, F.u', V.xx)
    mul!(Q.uu, tmp.ux, F.u)
    Q.uu .+= L.uu .+ Q.uu_μ

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
    V.xx .= Q.xx
    mul!(tmp.xx1, K', Q.ux)
    V.xx .-= tmp.xx1
    mul!(tmp.xu, K', Q.uu)
    mul!(tmp.xx1, tmp.xu, K)
    V.xx .+= tmp.xx1
    mul!(tmp.xx1, Q.xu, K)
    V.xx .-= tmp.xx1

    # Cost-to-go gradient with defects
    # V.x = V.xx*f̃ + Q.x - K'*u + K'*uu*d - xu*d
    mul!(V.x, V.xx, f̃)
    V.x .+= Q.x
    mul!(tmp.x, K', Q.u)
    V.x .-= tmp.x
    mul!(tmp.xu, K', Q.uu)
    mul!(tmp.x, tmp.xu, d)
    V.x .+= tmp.x
    mul!(tmp.x, Q.xu, d)
    V.x .-= tmp.x
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
    bwd.ΔJ1 = 0.0
    bwd.ΔJ2 = 0.0

    # Initialize value expansion
    tmp.x .= fwd.xs[end] .- params.xrefs[end]
    expand_terminal_cost!(V, tmp, params.cost, tmp.x)

    @inbounds for k = (params.N-1) : -1 : 1
        tmp.x .= fwd.xs[k] .- params.xrefs[k]
        tmp.u .= fwd.us[k] .- params.urefs[k]

        expand_L!(L, tmp, params.cost, tmp.x, tmp.u)
        expand_F!(
            F, tmp, params,
            fwd.trns[k].val, fwd.modes[k], fwd.xs[k], fwd.us[k]
        )
        expand_Q!(Q, tmp, V, L, F)
        update_gains!(bwd.Ks[k], bwd.ds[k], Q)
        expand_V!(V, tmp, Q, bwd.Ks[k], bwd.ds[k], fwd.f̃s[k])
        update_cost_prediction!(bwd, fwd, Q, V, k)
    end
    return nothing
end
