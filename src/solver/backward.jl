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

        # Hybrid dynamics jacobian wrt x: salt * Fxx
        ForwardDiff.jacobian!(
            tmp.xx2, δx -> params.igtr(mode.flow, δx, u, params.Δt), x
        )
        mul!(F.xx, tmp.xx1, tmp.xx2)

        # Hybrid dynamics jacobian wrt u: salt * Fxu
        ForwardDiff.jacobian!(
            tmp.xu, δu -> params.igtr(mode.flow, x, δu, params.Δt), u
        )
        mul!(F.xu, tmp.xx1, tmp.xu)

    else
        ForwardDiff.jacobian!(
            F.xx, δx -> params.igtr(mode.flow, δx, u, params.Δt), x
        )
        ForwardDiff.jacobian!(
            F.xu, δu -> params.igtr(mode.flow, x, δu, params.Δt), u
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
    # Q.x = L.x + F.xx'*V.x
    mul!(Q.x, F.xx', V.x)
    Q.x .+= L.x

    # Q.u = L.u + F.xu'*V.x
    mul!(Q.u, F.xu', V.x)
    Q.u .+= L.u

    # Action-value hessians
    ## Q.xx = L.xx + F.xx'*V.xx*F.xx
    mul!(tmp.xx1, F.xx', V.xx)
    mul!(Q.xx, tmp.xx1, F.xx)
    Q.xx .+= L.xx

    # Q.uu = L.uu + F.xu'*V.xx*F.xu
    mul!(tmp.ux, F.xu', V.xx)
    mul!(Q.uu, tmp.ux, F.xu)
    Q.uu .+= L.uu

    # Q.xu = F.xx'*V.xx*F.xu
    mul!(tmp.xx1, F.xx', V.xx)
    mul!(Q.xu, tmp.xx1, F.xu)

    # Q.ux = F.xu'*V.xx*F.xx
    mul!(tmp.ux, F.xu', V.xx)
    mul!(Q.ux, tmp.ux, F.xx)
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
    # V.xx = Q.xx - K'*ux + K'*uu*K - Q.xu*K
    V.xx .= Q.xx
    mul!(tmp.xx1, K', Q.ux)
    V.xx .-= tmp.xx1
    mul!(tmp.xu, K', Q.uu)
    mul!(tmp.xx1, tmp.xu, K)
    V.xx .+= tmp.xx1
    mul!(tmp.xx1, Q.xu, K)
    V.xx .-= tmp.xx1

    # Cost-to-go gradient with defects
    # x = V.xx*f̃ + x - K'*u + K'*uu*d - xu*d
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
    Q::ActionValueExpansion,
    tmp::TemporaryArrays,
    μ::Float64
)::Nothing
    # Regularized uu: uu += μ*I
    mul!(tmp.uu, μ, I)
    Q.uu .+= tmp.uu

    # Get pivoted LU factorization of regularized uu
    Q.uu, tmp.iu, info = LAPACK.getrf!(Q.uu)

    # Feedback gains: K = Q.uu \ Q.ux
    K .= Q.ux
    LAPACK.getrs!('N', Q.uu, tmp.iu, K)

    # Feedforward gains: d = Q.uu \ Q.u
    d .= Q.u
    LAPACK.getrs!('N', Q.uu, tmp.iu, d)
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
    params::Parameters,
    μ::Float64
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
        update_gains!(bwd.Ks[k], bwd.ds[k], Q, tmp, μ)
        expand_V!(V, tmp, Q, bwd.Ks[k], bwd.ds[k], fwd.f̃s[k])
        update_cost_prediction!(bwd, fwd, Q, V, k)
    end
    return nothing
end
