"""
"""
function expand_dynamics!(
    Qexp::ActionValueExpansion,
    tmp::TemporaryArrays,
    params::Parameters,
    trn::Union{Transition, Nothing},
    mode::HybridMode,
    x::Vector{Float64},
    u::Vector{Float64}
)::Nothing
    if typeof(trn) == Transition
        tmp.xx1 .= trn.saltation(x, u)

        # Hybrid dynamics jacobian wrt x: salt * A
        ForwardDiff.jacobian!(
            tmp.xx2, δx -> params.igtr(mode.flow, δx, u, params.Δt), x
        )
        mul!(Qexp.A, tmp.xx1, tmp.xx2)

        # Hybrid dynamics jacobian wrt u: salt * B
        ForwardDiff.jacobian!(
            tmp.xu, δu -> params.igtr(mode.flow, x, δu, params.Δt), u
        )
        mul!(Qexp.B, tmp.xx1, tmp.xu)

    else
        ForwardDiff.jacobian!(
            Qexp.A, δx -> params.igtr(mode.flow, δx, u, params.Δt), x
        )
        ForwardDiff.jacobian!(
            Qexp.B, δu -> params.igtr(mode.flow, x, δu, params.Δt), u
        )
    end
    return nothing
end

"""
"""
function expand_stage_cost!(
    Jexp::CostExpansion,
    cost::TrajectoryCost,
    xerr::Vector{Float64},
    uerr::Vector{Float64}
)::Nothing
    Jexp.Jxx_result = ForwardDiff.hessian!(
        Jexp.Jxx_result, δx -> cost.stage(δx, uerr), xerr
    )
    Jexp.Juu_result = ForwardDiff.hessian!(
        Jexp.Juu_result, δu -> cost.stage(xerr, δu), uerr
    )
    Jexp.Jx .= DiffResults.gradient(Jexp.Jxx_result)
    Jexp.Ju .= DiffResults.gradient(Jexp.Juu_result)
    Jexp.Jxx .= DiffResults.hessian(Jexp.Jxx_result)
    Jexp.Juu .= DiffResults.hessian(Jexp.Juu_result)
    return nothing
end

"""
"""
function expand_terminal_cost!(
    Qexp::ActionValueExpansion,
    Jexp::CostExpansion,
    cost::TrajectoryCost,
    xerr::Vector{Float64}
)::Nothing
    Jexp.Jxx_result = ForwardDiff.hessian!(Jexp.Jxx_result, cost.terminal, xerr)
    Qexp.Vx .= DiffResults.gradient(Jexp.Jxx_result)
    Qexp.Vxx .= DiffResults.hessian(Jexp.Jxx_result)
    return nothing
end

"""
"""
function expand_Q!(
    Qexp::ActionValueExpansion,
    Jexp::CostExpansion,
    tmp::TemporaryArrays,
    f̃::Vector{Float64}
)::Nothing
    # Cost-to-go gradient with defects
    # Qexp.V̂x .= Qexp.Vx + Qexp.Vxx*f̃
    mul!(Qexp.V̂x, Qexp.Vxx, f̃)
    Qexp.V̂x .+= Qexp.Vx

    # Action-value gradients
    # Qexp.Qx .= Jexp.Jx + Qexp.A'*Qexp.V̂x
    mul!(Qexp.Qx, Qexp.A', Qexp.V̂x)
    Qexp.Qx .+= Jexp.Jx

    # Qexp.Qu .= Jexp.Ju + Qexp.B'*Qexp.V̂x
    mul!(Qexp.Qu, Qexp.B', Qexp.V̂x)
    Qexp.Qu .+= Jexp.Ju

    # Action-value hessians
    ## Qexp.Qxx .= Jexp.Jxx + Qexp.A'*Qexp.Vxx*Qexp.A
    mul!(tmp.xx1, Qexp.A', Qexp.Vxx)
    mul!(Qexp.Qxx, tmp.xx1, Qexp.A)
    Qexp.Qxx .+= Jexp.Jxx

    # Qexp.Quu .= Jexp.Juu + Qexp.B'*Qexp.Vxx*Qexp.B
    mul!(tmp.ux, Qexp.B', Qexp.Vxx)
    mul!(Qexp.Quu, tmp.ux, Qexp.B)
    Qexp.Quu .+= Jexp.Juu

    # Qexp.Qxu .= Qexp.A'*Qexp.Vxx*Qexp.B
    mul!(tmp.xx1, Qexp.A', Qexp.Vxx)
    mul!(Qexp.Qxu, tmp.xx1, Qexp.B)

    # Qexp.Qux .= Qexp.B'*Qexp.Vxx*Qexp.A
    mul!(tmp.ux, Qexp.B', Qexp.Vxx)
    mul!(Qexp.Qux, tmp.ux, Qexp.A)
    return nothing
end

"""
"""
function expand_V!(
    Qexp::ActionValueExpansion,
    tmp::TemporaryArrays,
    K::Matrix{Float64},
    d::Vector{Float64}
)::Nothing
    # Cost-to-go gradient
    # Vx = Qx - K'*Qu + K'*Quu*d - Qxu*d
    Qexp.Vx .= Qexp.Qx
    mul!(tmp.x, K', Qexp.Qu)
    Qexp.Vx .-= tmp.x
    mul!(tmp.xu, K', Qexp.Quu)
    mul!(tmp.x, tmp.xu, d)
    Qexp.Vx .+= tmp.x
    mul!(tmp.x, Qexp.Qxu, d)
    Qexp.Vx .-= tmp.x

    # Cost-to-go hessian
    # Vxx = Qxx - K'*Qux + K'*Quu*K - Qxu*K
    Qexp.Vxx .= Qexp.Qxx
    mul!(tmp.xx1, K', Qexp.Qux)
    Qexp.Vxx .-= tmp.xx1
    mul!(tmp.xu, K', Qexp.Quu)
    mul!(tmp.xx1, tmp.xu, K)
    Qexp.Vxx .+= tmp.xx1
    mul!(tmp.xx1, Qexp.Qxu, K)
    Qexp.Vxx .-= tmp.xx1
    return nothing
end

"""
"""
function update_gains!(
    K::VecOrMat{Float64},
    d::Vector{Float64},
    Qexp::ActionValueExpansion,
    tmp::TemporaryArrays,
    μ::Float64
)::Nothing
    # Regularized Quu: Qexp.Quu_reg = Qexp.Quu + μ*I
    mul!(Qexp.Quu_reg, μ, I)
    Qexp.Quu_reg .+= Qexp.Quu
    #tmp.lu = lu!(Qexp.Quu_reg)
    #tmp.lu = lu(sparse(Qexp.Quu_reg))
    tmp.qr = qr!(Qexp.Quu_reg)

    # Feedback gains: bwd.Ks[k] .= Qexp.Quu_reg \ Qexp.Qux
    ldiv!(K, tmp.qr, Qexp.Qux)

    # Feedforward gains: bwd.ds[k] .= Qexp.Quu_reg \ Qexp.Qu
    ldiv!(d, tmp.qr, Qexp.Qu)
    return nothing
end

"""
"""
function backward_pass!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    params::Parameters,
    μ::Float64
)::Nothing
    # Reference expansion structs
    Jexp = bwd.Jexp
    Qexp = bwd.Qexp

    # Reset predicted change in cost
    bwd.ΔJ = 0.0

    # Initialize value expansion
    tmp.x .= fwd.xs[end] .- params.xrefs[end]
    expand_terminal_cost!(Qexp, Jexp, params.cost, tmp.x)

    @inbounds for k = (params.N-1) : -1 : 1
        tmp.x .= fwd.xs[k] .- params.xrefs[k]
        tmp.u .= fwd.us[k] .- params.urefs[k]
        expand_stage_cost!(Jexp, params.cost, tmp.x, tmp.u)

        expand_dynamics!(
            Qexp, tmp, params,
            fwd.trns[k].val, fwd.modes[k],
            fwd.xs[k], fwd.us[k]
        )

        expand_Q!(Qexp, Jexp, tmp, fwd.f̃s[k])
        update_gains!(bwd.Ks[k], bwd.ds[k], Qexp, tmp, μ)
        expand_V!(Qexp, tmp, bwd.Ks[k], bwd.ds[k])

        # Change in cost
        bwd.ΔJ += Qexp.Qu' * bwd.ds[k]
    end
    return nothing
end
