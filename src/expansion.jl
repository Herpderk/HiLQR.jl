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
