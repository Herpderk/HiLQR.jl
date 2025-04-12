"""
"""
mutable struct CostExpansion
    Jx::Vector{Float64}
    Ju::Vector{Float64}
    Jxx::Matrix{Float64}
    Juu::Matrix{Float64}
    Jxx_result::DiffResults.DiffResult
    Juu_result::DiffResults.DiffResult
end

function CostExpansion(
    nx::Int,
    nu::Int
)::CostExpansion
    Jx = zeros(nx)
    Ju = zeros(nu)
    Jxx = zeros(nx, nx)
    Juu = zeros(nu, nu)
    Jxx_result = DiffResults.HessianResult(zeros(nx))
    Juu_result = DiffResults.HessianResult(zeros(nu))
    return CostExpansion(Jx, Ju, Jxx, Juu, Jxx_result, Juu_result)
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
mutable struct ActionValueExpansion
    A::Matrix{Float64}
    B::Matrix{Float64}

    V̂x::Vector{Float64}
    Vx::Vector{Float64}
    Vxx::Matrix{Float64}

    Qx::Vector{Float64}
    Qu::Vector{Float64}

    Qxx::Matrix{Float64}
    Quu::Matrix{Float64}
    Quu_reg::Matrix{Float64}

    Qxu::Matrix{Float64}
    Qux::Matrix{Float64}
end

function ActionValueExpansion(
    nx::Int,
    nu::Int
)::ActionValueExpansion
    A = zeros(nx, nx)
    B = zeros(nx, nu)
    V̂x = zeros(nx)
    Vx = zeros(nx)
    Vxx = zeros(nx, nx)
    Qx = zeros(nx)
    Qu = zeros(nu)
    Qxx = zeros(nx, nx)
    Quu = zeros(nu, nu)
    Quu_reg = zeros(nu, nu)
    Qxu = zeros(nx, nu)
    Qux = zeros(nu, nx)
    return ActionValueExpansion(
        A, B,
        V̂x, Vx, Vxx,
        Qx, Qu,
        Qxx, Quu, Quu_reg,
        Qxu, Qux
    )
end

"""
"""
mutable struct TemporaryArrays
    x::Vector{Float64}
    u::Vector{Float64}
    xx::Matrix{Float64}
    uu::Matrix{Float64}
    xu::Matrix{Float64}
    ux::Matrix{Float64}
    lu::SparseArrays.UMFPACK.UmfpackLU{Float64, Int64}
end

function TemporaryArrays(
    nx::Int,
    nu::Int
)::TemporaryArrays
    x = zeros(nx)
    u = zeros(nu)
    xx = zeros(nx, nx)
    uu = zeros(nu, nu)
    xu = zeros(nx, nu)
    ux = zeros(nu, nx)
    lu_val = lu(sparse(I, nu, nu))
    return TemporaryArrays(x, u, xx, uu, xu, ux, lu_val)
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
    f̂::Vector{Float64}
)::Nothing
    # Cost-to-go gradient with defects
    # Qexp.V̂x .= Qexp.Vx + Qexp.Vxx*f̂
    mul!(Qexp.V̂x, Qexp.Vxx, f̂)
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
    mul!(tmp.xx, Qexp.A', Qexp.Vxx)
    mul!(Qexp.Qxx, tmp.xx, Qexp.A)
    Qexp.Qxx .+= Jexp.Jxx

    # Qexp.Quu .= Jexp.Juu + Qexp.B'*Qexp.Vxx*Qexp.B
    mul!(tmp.ux, Qexp.B', Qexp.Vxx)
    mul!(Qexp.Quu, tmp.ux, Qexp.B)
    Qexp.Quu .+= Jexp.Juu

    # Qexp.Qxu .= Qexp.A'*Qexp.Vxx*Qexp.B
    mul!(tmp.xx, Qexp.A', Qexp.Vxx)
    mul!(Qexp.Qxu, tmp.xx, Qexp.B)

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
    mul!(tmp.xx, K', Qexp.Qux)
    Qexp.Vxx .-= tmp.xx
    mul!(tmp.xu, K', Qexp.Quu)
    mul!(tmp.xx, tmp.xu, K)
    Qexp.Vxx .+= tmp.xx
    mul!(tmp.xx, Qexp.Qxu, K)
    Qexp.Vxx .-= tmp.xx
    return nothing
end
