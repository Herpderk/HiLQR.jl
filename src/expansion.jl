"""
"""
mutable struct CostExpansion
    Jx::Vector{Float64}
    Ju::Vector{Float64}
    Jxx::Matrix{Float64}
    Juu::Matrix{Float64}
    Jxx_result::DiffResults.DiffResult
    Juu_result::DiffResults.DiffResult
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
        return new(Jx, Ju, Jxx, Juu, Jxx_result, Juu_result)
    end
end

"""
"""
function expand_terminal_cost!(
    Jexp::CostExpansion,
    cost::TrajectoryCost,
    xerr::Vector{Float64}
)::Nothing
    Jexp.Jxx_result = ForwardDiff.hessian!(
        Jexp.Jxx_result, cost.terminal, xerr
    )
    Jexp.Jx = DiffResults.gradient(Jexp.Jxx_result)
    Jexp.Jxx = DiffResults.hessian(Jexp.Jxx_result)
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
    Jexp.Jx = DiffResults.gradient(Jexp.Jxx_result)
    Jexp.Ju = DiffResults.gradient(Jexp.Juu_result)
    Jexp.Jxx = DiffResults.hessian(Jexp.Jxx_result)
    Jexp.Juu = DiffResults.hessian(Jexp.Juu_result)
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
    Qxu::Matrix{Float64}
    Qux::Matrix{Float64}
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
        Qxu = zeros(nx, nu)
        Qux = zeros(nu, nx)
        return new(A, B, V̂x, Vx, Vxx, Qx, Qu, Qxx, Quu, Qxu, Qux)
    end
end

"""
"""
function expand_Q!(
    Qexp::ActionValueExpansion,
    Jexp::CostExpansion,
    f̂::Vector{Float64}
)::Nothing
    Qexp.V̂x = Qexp.Vx + Qexp.Vxx*f̂
    Qexp.Qx = Jexp.Jx + Qexp.A'*Qexp.V̂x
    Qexp.Qu = Jexp.Ju + Qexp.B'*Qexp.V̂x
    Qexp.Qxx = Jexp.Jxx + Qexp.A'*Qexp.Vxx*Qexp.A
    Qexp.Quu = Jexp.Juu + Qexp.B'*Qexp.Vxx*Qexp.B
    Qexp.Qux = Qexp.B' * Qexp.Vxx*Qexp.A
    Qexp.Qxu = Qexp.A' * Qexp.Vxx*Qexp.B
    return nothing
end

"""
"""
function expand_V!(
    Qexp::ActionValueExpansion,
    K::Matrix{Float64},
    d::Vector{Float64}
)::Nothing
    Qexp.Vx = Qexp.Qx - K'*Qexp.Qu + K'*Qexp.Quu*d - Qexp.Qxu*d
    Qexp.Vxx = Qexp.Qxx + K'*Qexp.Quu*K - Qexp.Qxu*K - K'*Qexp.Qux
    return nothing
end
