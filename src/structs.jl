"""
"""
mutable struct Parameters
    system::HybridSystem
    cost::TrajectoryCost
    integrator::ExplicitIntegrator
    N::Int
    Δt::Float64
    xrefs::Vector{Vector{Float64}}
    urefs::Vector{Vector{Float64}}
    x0::Vector{Float64}
    mI::Symbol
end

function Parameters(
    system::HybridSystem,
    stage_cost::Function,
    terminal_cost::Function,
    integrator::ExplicitIntegrator,
    N::Int,
    Δt::Float64,
    xrefs::Vector{Vector{Float64}} =Vector{Float64}[],
    urefs::Vector{Vector{Float64}} = Vector{Float64}[],
    x0::Vector{Float64} = Float64[],
    mI::Symbol = :nothing,
)::Parameters
    cost = TrajectoryCost(stage_cost, terminal_cost)
    return Parameters(system, cost, integrator, N, Δt, xrefs, urefs, x0, mI)
end

"""
"""
mutable struct Solution
    xs::Vector{Vector{Float64}}
    us::Vector{Vector{Float64}}
    f̂s::Vector{Vector{Float64}}
    f̂norm::Float64
    J::Float64
end

function Solution(
    nx::Int,
    nu::Int,
    N::Int
)::Solution
    xs = [zeros(nx) for k = 1:N]
    us = [zeros(nu) for k = 1:(N-1)]
    f̂s = [zeros(nx) for k = 1:N]
    f̂norm = 0.0
    J = 0.0
    return Solution(xs, us, f̂s, f̂norm, J)
end

function Solution(
    params::Parameters
)::Solution
    return Solution(params.system.nx, params.system.nu, params.N)
end

"""
"""
mutable struct ForwardTerms
    xs::Vector{Vector{Float64}}
    us::Vector{Vector{Float64}}
    f̂s::Vector{Vector{Float64}}
    α::Float64
end

function ForwardTerms(
    nx::Int,
    nu::Int,
    N::Int
)::ForwardTerms
    xs = [zeros(nx) for k = 1:N]
    us = [zeros(nu) for k = 1:(N-1)]
    f̂s = [zeros(nx) for k = 1:N]
    α = 1.0
    return ForwardTerms(xs, us, f̂s, α)
end

"""
"""
mutable struct BackwardTerms
    Ks::Vector{Matrix{Float64}}
    ds::Vector{Vector{Float64}}
    ΔJ::Float64
end

function BackwardTerms(
    nx::Int,
    nu::Int,
    N::Int
)::BackwardTerms
    Ks = [zeros(nu,nx) for k = 1:(N-1)]
    ds = [zeros(nu) for k = 1:(N-1)]
    ΔJ = 0.0
    return BackwardTerms(Ks, ds, ΔJ)
end


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
mutable struct Cache
    fwd::ForwardTerms
    bwd::BackwardTerms
    Jexp::CostExpansion
    Qexp::ActionValueExpansion
    tmp::TemporaryArrays
end

function Cache(
    nx::Int,
    nu::Int,
    N::Int
)::Cache
    fwd = ForwardTerms(nx, nu, N)
    bwd = BackwardTerms(nx, nu, N)
    Jexp = CostExpansion(nx, nu)
    Qexp = ActionValueExpansion(nx, nu)
    tmp = TemporaryArrays(nx, nu)
    return Cache(fwd, bwd, Jexp, Qexp, tmp)
end

function Cache(
    params::Parameters
)::Cache
    return Cache(params.system.nx, params.system.nu, params.N)
end
