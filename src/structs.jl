"""
"""
mutable struct Parameters
    sys::HybridSystem
    cost::TrajectoryCost
    igtr::ExplicitIntegrator
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
    f̃s::Vector{Vector{Float64}}
    f̃norm::Float64
    J::Float64
end

function Solution(
    nx::Int,
    nu::Int,
    N::Int
)::Solution
    xs = [zeros(nx) for k = 1:N]
    us = [zeros(nu) for k = 1:(N-1)]
    f̃s = [zeros(nx) for k = 1:N]
    f̃norm = 0.0
    J = 0.0
    return Solution(xs, us, f̃s, f̃norm, J)
end

function Solution(
    params::Parameters
)::Solution
    return Solution(params.sys.nx, params.sys.nu, params.N)
end


"""
"""
mutable struct NullTransition
    val::Union{Transition, Nothing}
end


"""
"""
mutable struct ForwardTerms
    modes::Vector{HybridMode}
    trns::Vector{NullTransition}
    xs::Vector{Vector{Float64}}
    us::Vector{Vector{Float64}}
    f̃s::Vector{Vector{Float64}}
    α::Float64
    ΔJ::Float64
end

function ForwardTerms(
    sys::HybridSystem,
    nx::Int,
    nu::Int,
    N::Int
)::ForwardTerms
    mode = first(values(sys.modes))
    modes = [mode for k = 1:N]
    trns = [NullTransition(nothing) for k = 1:(N-1)]
    xs = [zeros(nx) for k = 1:N]
    us = [zeros(nu) for k = 1:(N-1)]
    f̃s = [zeros(nx) for k = 1:N]
    α = 1.0
    ΔJ = Inf
    return ForwardTerms(modes, trns, xs, us, f̃s, α, ΔJ)
end

function ForwardTerms(
    params::Parameters
)::ForwardTerms
    return ForwardTerms(params.sys, params.sys.nx, params.sys.nu, params.N)
end


"""
"""
mutable struct FlowExpansion
    xx::Matrix{Float64}
    xu::Matrix{Float64}
end

function FlowExpansion(
    nx::Int,
    nu::Int
)::FlowExpansion
    xx = zeros(nx, nx)
    xu = zeros(nx, nu)
    return FlowExpansion(xx, xu)
end


"""
"""
mutable struct CostExpansion
    x::Vector{Float64}
    u::Vector{Float64}
    xx::Matrix{Float64}
    uu::Matrix{Float64}
end

function CostExpansion(
    nx::Int,
    nu::Int
)::CostExpansion
    x = zeros(nx)
    u = zeros(nu)
    xx = zeros(nx, nx)
    uu = zeros(nu, nu)
    return CostExpansion(x, u, xx, uu)
end


"""
"""
mutable struct ValueExpansion
    x::Vector{Float64}
    xx::Matrix{Float64}
end

function ValueExpansion(
    nx::Int
)::ValueExpansion
    x = zeros(nx)
    xx = zeros(nx, nx)
    return ValueExpansion(x, xx)
end


"""
"""
mutable struct ActionValueExpansion
    x::Vector{Float64}
    u::Vector{Float64}
    xx::Matrix{Float64}
    uu::Matrix{Float64}
    xu::Matrix{Float64}
    ux::Matrix{Float64}
end

function ActionValueExpansion(
    nx::Int,
    nu::Int
)::ActionValueExpansion
    x = zeros(nx)
    u = zeros(nu)
    xx = zeros(nx, nx)
    uu = zeros(nu, nu)
    xu = zeros(nx, nu)
    ux = zeros(nu, nx)
    return ActionValueExpansion(x, u, xx, uu, xu, ux)
end


"""
"""
mutable struct BackwardTerms
    F::FlowExpansion
    L::CostExpansion
    V::ValueExpansion
    Q::ActionValueExpansion
    Ks::Vector{VecOrMat{Float64}}
    ds::Vector{Vector{Float64}}
    ΔJ1::Float64
    ΔJ2::Float64
end

function BackwardTerms(
    nx::Int,
    nu::Int,
    N::Int
)::BackwardTerms
    F = FlowExpansion(nx, nu)
    L = CostExpansion(nx, nu)
    V = ValueExpansion(nx)
    Q = ActionValueExpansion(nx, nu)
    Ks = [zeros(nu, nx) for k = 1:(N-1)]
    ds = [zeros(nu) for k = 1:(N-1)]
    ΔJ1 = Inf
    ΔJ2 = Inf
    return BackwardTerms(F, L, V, Q, Ks, ds, ΔJ1, ΔJ2)
end

function BackwardTerms(
    params::Parameters
)::BackwardTerms
    return BackwardTerms(params.sys.nx, params.sys.nu, params.N)
end


"""
"""
mutable struct TemporaryArrays
    x::Vector{Float64}
    u::Vector{Float64}
    iu::Vector{Int}

    xx1::Matrix{Float64}
    xx2::Matrix{Float64}

    uu::Matrix{Float64}
    xu::Matrix{Float64}
    ux::Matrix{Float64}

    xx_hess::DiffResults.DiffResult
    uu_hess::DiffResults.DiffResult
end

function TemporaryArrays(
    nx::Int,
    nu::Int
)::TemporaryArrays
    x = zeros(nx)
    u = zeros(nu)
    iu = zeros(Int, nu)
    xx1 = zeros(nx, nx)
    xx2 = zeros(nx, nx)
    uu = zeros(nu, nu)
    xu = zeros(nx, nu)
    ux = zeros(nu, nx)
    xx_hess = DiffResults.HessianResult(zeros(nx))
    uu_hess = DiffResults.HessianResult(zeros(nu))
    return TemporaryArrays(x, u, iu, xx1, xx2, uu, xu, ux, xx_hess, uu_hess)
end

function TemporaryArrays(
    params::Parameters
)::TemporaryArrays
    return TemporaryArrays(params.sys.nx, params.sys.nu)
end


"""
"""
mutable struct Cache
    fwd::ForwardTerms
    bwd::BackwardTerms
    tmp::TemporaryArrays
end

function Cache(
    params::Parameters
)::Cache
    fwd = ForwardTerms(params)
    bwd = BackwardTerms(params)
    tmp = TemporaryArrays(params)
    return Cache(fwd, bwd, tmp)
end
