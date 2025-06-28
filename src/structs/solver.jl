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
    sys::HybridSystem,
    stage_cost::Function,
    term_cost::Function,
    integrator::ExplicitIntegrator,
    N::Int,
    Δt::Float64,
    xrefs::Vector{Vector{Float64}} =Vector{Float64}[],
    urefs::Vector{Vector{Float64}} = Vector{Float64}[],
    x0::Vector{Float64} = Float64[],
    mI::Symbol = :nothing,
)::Parameters
    cost = TrajectoryCost(stage_cost, term_cost, sys.nx, sys.nu, N)
    return Parameters(sys, cost, integrator, N, Δt, xrefs, urefs, x0, mI)
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
    f̃s = [zeros(nx) for k = 1:(N-1)]
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
mutable struct Cache
    fwd::ForwardTerms
    bwd::BackwardTerms
    tmp::TemporaryArrays
end

function Cache(
    params::Parameters
)::Cache
    fwd = ForwardTerms(params.sys, params.sys.nx, params.sys.nu, params.N)
    bwd = BackwardTerms(params.sys.nx, params.sys.nu, params.N)
    tmp = TemporaryArrays(params.sys.nx, params.sys.nu)
    return Cache(fwd, bwd, tmp)
end
