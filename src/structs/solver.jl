"""
"""
mutable struct ProblemParameters
    rev_trns_dict::Dict{Transition, Symbol}
    fwd_sys::HybridSystem
    bwd_sys::Union{Function, HybridSystem}
    fwd_cost::TrajectoryCost
    bwd_cost::TrajectoryCost
    igtr::ExplicitIntegrator
    N::Int
    Δt::Float64
    xrefs::Vector{Vector{Float64}}
    urefs::Vector{Vector{Float64}}
    x0::Vector{Float64}
    mI::Symbol
end

function ProblemParameters(
    fwd_sys::HybridSystem,
    bwd_sys::Union{Function, HybridSystem},
    fwd_stage_cost::Function,
    fwd_term_cost::Function,
    bwd_stage_cost::Function,
    bwd_term_cost::Function,
    integrator::ExplicitIntegrator,
    N::Int,
    Δt::Float64,
    xrefs::Vector{Vector{Float64}} = Vector{Float64}[],
    urefs::Vector{Vector{Float64}} = Vector{Float64}[],
    x0::Vector{Float64} = Float64[],
    mI::Symbol = :nothing
)::ProblemParameters
    # Assert that forward and backward systems have the same dimensions
    if fwd_sys.nx != bwd_sys.nx
        ArgumentError("Forward and backward system state dimensions must match")
    end
    if fwd_sys.nu != bwd_sys.nu
        ArgumentError("Forward and backward input dimensions must match")
    end

    # Assert that forward and backward transitions match
    if typeof(bwd_sys) === HybridSystem
        fwd_trns_symbols = collect(keys(fwd_sys.transitions))
        bwd_trns_symbols = collect(keys(bwd_sys.transitions))
        if fwd_trns_symbols != bwd_trns_symbols
            ArgumentError("Backward system must have only 1 flow or identical transition dict keys as that of the forward system")
        end

        # Generate reverse transitions dict
        rev_trns_dict = Dict(trn => sym for (sym, trn) in bwd_sys.transitions)
    else
        rev_trns_dict = Dict{Transition, Symbol}()
    end

    # Generate forward and backward cost functions
    fwd_cost = TrajectoryCost(
        fwd_stage_cost, fwd_term_cost, fwd_sys.nx, fwd_sys.nu, N
    )
    bwd_cost = TrajectoryCost(
        bwd_stage_cost, bwd_term_cost, bwd_sys.nx, bwd_sys.nu, N
    )
    return ProblemParameters(
        rev_trns_dict,
        fwd_sys,
        bwd_sys,
        fwd_cost,
        bwd_cost,
        integrator,
        N,
        Δt,
        xrefs,
        urefs,
        x0,
        mI
    )
end

function ProblemParameters(
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
)::ProblemParameters
    return ProblemParameters(
        sys,
        sys,
        stage_cost,
        term_cost,
        stage_cost,
        term_cost,
        integrator,
        N,
        Δt,
        xrefs,
        urefs,
        x0,
        mI
    )
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
    params::ProblemParameters
)::Solution
    return Solution(params.fwd_sys.nx, params.fwd_sys.nu, params.N)
end


"""
"""
mutable struct SolverCache
    fwd::ForwardTerms
    bwd::BackwardTerms
    tmp::TemporaryArrays
end

function SolverCache(
    params::ProblemParameters
)::SolverCache
    fwd = ForwardTerms(
        params.fwd_sys, params.fwd_sys.nx, params.fwd_sys.nu, params.N
    )
    bwd = BackwardTerms(params.bwd_sys.nx, params.bwd_sys.nu, params.N)
    tmp = TemporaryArrays(params.bwd_sys.nx, params.bwd_sys.nu)
    return SolverCache(fwd, bwd, tmp)
end


"""
"""
mutable struct SolverOptions
    regularizer::Float64
    max_step::Float64
    defect_rate::Float64
    stat_tol::Float64
    defect_tol::Float64
    max_iter::Int
    max_ls_iter::Int
    multishoot::Bool
    verbose::Bool
end

function SolverOptions(;
    regularizer::Float64 = 1e-6,
    max_step::Float64 = 1.0,
    defect_rate::Float64 = 1.0,
    stat_tol::Float64 = 1e-9,
    defect_tol::Float64 = 1e-9,
    max_iter::Int = 100,
    max_ls_iter::Int = 20,
    multishoot::Bool = false,
    verbose::Bool = true,
)::SolverOptions
    return SolverOptions(
        regularizer,
        max_step,
        defect_rate,
        stat_tol,
        defect_tol,
        max_iter,
        max_ls_iter,
        multishoot,
        verbose
    )
end
