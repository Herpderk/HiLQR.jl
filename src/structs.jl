"""
    TrajectoryCost(stage_cost, term_cost, nx, nu, N)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
mutable struct TrajectoryCost
    stage::Function
    terminal::Function
    stage_ℓs::Vector{<:DiffFloat64}
    stage_xs::Vector{Vector{<:DiffFloat64}}
    stage_us::Vector{Vector{<:DiffFloat64}}
    term_x::Vector{<:DiffFloat64}

    function TrajectoryCost(
        stage_cost::Function,
        term_cost::Function,
        nx::Int,
        nu::Int,
        N::Int
    )::TrajectoryCost
        stage(
            x::Vector{<:DiffFloat64},
            u::Vector{<:DiffFloat64}
        ) = stage_cost(x, u)::DiffFloat64

        terminal(
            x::Vector{<:DiffFloat64}
        ) = term_cost(x)::DiffFloat64

        stage_ℓs = zeros(N-1)
        term_x = zeros(nx)
        stage_xs = [zeros(nx) for k = 1:(N-1)]
        stage_us = [zeros(nu) for k = 1:(N-1)]
        return new(stage, terminal, stage_ℓs, stage_xs, stage_us, term_x)
    end
end

"""
    cost(xs, us, xrefs, urefs)

Callable struct method for the `TrajectoryCost` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cost::TrajectoryCost)(
    xs::Vector{Vector{Float64}},
    us::Vector{Vector{Float64}},
    xrefs::Vector{Vector{Float64}},
    urefs::Vector{Vector{Float64}}
)::Float64
    # Broadcast stage x - xref
    BLAS.copy!.(cost.stage_xs, (@view xs[1:(end-1)]))
    BLAS.axpy!.(-1.0, (@view xrefs[1:(end-1)]), cost.stage_xs)

    # Broadcast stage u - ref
    BLAS.copy!.(cost.stage_us, (@view us[1:end]))
    BLAS.axpy!.(-1.0, (@view urefs[1:end]), cost.stage_us)

    # Broadcast stage cost
    cost.stage_ℓs .= cost.stage.(cost.stage_xs, cost.stage_us)

    # Get terminal x - xref
    BLAS.copy!(cost.term_x, xs[end])
    BLAS.axpy!(-1.0, xrefs[end], cost.term_x)
    return sum(cost.stage_ℓs) + cost.terminal(cost.term_x)
end


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
    term_cost::Function,
    integrator::ExplicitIntegrator,
    N::Int,
    Δt::Float64,
    xrefs::Vector{Vector{Float64}} =Vector{Float64}[],
    urefs::Vector{Vector{Float64}} = Vector{Float64}[],
    x0::Vector{Float64} = Float64[],
    mI::Symbol = :nothing,
)::Parameters
    cost = TrajectoryCost(stage_cost, term_cost, system.nx, system.nu, N)
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
    f̃s = [zeros(nx) for k = 1:(N-1)]
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
    x::Matrix{Float64}
    u::Matrix{Float64}
end

function FlowExpansion(
    nx::Int,
    nu::Int
)::FlowExpansion
    Fx = zeros(nx, nx)
    Fu = zeros(nx, nu)
    return FlowExpansion(Fx, Fu)
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
    Lx = zeros(nx)
    Lu = zeros(nu)
    Lxx = zeros(nx, nx)
    Luu = zeros(nu, nu)
    return CostExpansion(Lx, Lu, Lxx, Luu)
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
    Vx = zeros(nx)
    Vxx = zeros(nx, nx)
    return ValueExpansion(Vx, Vxx)
end


"""
"""
mutable struct ActionValueExpansion
    x::Vector{Float64}
    u::Vector{Float64}
    xx::Matrix{Float64}
    xu::Matrix{Float64}
    ux::Matrix{Float64}
    uu::Matrix{Float64}
    uu_μ::Matrix{Float64}
    uu_lu::SparseArrays.UMFPACK.UmfpackLU{Float64, Int64}
end

function ActionValueExpansion(
    nx::Int,
    nu::Int
)::ActionValueExpansion
    Qx = zeros(nx)
    Qu = zeros(nu)
    Qxx = zeros(nx, nx)
    Qxu = zeros(nx, nu)
    Qux = zeros(nu, nx)
    Quu = zeros(nu, nu)
    Quu_μ = zeros(nu, nu)
    Quu_lu = lu(sparse(rand(nu, nu)))
    return ActionValueExpansion(Qx, Qu, Qxx, Qxu, Qux, Quu, Quu_μ, Quu_lu)
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
    u2::Vector{Float64}

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
    u2 = zeros(nu)
    xx1 = zeros(nx, nx)
    xx2 = zeros(nx, nx)
    uu = zeros(nu, nu)
    xu = zeros(nx, nu)
    ux = zeros(nu, nx)
    xx_hess = DiffResults.HessianResult(zeros(nx))
    uu_hess = DiffResults.HessianResult(zeros(nu))
    return TemporaryArrays(x, u, u2, xx1, xx2, uu, xu, ux, xx_hess, uu_hess)
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
