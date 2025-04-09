"""
"""
mutable struct ProblemParameters
    system::HybridSystem
    cost::TrajectoryCost
    integrator::ExplicitIntegrator
    N::Int
    Δt::Float64
    xrefs::Vector{Vector{Float64}}
    urefs::Vector{Vector{Float64}}
    x0::Vector{Float64}
    mI::Symbol
    function ProblemParameters(
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
    )::ProblemParameters
        cost = TrajectoryCost(stage_cost, terminal_cost)
        return new(system, cost, integrator, N, Δt, xrefs, urefs, x0, mI)
    end
end

"""
"""
mutable struct ForwardTerms
    xs::Vector{Vector{Float64}}
    us::Vector{Vector{Float64}}
    f̂s::Vector{Vector{Float64}}
    f̂norm::Float64
    J::Float64
    α::Float64
    function ForwardTerms(
        nx::Int,
        nu::Int,
        N::Int
    )::ForwardTerms
        xs = [zeros(nx) for k = 1:N]
        us = [zeros(nu) for k = 1:(N-1)]
        f̂s = [zeros(nx) for k = 1:N]
        f̂norm = 0.0
        J = 0.0
        α = 1.0
        return new(xs, us, f̂s, f̂norm, J, α)
    end
end

"""
"""
mutable struct BackwardTerms
    Ks::Vector{Matrix{Float64}}
    ds::Vector{Vector{Float64}}
    ΔJ::Float64
    function BackwardTerms(
        nx::Int,
        nu::Int,
        N::Int
    )::BackwardTerms
        Ks = [zeros(nu,nx) for k = 1:(N-1)]
        ds = [zeros(nu) for k = 1:(N-1)]
        ΔJ = 0.0
        return new(Ks, ds, ΔJ)
    end
end

"""
"""
mutable struct ProblemTerms
    fwd::ForwardTerms
    bwd::BackwardTerms
    Jexp::CostExpansion
    Qexp::ActionValueExpansion
    function ProblemTerms(
        params::ProblemParameters
    )::ProblemTerms
        fwd = ForwardTerms(params.system.nx, params.system.nu, params.N)
        bwd = BackwardTerms(params.system.nx, params.system.nu, params.N)
        Jexp = CostExpansion(params.system.nx, params.system.nu)
        Qexp = ActionValueExpansion(params.system.nx, params.system.nu)
        return new(fwd, bwd, Jexp, Qexp)
    end
end

"""
"""
function update_backward_terms!(
    bwd::BackwardTerms,
    Qexp::ActionValueExpansion,
    k::Int
)::Nothing
    bwd.Ks[k] .= Qexp.Quu \ Qexp.Qux
    bwd.ds[k] .= Qexp.Quu \ Qexp.Qu
    bwd.ΔJ += Qexp.Qu' * bwd.ds[k]
    return nothing
end

"""
"""
function differentiate_flow!(
    Qexp::ActionValueExpansion,
    params::ProblemParameters,
    flow::Function,
    x::Vector{Float64},
    u::Vector{Float64}
)::Nothing
    ForwardDiff.jacobian!(
        Qexp.A, δx -> params.integrator(flow, δx, u, params.Δt), x
    )
    ForwardDiff.jacobian!(
        Qexp.B, δu -> params.integrator(flow, x, δu, params.Δt), u
    )
    return nothing
end

"""
"""
function backward_pass!(
    bwd::BackwardTerms,
    fwd::ForwardTerms,
    Jexp::CostExpansion,
    Qexp::ActionValueExpansion,
    params::ProblemParameters,
    #sequence::Vector{TransitionTiming}, # TODO
)::Nothing
    bwd.ΔJ = 0.0

    xerr = fwd.xs[end] - params.xrefs[end]
    uerr = zeros(params.system.nu)
    expand_terminal_cost!(Qexp, Jexp, params.cost, xerr)

    for k = (params.N-1) : -1 : 1
        xerr = fwd.xs[k] - params.xrefs[k]
        uerr = fwd.us[k] - params.urefs[k]

        expand_stage_cost!(Jexp, params.cost, xerr, uerr)
        differentiate_flow!(Qexp, params, params.system.modes[:nominal].flow, fwd.xs[k], fwd.us[k])
        expand_Q!(Qexp, Jexp, fwd.f̂s[k])
        update_backward_terms!(bwd, Qexp, k)
        expand_V!(Qexp, bwd.Ks[k], bwd.ds[k])
    end
    return nothing
end

"""
"""
function linear_rollout!()
end

"""
"""
function nonlinear_rollout!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    params::ProblemParameters
)::Nothing
    prev_xs = copy(fwd.xs)
    prev_us = copy(fwd.us)
    mI = params.system.modes[params.mI]

    for k = 1:(params.N-1)
        x = fwd.xs[k]

        # Reset and update mode if a guard is hit
        for (transition, mJ) in mI.transitions
            if transition.guard(fwd.xs[k]) <= 0.0
                x = transition.reset(fwd.xs[k])
                mI = mJ
                break
            end
        end

        #fwd.us[k] .= prev_us[k] + fwd.α*bwd.ds[k] + bwd.Ks[k]*(x - prev_xs[k])
        #fwd.f̂s[k] .= zeros(params.system.nx)
        #"""
        #fwd.f̂s[k] .= params.integrator(mI.flow, x, u, params.Δt) - fwd.xs[k+1]

        c = 1 - fwd.α
        x̂ = x - c*fwd.f̂s[k]
        fwd.us[k] .= prev_us[k] - fwd.α * bwd.ds[k] - bwd.Ks[k] * (x̂-x)
        fwd.xs[k+1] .= -c*fwd.f̂s[k] + params.integrator(
            mI.flow, x̂, fwd.us[k], params.Δt
        )
    end
    return nothing
end

"""
"""
function forward_pass!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    params::ProblemParameters;
    max_ls_iter::Int = 10
)::Nothing
    #Jprev = params.cost(params.xrefs, params.urefs, fwd.xs, fwd.us)
    nonlinear_rollout!(fwd, bwd, params)
    fwd.f̂norm = norm(fwd.f̂s, Inf)
    fwd.J = params.cost(params.xrefs, params.urefs, fwd.xs, fwd.us)
    return nothing
end

"""
"""
function log(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    iter::Int
)::Nothing
    if rem(iter - 1, 10) == 0
        @printf "iter      J           ΔJ        |f̂|        α         \n"
        @printf "-----------------------------------------------\n"
    end
    @printf("%3d   %10.3e  %9.2e  %9.2e  %6.4f    \n",
    iter, fwd.J, bwd.ΔJ, fwd.f̂norm, fwd.α)
end

"""
"""
function terminate(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    defect_tol::Float64,
    stat_tol::Float64
)::Bool
    return (fwd.f̂norm < defect_tol) & (bwd.ΔJ < stat_tol) ? true : false
end

"""
"""
function SiLQR_solve!(
    terms::ProblemTerms,
    params::ProblemParameters;
    α::Float64 = 1.0,
    defect_tol::Float64 = 1e-6,
    stat_tol::Float64 = 1e-3,
    max_iter::Int = 100,
    verbose::Bool = true
)::Nothing
    fwd, bwd, Jexp, Qexp = terms.fwd, terms.bwd, terms.Jexp, terms.Qexp
    fwd.α = α
    fwd.xs[1] .= params.x0

    for i = 1:max_iter
        backward_pass!(bwd, fwd, Jexp, Qexp, params)
        forward_pass!(fwd, bwd, params)

        if verbose
            log(fwd, bwd, i)
        end

        if terminate(fwd, bwd, defect_tol, stat_tol)
            println("Optimal solution found!")
            return nothing
        end
    end

    println("Max iterations exceeded!")
    return nothing
end

function SiLQR_solve(
    params::ProblemParameters;
    α::Float64 = 1.0,
    defect_tol::Float64 = 1e-6,
    stat_tol::Float64 = 1e-3,
    max_iter::Int = 100,
    verbose::Bool = true
)::ProblemTerms
    terms = ProblemTerms(params)
    SiLQR_solve!(terms, params; α, defect_tol, stat_tol, max_iter, verbose)
    return terms
end
