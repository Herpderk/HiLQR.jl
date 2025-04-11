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
mutable struct Solution
    xs::Vector{Vector{Float64}}
    us::Vector{Vector{Float64}}
    f̂s::Vector{Vector{Float64}}
    J::Float64
    function Solution(
        nx::Int,
        nu::Int,
        N::Int
    )::Solution
        xs = [zeros(nx) for k = 1:N]
        us = [zeros(nu) for k = 1:(N-1)]
        f̂s = [zeros(nx) for k = 1:N]
        J = 0.0
        return new(xs, us, f̂s, J)
    end
end

"""
"""
mutable struct ForwardTerms
    xs::Vector{Vector{Float64}}
    us::Vector{Vector{Float64}}
    xs_ls::Vector{Vector{Float64}}
    us_ls::Vector{Vector{Float64}}
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
        xs_ls = [zeros(nx) for k = 1:N]
        us_ls = [zeros(nu) for k = 1:(N-1)]
        f̂s = [zeros(nx) for k = 1:N]
        f̂norm = 0.0
        J = 0.0
        α = 1.0
        return new(xs, us, xs_ls, us_ls, f̂s, f̂norm, J, α)
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
    sol::Solution
    fwd::ForwardTerms
    bwd::BackwardTerms
    Jexp::CostExpansion
    Qexp::ActionValueExpansion
    function ProblemTerms(
        params::ProblemParameters
    )::ProblemTerms
        dims = (params.system.nx, params.system.nu, params.N)
        sol = Solution(dims...)
        fwd = ForwardTerms(dims...)
        bwd = BackwardTerms(dims...)
        Jexp = CostExpansion(dims[1:2]...)
        Qexp = ActionValueExpansion(dims[1:2]...)
        return new(sol, fwd, bwd, Jexp, Qexp)
    end
end

"""
"""
function get_flow_jacobians!(
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
function update_backward_terms!(
    bwd::BackwardTerms,
    Qexp::ActionValueExpansion,
    k::Int
)::Nothing
    Quu_reg = Qexp.Quu + 1e-6*I
    try
        bwd.Ks[k] .= Quu_reg \ Qexp.Qux
        bwd.ds[k] .= Quu_reg \ Qexp.Qu
    catch e
        @show Quu_reg
        error()
    end
    bwd.ΔJ += Qexp.Qu' * bwd.ds[k]
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
        get_flow_jacobians!(Qexp, params, params.system.modes[:nominal].flow, fwd.xs[k], fwd.us[k])
        xerr .= fwd.xs[k] - params.xrefs[k]
        uerr .= fwd.us[k] - params.urefs[k]
        expand_stage_cost!(Jexp, params.cost, xerr, uerr)
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
    mI = params.system.modes[params.mI]

    for k = 1:(params.N-1)
        x = fwd.xs_ls[k]

        # Reset and update mode if a guard is hit
        for (transition, mJ) in mI.transitions
            if transition.guard(x) <= 0.0
                x = transition.reset(x)
                mI = mJ
                break
            end
        end

        #fwd.us[k] .= prev_us[k] + fwd.α*bwd.ds[k] + bwd.Ks[k]*(x - prev_xs[k])
        #fwd.f̂s[k] .= params.integrator(mI.flow, x, u, params.Δt) - fwd.xs[k+1]

        #c = 1 - fwd.α
        #x̂ = x - c*fwd.f̂s[k]
        fwd.us_ls[k] = fwd.us[k] - fwd.α*bwd.ds[k] - bwd.Ks[k]*(x - fwd.xs[k])
        fwd.xs_ls[k+1] = params.integrator(mI.flow, x, fwd.us_ls[k], params.Δt)
        fwd.f̂s[k] .= zeros(params.system.nx)
        #fwd.xs[k+1] .= -c*fwd.f̂s[k] + params.integrator(
        #    mI.flow, x, fwd.us[k], params.Δt
        #)
    end
    return nothing
end

"""
"""
function forward_pass!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    params::ProblemParameters,
    ls_iter::Int
)::Nothing
    fwd.α = 1.0
    J_ls = 0.0

    for i = 1:ls_iter
        nonlinear_rollout!(fwd, bwd, params)
        J_ls = params.cost(params.xrefs, params.urefs, fwd.xs_ls, fwd.us_ls)
        J_ls < fwd.J ? break : nothing
        fwd.α *= 0.5
    end

    fwd.J = J_ls
    fwd.xs .= fwd.xs_ls
    fwd.us .= fwd.us_ls
    fwd.f̂norm = norm(fwd.f̂s, Inf)
    return nothing
end

"""
"""
function log(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    iter::Int
)::Nothing
    if rem(iter-1, 20) == 0
        println("\niter       J          ΔJ        |f̂|        α")
        println("------------------------------------------------")
    end
    @printf(
        "%3d    %9.2e  %9.2e  %9.2e   %6.4f\n",
        iter, fwd.J, bwd.ΔJ, fwd.f̂norm, fwd.α
    )
end

"""
"""
function terminate(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    defect_tol::Float64,
    stat_tol::Float64
)::Bool
    return (fwd.f̂norm < defect_tol) && (bwd.ΔJ < stat_tol) ? true : false
end

"""
"""
function inner_solve!(
    terms::ProblemTerms,
    params::ProblemParameters,
    defect_tol::Float64,
    stat_tol::Float64,
    max_iter::Int,
    max_ls_iter::Int,
    verbose::Bool
)::Nothing
    fwd = terms.fwd
    bwd = terms.bwd
    Jexp = terms.Jexp
    Qexp = terms.Qexp

    terms.sol.xs = fwd.xs
    terms.sol.us = fwd.us
    terms.sol.f̂s = fwd.f̂s
    terms.sol.J = fwd.J

    fwd.xs[1] .= params.x0
    fwd.xs_ls[1] .= params.x0
    fwd.J = Inf
    forward_pass!(fwd, bwd, params, 1)

    for i = 1:max_iter
        backward_pass!(bwd, fwd, Jexp, Qexp, params)
        forward_pass!(fwd, bwd, params, max_ls_iter)
        verbose ? log(fwd, bwd, i) : nothing

        if terminate(fwd, bwd, defect_tol, stat_tol)
            verbose ? println("Optimal solution found!") : nothing
            return nothing
        end
    end

    println("Maximum iterations exceeded!")
    return nothing
end

function SiLQR_solve!(
    terms::ProblemTerms,
    params::ProblemParameters;
    defect_tol::Float64 = 1e-6,
    stat_tol::Float64 = 1e-4,
    max_iter::Int = 100,
    max_ls_iter::Int = 10,
    verbose::Bool = true
)::Nothing
    inner_solve!(
        terms, params,
        defect_tol, stat_tol,
        max_iter, max_ls_iter,
        verbose
    )
    return nothing
end

function SiLQR_solve(
    params::ProblemParameters;
    defect_tol::Float64 = 1e-6,
    stat_tol::Float64 = 1e-4,
    max_iter::Int = 100,
    max_ls_iter::Int = 10,
    verbose::Bool = true
)::Solution
    terms = ProblemTerms(params)
    inner_solve!(
        terms, params,
        defect_tol, stat_tol,
        max_iter, max_ls_iter,
        verbose
    )
    return terms.sol
end
