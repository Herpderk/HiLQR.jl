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
end

"""
"""
function forward_pass!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    params::ProblemParameters;
    max_ls_iter::Int = 10
)::Nothing
end

"""
"""
function backward_pass!(
    bwd::BackwardTerms,
    fwd::ForwardTerms,
    Jexp::CostExpansion,
    Qexp::StateActionExpansion,
    params::ProblemParameters,
    #sequence::Vector{TransitionTiming}, # TODO
)::Nothing
    xerr = fwd.xs[end] - params.xrefs[end]
    uerr = zeros(params.system.nu)
    expand_terminal_cost!(Jexp, params.cost, xerr)
    for k = (params.N-1) : -1 : 1
        xerr = fwd.xs[k] - params.xrefs[k]
        uerr = fwd.us[k] - params.urefs[k]
        expand_stage_cost!(Jexp, params.cost, xerr, uerr)
        differentiate_flow!(Qexp, params, flow, fwd.xs[k], fwd.us[k])
        expand_Q!(Qexp, Jexp, fwd.f̂s[k])
        update_backward_terms!(bwd, Qexp)
        expand_V!(Qexp, bwd.Ks[k], bwd.ds[k])
    end
    return nothing
end

"""
"""
function differentiate_flow!(
    Qexp::StateActionExpansion,
    params::ProblemParameters,
    flow::Function,
    x::Vector{Float64},
    u::Vector{Float64}
)::Nothing
    Qexp.A = ForwardDiff.jacobian(
        δx -> params.integrator(flow, δx, u, params.Δt), x
    )
    Qexp.B = ForwardDiff.jacobian(
        δu -> params.integrator(flow, x, δu, params.Δt), u
    )
    return nothing
end

"""
"""
mutable struct BackwardTerms
    Ks::Vector{Matrix{Float64}}
    ds::Vector{Vector{Float64}}
    ΔJ::Float64
    function BackwardTerms(
        params::ProblemParameters
    )::BackwardTerms
        nx = params.system.nx
        nu = params.system.nu
        N = params.N
        Ks = [zeros(nu,nx) for k = 1:(N-1)]
        ds = [zeros(nu) for k = 1:(N-1)]
        ΔJ = 0.0
        return new(Ks, ds, ΔJ)
    end
end

"""
"""
function update_backward_terms!(
    bwd::BackwardTerms,
    Qexp::StateActionExpansion
)::Nothing
    bwd.Ks[k] = Qexp.Quu \ Qexp.Qux
    bwd.ds[k] = Qexp.Quu \ Qexp.Qu
    bwd.ΔJ += Qexp.Qu' * bwd.ds[k]
    return nothing
end

"""
"""
mutable struct ForwardTerms
    f̂s::Vector{Vector{Float64}}
    xs::Vector{Vector{Float64}}
    us::Vector{Vector{Float64}}
    J::Float64
    α::Float64
    function ForwardTerms(
        params::ProblemParameters
    )::ForwardTerms
        nx = params.system.nx
        nu = params.system.nu
        N = params.N
        f̂s = [zeros(nx) for k = 1:N]
        xs = [zeros(nx) for k = 1:N]
        us = [zeros(nu) for k = 1:(N-1)]
        J = 0.0
        α = 1.0
        return new(f̂s, xs, us, J, α)
    end
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
    fwd.xs[1] = params.x0
    mI = system.modes[params.mI]

    for k = 1:(N-1)
        x, u = fwd.xs[k], fwd.us[k]

        # Reset and update mode if a guard is hit
        for (transition, mJ) in mI.transitions
            if transition.guard(x) <= 0.0
                x = transition.reset(x)
                mI = mJ
                break
            end
        end

        Δt, α = params.Δt, fwd.α
        fwd.f̂s[k] = (1-α) * (params.integrator(mI.flow, x, u, Δt) - x)

        d, K = bwd.ds[k], bwd.Ks[k]
        x̂ = x - (1-α)*fwd.f̂s[k]
        û = u - α*d - K*(x̂-x)
        fwd.xs[k+1] = params.integrator(mI.flow, x̂, û, Δt) - (1-α)*fwd.f̂s[k]
    end

    return nothing
end
