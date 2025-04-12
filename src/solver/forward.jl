"""
"""
function nonlinear_rollout!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    sol::Solution,
    params::Parameters
)::Nothing
    mI = params.system.modes[params.mI]

    for k = 1:(params.N-1)
        x = fwd.xs[k]

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

        # Get new input
        # fwd.us[k] = sol.us[k] - fwd.α*bwd.ds[k] - bwd.Ks[k]*(x - sol.xs[k])
        fwd.us[k] = sol.us[k]
        mul!(tmp.u, fwd.α, bwd.ds[k])
        fwd.us[k] -= tmp.u
        tmp.x = x .- sol.xs[k]
        mul!(tmp.u, bwd.Ks[k], tmp.x)
        fwd.us[k] -= tmp.u

        # Roll out next state
        fwd.xs[k+1] = params.integrator(mI.flow, x, fwd.us[k], params.Δt)
        #fwd.xs[k+1] .= -c*fwd.f̂s[k] + params.integrator(
        #    mI.flow, x, fwd.us[k], params.Δt
        #)

        # Compute defects
        fwd.f̂s[k] .= zeros(params.system.nx)
    end
    return nothing
end

"""
"""
function forward_pass!(
    sol::Solution,
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    params::Parameters,
    ls_iter::Int
)::Nothing
    fwd.α = 1.0
    Jls = 0.0

    for i = 1:ls_iter
        nonlinear_rollout!(fwd, bwd, tmp, sol, params)
        Jls = params.cost(params.xrefs, params.urefs, fwd.xs, fwd.us)
        Jls < sol.J ? break : nothing
        fwd.α *= 0.5
    end

    sol.J = Jls
    sol.xs .= fwd.xs
    sol.us .= fwd.us
    sol.f̂s .= fwd.f̂s
    sol.f̂norm = norm(sol.f̂s, Inf)
    return nothing
end

"""
"""
function init_forward_terms!(
    sol::Solution,
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    params::Parameters
)::Nothing
    fwd.xs[1] = params.x0
    sol.xs[1] = params.x0
    sol.J = Inf
    forward_pass!(sol, fwd, bwd, tmp, params, 1)
    return nothing
end
