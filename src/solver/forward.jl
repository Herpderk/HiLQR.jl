"""
"""
function nonlinear_rollout!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    sol::Solution,
    params::Parameters
)::Nothing
    @inbounds for k = 1:(params.N-1)
        #fwd.us[k] .= prev_us[k] + fwd.α*bwd.ds[k] + bwd.Ks[k]*(x - prev_xs[k])
        #fwd.f̂s[k] .= params.igtr(mI.flow, x, u, params.Δt) - fwd.xs[k+1]

        #x̂ = x - c*fwd.f̂s[k]

        # Get new input
        # fwd.us[k] = sol.us[k] - fwd.α*bwd.ds[k] - bwd.Ks[k]*(fwd.xs[k] - sol.xs[k])
        fwd.us[k] = sol.us[k]
        mul!(tmp.u, fwd.α, bwd.ds[k])
        fwd.us[k] -= tmp.u
        tmp.x = fwd.xs[k] .- sol.xs[k]
        mul!(tmp.u, bwd.Ks[k], tmp.x)
        fwd.us[k] -= tmp.u

        # Compute defects
        fwd.f̂s[k] = (1-fwd.α) * sol.f̂s[k]

        # Roll out next state
        fwd.xs[k+1] = -fwd.f̂s[k] + params.igtr(
            fwd.modes[k].flow, fwd.xs[k], fwd.us[k], params.Δt
        )

        # Reset and update mode if a guard is hit
        Rflag = false
        for (trn, mJ) in fwd.modes[k].transitions
            if trn.guard(fwd.xs[k+1]) < 0.0
                fwd.xs[k+1] = trn.reset(fwd.xs[k+1])
                fwd.trns[k].val = trn
                fwd.modes[k+1] = mJ
                Rflag = true
                break
            end
        end

        # Do not update mode if a guard is not hit
        if !Rflag
            fwd.trns[k].val = nothing
            fwd.modes[k+1] = fwd.modes[k]
        end
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
    ls_iter::Int,
    αmax::Float64,
    multishoot::Bool
)::Nothing
    fwd.α = multishoot ? clamp(αmax, 0.0, 1.0) : 1.0
    Jls = 0.0

    for i = 1:ls_iter
        nonlinear_rollout!(fwd, bwd, tmp, sol, params)
        Jls = params.cost(params.xrefs, params.urefs, fwd.xs, fwd.us)
        Jls < sol.J ? break : nothing
        fwd.α *= 0.5
    end

    fwd.ΔJ = abs(Jls - sol.J)
    sol.J = Jls
    sol.xs .= fwd.xs
    sol.us .= fwd.us
    sol.f̂s .= fwd.f̂s
    sol.f̂norm = norm(sol.f̂s, Inf)
    return nothing
end

"""
"""
function init_terms!(
    sol::Solution,
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    params::Parameters,
    αmax::Float64,
    multishoot::Bool
)::Nothing
    fwd.modes[1] = params.sys.modes[params.mI]
    fwd.xs[1] = params.x0
    sol.xs[1] = params.x0

    @inbounds @simd for k = 1:(params.N-1)
        fill!(bwd.Ks[k], 0.0)
        fill!(bwd.ds[k], 0.0)
    end

    if multishoot
        # Initialize defects
        sol.J = params.cost(params.xrefs, params.urefs, sol.xs, sol.us)
        @inbounds for k = 1:(params.N-1)
            sol.f̂s[k] .= sol.xs[k+1] - params.igtr(
                fwd.modes[1].flow, sol.xs[k], sol.us[k], params.Δt
            )
        end
    else
        # Roll out with a full newton step
        sol.J = Inf
        @inbounds for k = 1:(params.N-1)
            fill!(sol.f̂s[k], 0.0)
        end
        forward_pass!(sol, fwd, bwd, tmp, params, 1, αmax, false)
    end
    return nothing
end
