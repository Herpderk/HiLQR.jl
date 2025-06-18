"""
"""
function nonlinear_rollout!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    sol::Solution,
    params::Parameters
)::Nothing
    # Defect closure rate
    c = 1.0 - fwd.α

    # Forward roll-out
    @inbounds for k = 1:(params.N-1)
        # Get new input
        # fwd.us[k] = sol.us[k] - fwd.α*bwd.ds[k] - bwd.Ks[k]*(fwd.xs[k] - sol.xs[k])
        fwd.us[k] .= sol.us[k]
        mul!(tmp.u, fwd.α, bwd.ds[k])
        fwd.us[k] -= tmp.u
        tmp.x .= fwd.xs[k] .- sol.xs[k]
        mul!(tmp.u, bwd.Ks[k], tmp.x)
        fwd.us[k] .-= tmp.u

        # Integrate smooth dynamics
        tmp.x .= params.igtr(fwd.modes[k].flow, fwd.xs[k], fwd.us[k], params.Δt)

        # Reset and update mode if a guard is hit
        Rflag = false
        for (trn, mJ) in fwd.modes[k].transitions
            if trn.guard(tmp.x) < 0.0
                tmp.x .= trn.reset(tmp.x)
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

        # Compute defects and roll out next state
        mul!(fwd.f̃s[k], sol.f̃s[k], c)
        fwd.xs[k+1] = tmp.x - fwd.f̃s[k]
    end
    return nothing
end

"""
"""
function forward_pass!(
    sol::Solution,
    cache::Cache,
    params::Parameters,
    max_step::Float64,
    ls_iter::Int
)::Nothing
    # Get references to Cache structs
    fwd = cache.fwd
    bwd = cache.bwd
    tmp = cache.tmp

    # Initialize line search step size and trajectory cost
    fwd.α = clamp(max_step, 0.0, 1.0)
    Jls = 0.0

    for i = 1:ls_iter
        nonlinear_rollout!(fwd, bwd, tmp, sol, params)
        Jls = params.cost(params.xrefs, params.urefs, fwd.xs, fwd.us)
        Jls < sol.J ? break : nothing
        #=
        ΔJ_actual = Jls - sol.J
        ΔJ_pred = bwd.ΔJ1*fwd.α + 0.5*bwd.ΔJ2*fwd.α^2
        #Jls < sol.J ? break : nothing
        #Jls < sol.J - 1e-2*fwd.α*bwd.ΔJ ? break : nothing

        if ΔJ_pred <= 0.0
            ΔJ_actual < 0.1*ΔJ_pred ? break : nothing
        else
            ΔJ_actual < 2.0*ΔJ_pred ? break : nothing
        end
        =#
        fwd.α *= 0.5
    end

    fwd.ΔJ = abs(Jls - sol.J)
    sol.J = Jls
    sol.xs .= fwd.xs
    sol.us .= fwd.us
    sol.f̃s .= fwd.f̃s
    sol.f̃norm = norm(sol.f̃s, Inf)
    #@show norm.(sol.f̃s)
    return nothing
end

"""
"""
function init_terms!(
    sol::Solution,
    cache::Cache,
    params::Parameters,
    regularizer::Float64,
    multishoot::Bool
)::Nothing
    # Get references to Cache structs
    fwd = cache.fwd
    bwd = cache.bwd

    # Set regularizer
    mul!(bwd.Q.uu_μ, regularizer, I)

    # Set initial conditions
    fwd.modes[1] = params.sys.modes[params.mI]
    fwd.xs[1] = params.x0
    sol.xs[1] = params.x0

    # Initialize gains
    @inbounds @simd for k = 1:(params.N-1)
        fill!(bwd.Ks[k], 0.0)
        fill!(bwd.ds[k], 0.0)
    end

    if multishoot
        # Initialize defects
        sol.J = params.cost(params.xrefs, params.urefs, sol.xs, sol.us)
        @inbounds for k = 1:(params.N-1)
            sol.f̃s[k] .= -sol.xs[k+1] + params.igtr(
                fwd.modes[1].flow, sol.xs[k], sol.us[k], params.Δt
            )
        end
        #forward_pass!(sol, cache, params, 0.0, 0)
    else
        # Roll out with a full newton step
        sol.J = Inf
        @inbounds @simd for k = 1:(params.N-1)
            fill!(sol.f̃s[k], 0.0)
        end
        forward_pass!(sol, cache, params, 1.0, 1)
    end
    return nothing
end
