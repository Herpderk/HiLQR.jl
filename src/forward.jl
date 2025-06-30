"""
"""
function nonlinear_rollout!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    sol::Solution,
    params::ProblemParameters,
    defect_rate::Float64
)::Nothing
    # Close defects
    fwd.c = 1.0 - fwd.α * defect_rate
    mul!.(fwd.f̃s, fwd.c, sol.f̃s)

    # Initialize trajectory with previous solution
    BLAS.copy!.(fwd.xs, sol.xs)
    BLAS.copy!.(fwd.us, sol.us)

    # Forward rollout
    @inbounds for k = 1:(params.N-1)
        # Update control input
        #fwd.us[k] = sol.us[k] - α*ds[k] - Ks[k]*(fwd.xs[k] - sol.xs[k])
        mul!(tmp.u, fwd.α, bwd.ds[k])
        BLAS.axpy!(-1.0, tmp.u, fwd.us[k])
        BLAS.copy!(tmp.x, fwd.xs[k])
        BLAS.axpy!(-1.0, sol.xs[k], tmp.x)
        mul!(tmp.u, bwd.Ks[k], tmp.x)
        BLAS.axpy!(-1.0, tmp.u, fwd.us[k])

        # Integrate smooth dynamics
        BLAS.copy!(fwd.xs[k+1], params.igtr(
            fwd.modes[k].flow, fwd.xs[k], fwd.us[k], params.Δt
        ))

        # Reset and update mode if a guard is hit
        Rflag = false
        @inbounds for (trn, mJ) in fwd.modes[k].transitions
            if trn.guard(fwd.xs[k+1]) < 0.0
                BLAS.copy!(fwd.xs[k+1], trn.reset(fwd.xs[k+1]))
                fwd.trn_syms[k] = params.rev_trns_dict[trn]
                fwd.modes[k+1] = mJ
                Rflag = true
                break
            end
        end

        # Don't update mode if a guard is not hit
        if !Rflag
            fwd.trn_syms[k] = NULL_TRANSITION
            fwd.modes[k+1] = fwd.modes[k]
        end

        # Apply defects to rollout
        BLAS.axpy!(-1.0, fwd.f̃s[k], fwd.xs[k+1])
    end
    return
end

"""
"""
function forward_pass!(
    sol::Solution,
    cache::SolverCache,
    params::ProblemParameters,
    max_step::Float64,
    defect_rate::Float64,
    ls_iter::Int
)::Nothing
    # Get references to SolverCache structs
    fwd = cache.fwd
    bwd = cache.bwd
    tmp = cache.tmp

    # Initialize line search step size and trajectory cost
    fwd.α = max_step
    Jls = 0.0

    # Iterate backtracking line search
    @inbounds for i = 1:ls_iter
        # Roll out new gains
        nonlinear_rollout!(fwd, bwd, tmp, sol, params, defect_rate)

        # Evaluate trajectory cost
        Jls = params.fwd_cost(fwd.xs, fwd.us, params.xrefs, params.urefs)

        # Use decreasing cost as line search criteria
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

        # Shrink step size
        fwd.α *= 0.5
    end

    # Save solver iteration data
    fwd.ΔJ = abs(Jls - sol.J)
    sol.J = Jls
    BLAS.copy!.(sol.xs, fwd.xs)
    BLAS.copy!.(sol.us, fwd.us)
    BLAS.copy!.(sol.f̃s, fwd.f̃s)
    sol.f̃norm = norm(sol.f̃s, Inf)
    return
end
