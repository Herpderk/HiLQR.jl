"""
"""
function terminate(
    sol::Solution,
    cache::SolverCache,
    defect_tol::Float64,
    stat_tol::Float64
)::Bool
    return (sol.f̃norm < defect_tol) && (cache.fwd.ΔJ < stat_tol)
end

"""
"""
function log(
    sol::Solution,
    cache::SolverCache,
    iter::Int
)::Nothing
    if rem(iter-1, 20) == 0
        println("-------------------------------------------------------")
        println("iter        J          ΔJ         ‖f̃‖        α       τ")
        println("-------------------------------------------------------")
    end

    τ = 0
    for trn in cache.fwd.trn_syms
        τ = trn != NULL_TRANSITION ? τ+1 : τ
    end

    @printf(
        "%4.04i     %8.2e   %8.2e   %8.2e   %7.5f   %3.03i\n",
        iter, sol.J, cache.fwd.ΔJ, sol.f̃norm, cache.fwd.α, τ
    )
end

"""
"""
function assert_opts!(
    opts::SolverOptions
)::Nothing
    if opts.max_step <= 0.0 || opts.max_step > 1.0
        ArgumentError("The max step size should be between 0 and 1")
    end
    if opts.max_step <= 0.0 || opts.defect_rate > 1.0
        ArgumentError("The defect closure rate should be between 0 and 1")
    end
    if opts.stat_tol <= 0.0
        ArgumentError("The stationarity tolerance should be greater than 0")
    end
    if opts.defect_tol <= 0.0
        ArgumentError("The defect tolerance should be greater than 0")
    end
    if opts.max_iter <= 0
        ArgumentError("The max number of iterations should be greater than 0")
    end
    if opts.max_ls_iter <= 0
        ArgumentError(
            "The max number of line search iterations should be greater than 0"
        )
    end
end

"""
"""
function init_solver!(
    sol::Solution,
    cache::SolverCache,
    params::ProblemParameters,
    regularizer::Float64,
    multishoot::Bool
)::Nothing
    # Get references to SolverCache structs
    fwd = cache.fwd
    bwd = cache.bwd

    # Set regularizer matrix
    mul!(bwd.μ, regularizer, I)

    # Initialize gains
    fill!.(bwd.Ks, 0.0)
    fill!.(bwd.ds, 0.0)

    # Set initial conditions
    fwd.modes[1] = params.fwd_sys.modes[params.mI]
    BLAS.copy!(sol.xs[1], params.x0)

    # Initialize trajectory cost
    sol.J = Inf

    if multishoot
        # Initialize defects
        @inbounds for k = 1:(params.N-1)
            BLAS.copy!(sol.f̃s[k], params.igtr(   # TODO handle mode schedule
                fwd.modes[1].flow, sol.xs[k], sol.us[k], params.Δt
            ))
            BLAS.axpy!(-1.0, sol.xs[k+1], sol.f̃s[k])
        end

        # Initialize forward terms
        BLAS.copy!.(fwd.xs, sol.xs)
        BLAS.copy!.(fwd.us, sol.us)
        BLAS.copy!.(fwd.f̃s, sol.f̃s)
        fwd.α = 0.0
    else
        # Set defects to 0
        fill!.(sol.f̃s, 0.0)

        # Roll out with a full newton step
        forward_pass!(sol, cache, params, 1.0, 1.0, 1)
    end
    return
end

"""
"""
function solve!(
    sol::Solution,
    cache::SolverCache,
    params::ProblemParameters,
    opts::SolverOptions
)::Nothing
    # Verify options are valid
    assert_opts!(opts)

    # Initialize solver variables
    init_solver!(sol, cache, params, opts.regularizer, opts.multishoot)

    # Main solve loop
    for i = 1:opts.max_iter
        backward_pass!(cache, params)
        forward_pass!(
            sol,
            cache,
            params,
            opts.max_step,
            opts.defect_rate,
            opts.max_ls_iter
        )

        opts.verbose ? log(sol, cache, i) : nothing
        if terminate(sol, cache, opts.defect_tol, opts.stat_tol)
            opts.verbose ? println("\nOptimal solution found!") : nothing
            return
        end
    end

    opts.verbose ? println("\nMaximum iterations exceeded!") : nothing
    return
end

function solve(
    params::ProblemParameters,
    opts::SolverOptions = SolverOptions()
)::Solution
    sol = Solution(params)
    cache = SolverCache(params)
    solve!(sol, cache, params, opts)
    return sol
end
