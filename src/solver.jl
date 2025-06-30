"""
"""
function terminate(
    sol::Solution,
    cache::Cache,
    defect_tol::Float64,
    stat_tol::Float64
)::Bool
    return (sol.f̃norm < defect_tol) && (cache.fwd.ΔJ < stat_tol)
end

"""
"""
function log(
    sol::Solution,
    cache::Cache,
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
    max_step::Float64,
    defect_rate::Float64,
    stat_tol::Float64,
    defect_tol::Float64,
    max_iter::Int,
    max_ls_iter::Int,
)::Nothing
    if max_step <= 0.0 || max_step > 1.0
        ArgumentError("The max step size should be between 0 and 1")
    end
    if max_step <= 0.0 || defect_rate > 1.0
        ArgumentError("The defect closure rate should be between 0 and 1")
    end
    if stat_tol <= 0.0
        ArgumentError("The stationarity tolerance should be greater than 0")
    end
    if defect_tol <= 0.0
        ArgumentError("The defect tolerance should be greater than 0")
    end
    if max_iter <= 0
        ArgumentError("The max number of iterations should be greater than 0")
    end
    if max_ls_iter <= 0
        ArgumentError(
            "The max number of line search iterations should be greater than 0"
        )
    end
end

"""
"""
function init_solver!(
    sol::Solution,
    cache::Cache,
    params::Parameters,
    regularizer::Float64,
    multishoot::Bool
)::Nothing
    # Get references to Cache structs
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
function core_solve!(
    sol::Solution,
    cache::Cache,
    params::Parameters,
    regularizer::Float64,
    max_step::Float64,
    defect_rate::Float64,
    stat_tol::Float64,
    defect_tol::Float64,
    max_iter::Int,
    max_ls_iter::Int,
    multishoot::Bool,
    verbose::Bool
)::Nothing
    # Verify options are valid
    assert_opts!(
        max_step,
        defect_rate,
        stat_tol,
        defect_tol,
        max_iter,
        max_ls_iter
    )

    # Initialize solver variables
    init_solver!(sol, cache, params, regularizer, multishoot)

    # Main solve loop
    for i = 1:max_iter
        backward_pass!(cache, params)
        forward_pass!(sol, cache, params, max_step, defect_rate, max_ls_iter)

        verbose ? log(sol, cache, i) : nothing
        if terminate(sol, cache, defect_tol, stat_tol)
            verbose ? println("\nOptimal solution found!") : nothing
            return
        end
    end

    verbose ? println("\nMaximum iterations exceeded!") : nothing
    return
end

function solve!(
    sol::Solution,
    cache::Cache,
    params::Parameters;
    regularizer::Float64 = 1e-6,
    max_step::Float64 = 1.0,
    defect_rate::Float64 = 1.0,
    stat_tol::Float64 = 1e-9,
    defect_tol::Float64 = 1e-9,
    max_iter::Int = 1000,
    max_ls_iter::Int = 20,
    multishoot::Bool = false,
    verbose::Bool = true,
)::Nothing
    core_solve!(
        sol,
        cache,
        params,
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
    return
end

function solve(
    params::Parameters;
    options...
)::Solution
    sol = Solution(params)
    cache = Cache(params)
    solve!(sol, cache, params, options...)
    return sol
end
