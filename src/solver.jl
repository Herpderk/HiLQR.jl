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
    for trn in cache.fwd.trns
        τ = !isnothing(trn.val) ? τ+1 : τ
    end

    @printf(
        "%4.04i     %8.2e   %8.2e   %8.2e   %7.5f   %3.03i\n",
        iter, sol.J, cache.fwd.ΔJ, sol.f̃norm, cache.fwd.α, τ
    )
end

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
    # Initialize solver variables
    init_solver!(sol, cache, params, regularizer, multishoot)

    # Main solve loop
    for i = 1:max_iter
        backward_pass!(cache, params)
        forward_pass!(sol, cache, params, max_step, defect_rate, max_ls_iter)

        verbose ? log(sol, cache, i) : nothing
        if terminate(sol, cache, defect_tol, stat_tol)
            verbose ? println("\nOptimal solution found!") : nothing
            return nothing
        end
    end

    verbose ? println("\nMaximum iterations exceeded!") : nothing
    return nothing
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
    return nothing
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
