"""
"""
function log(
    sol::Solution,
    fwd::ForwardTerms,
    iter::Int
)::Nothing
    if rem(iter-1, 20) == 0
        println("-------------------------------------------------------")
        println("iter        J          ΔJ         ‖f̃‖        α       τ")
        println("-------------------------------------------------------")
    end

    τ = 0
    for trn in fwd.trns
        τ = !isnothing(trn.val) ? τ+1 : τ
    end

    @printf(
        "%4.04i     %8.2e   %8.2e   %8.2e   %7.5f   %3.03i\n",
        iter, sol.J, fwd.ΔJ, sol.f̃norm, fwd.α, τ
    )
end

"""
"""
function terminate(
    sol::Solution,
    fwd::ForwardTerms,
    defect_tol::Float64,
    stat_tol::Float64
)::Bool
    return (sol.f̃norm < defect_tol) && (fwd.ΔJ < stat_tol)
end

"""
"""
function inner_solve!(
    sol::Solution,
    cache::Cache,
    params::Parameters,
    αmax::Float64,
    reg::Float64,
    stat_tol::Float64,
    defect_tol::Float64,
    max_iter::Int,
    max_ls_iter::Int,
    multishoot::Bool,
    verbose::Bool
)::Nothing
    # References to cache attributes
    fwd = cache.fwd
    bwd = cache.bwd
    Jexp = cache.Jexp
    Qexp = cache.Qexp
    tmp = cache.tmp

    # Initial roll-out
    init_terms!(sol, fwd, bwd, tmp, params, αmax, multishoot)

    # Main solve loop
    for i = 1:max_iter
        backward_pass!(bwd, fwd, Jexp, Qexp, tmp, sol, params, reg)
        forward_pass!(sol, fwd, bwd, tmp, params, max_ls_iter, αmax, multishoot)

        verbose ? log(sol, fwd, i) : nothing
        if terminate(sol, fwd, defect_tol, stat_tol)
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
    αmax::Float64 = 1.0,
    reg::Float64 = 1e-6,
    stat_tol::Float64 = 1e-9,
    defect_tol::Float64 = 1e-9,
    max_iter::Int = 1000,
    max_ls_iter::Int = 20,
    multishoot::Bool = false,
    verbose::Bool = true,
)::Nothing
    inner_solve!(
        sol,
        cache,
        params,
        αmax,
        reg,
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
