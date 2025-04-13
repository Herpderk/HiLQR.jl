"""
"""
function log(
    sol::Solution,
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    iter::Int
)::Nothing
    if rem(iter-1, 20) == 0
        println("-------------------------------------------------------")
        println("iter        J          ΔJ        ‖f̂‖         α       τ")
        println("-------------------------------------------------------")
    end

    τ = 0
    for trn in fwd.trns
        τ = !isnothing(trn.val) ? τ+1 : τ
    end

    @printf(
        "%4.04i     %8.2e   %8.2e   %8.2e   %7.5f   %3.03i\n",
        iter, sol.J, bwd.ΔJ, sol.f̂norm, fwd.α, τ
    )
end

"""
"""
function terminate(
    sol::Solution,
    bwd::BackwardTerms,
    defect_tol::Float64,
    stat_tol::Float64
)::Bool
    return (sol.f̂norm < defect_tol) && (bwd.ΔJ < stat_tol)
end

"""
"""
function inner_solve!(
    sol::Solution,
    cache::Cache,
    params::Parameters,
    regularizer::Float64,
    defect_tol::Float64,
    stat_tol::Float64,
    max_iter::Int,
    max_ls_iter::Int,
    verbose::Bool
)::Nothing
    # References to cache attributes
    fwd = cache.fwd
    bwd = cache.bwd
    Jexp = cache.Jexp
    Qexp = cache.Qexp
    tmp = cache.tmp

    # Initial roll-out
    init_forward_terms!(sol, fwd, bwd, tmp, params)

    # Main solve loop
    for i = 1:max_iter
        backward_pass!(bwd, fwd, Jexp, Qexp, tmp, sol, params, regularizer)
        forward_pass!(sol, fwd, bwd, tmp, params, max_ls_iter)

        verbose ? log(sol, fwd, bwd, i) : nothing
        if terminate(sol, bwd, defect_tol, stat_tol)
            verbose ? println("Optimal solution found!") : nothing
            return nothing
        end
    end

    verbose ? println("Maximum iterations exceeded!") : nothing
    return nothing
end

function solve!(
    sol::Solution,
    cache::Cache,
    params::Parameters;
    regularizer::Float64 = 1e-6,
    defect_tol::Float64 = 1e-6,
    stat_tol::Float64 = 1e-4,
    max_iter::Int = 1000,
    max_ls_iter::Int = 10,
    verbose::Bool = true
)::Nothing
    inner_solve!(
        sol,
        cache,
        params,
        regularizer,
        defect_tol,
        stat_tol,
        max_iter,
        max_ls_iter,
        verbose
    )
    return nothing
end

function solve(
    params::Parameters;
    regularizer::Float64 = 1e-6,
    defect_tol::Float64 = 1e-6,
    stat_tol::Float64 = 1e-4,
    max_iter::Int = 1000,
    max_ls_iter::Int = 10,
    verbose::Bool = true
)::Solution
    sol = Solution(params)
    cache = Cache(params)
    inner_solve!(
        sol,
        cache,
        params,
        regularizer,
        defect_tol,
        stat_tol,
        max_iter,
        max_ls_iter,
        verbose
    )
    return sol
end
