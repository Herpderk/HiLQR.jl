"""
    TrajectoryCost(stage_cost, terminal_cost)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
struct TrajectoryCost
    stage::Function
    terminal::Function
end

function TrajectoryCost(
    stage_cost::Function,
    terminal_cost::Function
)::TrajectoryCost
    annotated_stage_cost = (
        x::Vector{<:DiffFloat},
        u::Vector{<:DiffFloat}
    ) -> stage_cost(x, u)::DiffFloat

    annotated_terminal_cost = (
        x::Vector{<:DiffFloat}
    ) -> terminal_cost(x, u)::DiffFloat

    return new(annotated_stage_cost, annotated_terminal_cost)
end

"""
    cost(xrefs, urefs, xs, us)

Callable struct method for the `TrajectoryCost` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cost::TrajectoryCost)(
    xrefs::Vector{Vector{Float64}},
    urefs::Vector{Vector{Float64}},
    xs::Vector{Vector{<:DiffFloat}},
    us::Vector{Vector{<:DiffFloat}}
)::DiffFloat
    N = length(xs)
    Js = zeros(N)
    @simd for k = 1:(N-1)
        Js[k] = cost.stage(xs[k] - xrefs[k], us[k] - urefs[k])
    end
    Js[end] = cost.terminal(xs[end] - xrefs[end])
    return sum(Js)
end

"""
"""
mutable struct CostExpansion
    J::Float64
    Jx::Vector{Float64}
    Ju::Vector{Float64}
    Jxx::Matrix{Float64}
    Juu::Matrix{Float64}
    Jxx_result::DiffResults.DiffResult
    Juu_result::DiffResults.DiffResult
end

function CostExpansion(
    nx::Int,
    nu::Int
)::CostExpansion
    Jx = zeros(nx)
    Ju = zeros(nu)
    Jxx = zeros(nx, nx)
    Juu = zeros(nu, nu)
    Jxx_result = DiffResults.HessianResult(zeros(nx))
    Juu_result = DiffResults.HessianResult(zeros(nu))
    return new(Jx, Ju, Jxx, Juu, Jxx_result, Juu_result)
end

"""
"""
function expand_terminal_cost!(
    exp::CostExpansion,
    cost::TrajectoryCost,
    xerr::Vector{Float64}
)::Nothing
    exp.Jxx_result = ForwardDiff.hessian!(
        exp.Jxx_result, cost.terminal, xerr
    )
    exp.J += DiffResults.value(exp.Jxx_result)
    exp.Jx = DiffResults.gradient(exp.Jxx_result)
    exp.Jxx = DiffResults.hessian(exp.Jxx_result)
    return nothing
end

"""
"""
function expand_stage_cost!(
    exp::CostExpansion,
    cost::TrajectoryCost,
    xerr::Vector{Float64},
    uerr::Vector{Float64}
)::Nothing
    exp.Jxx_result = ForwardDiff.hessian!(
        exp.Jxx_result, δx -> cost.stage(δx, uerr), xerr
    )
    exp.Juu_result = ForwardDiff.hessian!(
        exp.Juu_result, δu -> cost.stage(xerr, δu), uerr
    )
    exp.J += DiffResults.value(exp.Jxx_result)
    exp.Jx = DiffResults.gradient(exp.Jxx_result)
    exp.Ju = DiffResults.gradient(exp.Juu_result)
    exp.Jxx = DiffResults.hessian(exp.Jxx_result)
    exp.Juu = DiffResults.hessian(exp.Juu_result)
    return nothing
end
