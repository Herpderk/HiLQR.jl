"""
    TrajectoryCost(dims, idx, stage_cost, terminal_cost)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
struct TrajectoryCost
    idx::PrimalIndices
    stage::Function
    terminal::Function
end

function TrajectoryCost(
    idx::PrimalIndices,
    stage_cost::Function,
    terminal_cost::Function
)::TrajectoryCost
    annotated_stage_cost = (
        x::Vector{<:DiffFloat},
        u::VectOr{<:DiffFloat}
    ) -> stage_cost(x, u)::DiffFloat

    annotated_terminal_cost = (
        x::Vector{<:DiffFloat}
    ) -> terminal_cost(x, u)::DiffFloat

    return new(idx, annotated_stage_cost, annotated_terminal_cost)
end

"""
    cost(yref, y)

Callable struct method for the `TrajectoryCost` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cost::TrajectoryCost)(
    yref::Vector{<:AbstractFloat},
    y::Vector{<:DiffFloat}
)::DiffFloat
    Js = zeros(eltype(y), cost.dims.N)
    @simd for k in 1 : cost.dims.N-1
        Js[k] = cost.stage(
            y[cost.idx.x[k]] - yref[cost.idx.x[k]],
            y[cost.idx.u[k]] - yref[cost.idx.u[k]]
        )
    end
    Js[end] = cost.terminal(y[cost.idx.x[end]] - yref[cost.idx.x[end]])
    return sum(Js)
end

"""
"""
function expand_terminal_cost!(
    Jxx_result::DiffResults.DiffResult,
    J::AbstractFloat,
    Jx::Vector{<:AbstractFloat},
    Jxx::Matrix{<:AbstractFloat},
    cost::TrajectoryCost,
    xerr::Vector{<:AbstractFloat}
)::Nothing
    Jxx_result = ForwardDiff.hessian!(Jxx_result, cost.terminal, xerr)
    J = DiffResults.value(Jxx_result)
    Jx = DiffResults.gradient(Jxx_result)
    Jxx = DiffResults.hessian(Jxx_result)
    return nothing
end

"""
"""
function expand_stage_cost!(
    Jxx_result::DiffResults.DiffResult,
    Juu_result::DiffResults.DiffResult,
    J::AbstractFloat,
    Jx::Vector{<:AbstractFloat},
    Ju::Vector{<:AbstractFloat},
    Jxx::Matrix{<:AbstractFloat},
    Juu::Matrix{<:AbstractFloat},
    cost::TrajectoryCost,
    xerr::Vector{<:AbstractFloat},
    uerr::Vector{<:AbstractFloat},
)::Nothing
    Jxx_result = ForwardDiff.hessian!(
        Jxx_result, δx -> cost.stage(δx, uerr), xerr
    )
    Juu_result = ForwardDiff.hessian!(
        Juu_result, δu -> cost.stage(xerr, δu), uerr
    )
    J = DiffResults.value(Jxx_result)
    Jx = DiffResults.gradient(Jxx_result)
    Ju = DiffResults.gradient(Juu_result)
    Jxx = DiffResults.hessian(Jxx_result)
    Juu = DiffResults.hessian(Juu_result)
    return nothing
end
