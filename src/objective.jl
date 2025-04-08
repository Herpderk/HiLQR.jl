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
function expand_cost!(
    Jxx_result::DiffResults.DiffResult,
    Juu_result::DiffResults.DiffResult,
    J::AbstractFloat,
    Jx::Vector{<:AbstractFloat},
    Ju::Vector{<:AbstractFloat},
    Jxx::Matrix{<:AbstractFloat},
    Juu::Matrix{<:AbstractFloat},
    xerr::Vector{<:AbstractFloat},
    uerr::Vector{<:AbstractFloat}
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

"""
"""
function expand_terminal_cost(
    cost::TrajectoryCost,
    yref::Vector{<:AbstractFloat},
    y::Vector{<:AbstractFloat}
)::NamedTuple{
    AbstractFloat,
    Vector{<:AbstractFloat},
    Vector{<:AbstractFloat},
    Matrix{<:AbstractFloat},
    Matrix{<:AbstractFloat}
}
    nx = cost.idx.dims.nx
    nu = cost.idx.dims.nu

    Jxx_result = DiffResults.HessianResult(zeros(eltype(y), nx))
    Juu_result = DiffResults.HessianResult(zeros(eltype(y), nu))

    J = zero(eltype(y))
    Jx = zeros(eltype(y), nx)
    Ju = zeros(eltype(y), nu)
    Jxx = zeros(eltype(y), nx, nx)
    Juu = zeros(eltype(y), nu, nu)

    xerr = y[cost.idx.x[end]] - yref[cost.idx.x[end]]
    uerr = y[cost.idx.u[end]] - yref[cost.idx.u[end]]

    expand_cost!(Jxx_result, Juu_result, J, Jx, Ju, Jxx, Juu, xerr, uerr)
    return(J=J, Jx=Jx, Ju=Ju, Jxx=Jxx, Juu=Juu)
end

"""
"""
function expand_stage_costs(
    cost::TrajectoryCost,
    yref::Vector{<:AbstractFloat},
    y::Vector{<:AbstractFloat},
)::NamedTuple{
    Vector{<:AbstractFloat},
    Vector{<:Vector{<:AbstractFloat}},
    Vector{<:Vector{<:AbstractFloat}},
    Vector{<:Matrix{<:AbstractFloat}},
    Vector{<:Matrix{<:AbstractFloat}}
}
    N = cost.idx.dims.N
    nx = cost.idx.dims.nx
    nu = cost.idx.dims.nu

    Jxx_result = DiffResults.HessianResult(zeros(eltype(y), nx))
    Juu_result = DiffResults.HessianResult(zeros(eltype(y), nu))

    Js = zeros(eltype(y), N-1)
    Jxs = [zeros(eltype(y), nx) for k = 1:(N-1)]
    Jus = [zeros(eltype(y), nu) for k = 1:(N-1)]
    Jxxs = [zeros(eltype(y), nx,nx) for k = 1:(N-1)]
    Juus = [zeros(eltype(y), nu,nu) for k = 1:(N-1)]

    @simd for k = 1:(N-1)
        xerr = y[cost.idx.x[k]] - yref[cost.idx.x[k]]
        uerr = y[cost.idx.u[k]] - yref[cost.idx.u[k]]
        expand_cost!(
            Jxx_result, Juu_result,
            Js[k], Jxs[k], Jus[k], Jxxs[k], Juus[k],
            xerr, uerr
        )
    end

    return (Js=Js, Jxs=Jxs, Jus=Jus, Jxxs=Jxxs, Juus=Juus)
end
