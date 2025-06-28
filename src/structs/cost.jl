"""
    TrajectoryCost(stage_cost, term_cost, nx, nu, N)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
mutable struct TrajectoryCost
    stage::Function
    terminal::Function
    stage_ℓs::Vector{<:DiffFloat64}
    stage_xs::Vector{Vector{<:DiffFloat64}}
    stage_us::Vector{Vector{<:DiffFloat64}}
    term_x::Vector{<:DiffFloat64}

    function TrajectoryCost(
        stage_cost::Function,
        term_cost::Function,
        nx::Int,
        nu::Int,
        N::Int
    )::TrajectoryCost
        stage(
            x::Vector{<:DiffFloat64},
            u::Vector{<:DiffFloat64}
        ) = stage_cost(x, u)::DiffFloat64

        terminal(
            x::Vector{<:DiffFloat64}
        ) = term_cost(x)::DiffFloat64

        stage_ℓs = zeros(N-1)
        term_x = zeros(nx)
        stage_xs = [zeros(nx) for k = 1:(N-1)]
        stage_us = [zeros(nu) for k = 1:(N-1)]
        return new(stage, terminal, stage_ℓs, stage_xs, stage_us, term_x)
    end
end

"""
    cost(xs, us, xrefs, urefs)

Callable struct method for the `TrajectoryCost` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cost::TrajectoryCost)(
    xs::Vector{Vector{Float64}},
    us::Vector{Vector{Float64}},
    xrefs::Vector{Vector{Float64}},
    urefs::Vector{Vector{Float64}}
)::Float64
    # Broadcast stage x - xref
    BLAS.copy!.(cost.stage_xs, (@view xs[1:(end-1)]))
    BLAS.axpy!.(-1.0, (@view xrefs[1:(end-1)]), cost.stage_xs)

    # Broadcast stage u - ref
    BLAS.copy!.(cost.stage_us, (@view us[1:end]))
    BLAS.axpy!.(-1.0, (@view urefs[1:end]), cost.stage_us)

    # Broadcast stage cost
    BLAS.copy!(cost.stage_ℓs, cost.stage.(cost.stage_xs, cost.stage_us))

    # Get terminal x - xref
    BLAS.copy!(cost.term_x, xs[end])
    BLAS.axpy!(-1.0, xrefs[end], cost.term_x)
    return sum(cost.stage_ℓs) + cost.terminal(cost.term_x)
end
