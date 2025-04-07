"""
    PrimalDimensions(N, nx, nu, nt)

Contains dimensions corresponding to the number of horizon time steps, states, inputs, time, and total decision variables. Time with 0 dimensions implies that the time step durations are fixed.
"""
struct PrimalDimensions
    N::Int
    nx::Int
    nu::Int
    nt::Int
    ny::Int
    function PrimalDimensions(
        N::Int,
        nx::Int,
        nu::Int,
        nt::Int
    )::PrimalDimensions
        !(nt in (0, 1)) ? error("Dimension of Δt must be 0 or 1!") : nothing
        ny = N*nx + (N-1)*(nu + nt)
        return new(N, nx, nu, nt, ny)
    end
end

"""
    get_primal_indices(dims, Δstart, Δstop)

Returns a range of indices given the [x, u, h] order of decision variables.
"""
function get_primal_indices(
    dims::PrimalDimensions,
    N::Int,
    Δstart::Int,
    Δstop::Int
)::Vector{UnitRange{Int}}
    ny_per_step = dims.nx + dims.nu + dims.nt
    return [(1+Δstart : dims.nx+Δstop) .+ (k-1)*(ny_per_step) for k = 1:N]
end

"""
    PrimalIndices(dims)

Contains ranges of indices for getting instances of x, u, or Δt given the [x, u, Δt] order of decision variables.
"""
struct PrimalIndices
    x::Vector{UnitRange{Int}}
    u::Vector{UnitRange{Int}}
    Δt::Union{Nothing, Vector{UnitRange{Int}}}
    function PrimalIndices(
        dims::PrimalDimensions
    )::PrimalIndices
        x_idx = get_primal_indices(dims, dims.N, 0, 0)
        u_idx = get_primal_indices(dims, dims.N-1, dims.nx, dims.nu)
        if dims.nt == 1
            Δt_idx = get_primal_indices(
                dims, dims.N-1, dims.nx+dims.nu, dims.nu+dims.nt)
        elseif dims.nt == 0
            Δt_idx = nothing
        end
        return new(x_idx, u_idx, Δt_idx)
    end
end

"""
    compose_trajectory(dims, idx, xs, us)

Interleaves sequences of states and inputs into a single trajectory of primal variables given the problem dimensions and indices.
"""
function compose_trajectory(
    dims::PrimalDimensions,
    idx::PrimalIndices,
    xs::Vector{<:AbstractFloat},
    us::Vector{<:AbstractFloat}
)::Vector{<:AbstractFloat}
    y = zeros(eltype(xs), dims.ny)
    xcurr = 1 : dims.nx
    ucurr = 1 : dims.nu
    for k in 1 : dims.N-1
        y[idx.x[k]] = xs[xcurr]
        y[idx.u[k]] = us[ucurr]
        xcurr = xcurr .+ dims.nx
        ucurr = ucurr .+ dims.nu
    end
    y[idx.x[dims.N]] = xs[xcurr]
    return y
end

"""
    compose_trajectory(dims, idx, xs, us, Δts)

Interleaves sequences of states, inputs and time steps into a single trajectory of primal variables given the problem dimensions and indices.
"""
function compose_trajectory(
    dims::PrimalDimensions,
    idx::PrimalIndices,
    xs::Vector{<:AbstractFloat},
    us::Vector{<:AbstractFloat},
    Δts::Vector{<:AbstractFloat}
)::Vector{<:AbstractFloat}
    y = compose_trajectory(dims, idx, xs, us)
    for k in 1 : dims.N-1
        y[idx.Δt[k]] = Δts[k:k]
    end
    return y
end

"""
    decompose_trajectory(idx, y)

Separates a trajectory of primal variables into sequences of states, inputs, and time steps given the problem indices.
"""
function decompose_trajectory(
    idx::PrimalIndices,
    y::Vector{<:AbstractFloat}
)::Tuple{
    Vector{<:AbstractFloat},
    Vector{<:AbstractFloat},
    Union{Nothing, Vector{<:AbstractFloat}}
}
    xs = vcat([y[i] for i = idx.x[1 : end]]...)
    us = vcat([y[i] for i = idx.u[1 : end]]...)
    if !isnothing(idx.Δt)
        Δts = vcat([y[i] for i = idx.u[1 : end]]...)
    else
        Δts = nothing
    end
    return xs, us, Δts
end
