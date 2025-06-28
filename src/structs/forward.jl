"""
"""
mutable struct NullTransition
    val::Union{Transition, Nothing}
end


"""
"""
mutable struct ForwardTerms
    modes::Vector{HybridMode}
    trns::Vector{NullTransition}
    xs::Vector{Vector{Float64}}
    us::Vector{Vector{Float64}}
    f̃s::Vector{Vector{Float64}}
    α::Float64
    ΔJ::Float64
end

function ForwardTerms(
    sys::HybridSystem,
    nx::Int,
    nu::Int,
    N::Int
)::ForwardTerms
    mode = first(values(sys.modes))
    modes = [mode for k = 1:N]
    trns = [NullTransition(nothing) for k = 1:(N-1)]
    xs = [zeros(nx) for k = 1:N]
    us = [zeros(nu) for k = 1:(N-1)]
    f̃s = [zeros(nx) for k = 1:(N-1)]
    α = 1.0
    ΔJ = Inf
    return ForwardTerms(modes, trns, xs, us, f̃s, α, ΔJ)
end
