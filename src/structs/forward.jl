"""
"""
mutable struct ForwardTerms
    modes::Vector{HybridMode}
    trn_syms::Vector{Symbol}

    xs::Vector{Vector{Float64}}
    us::Vector{Vector{Float64}}
    f̃s::Vector{Vector{Float64}}

    c::Float64
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
    trn_syms = [NULL_TRANSITION for k = 1:(N-1)]

    xs = [zeros(nx) for k = 1:N]
    us = [zeros(nu) for k = 1:(N-1)]
    f̃s = [zeros(nx) for k = 1:(N-1)]

    c = 0.0
    α = 0.0
    ΔJ = 0.0
    return ForwardTerms(modes, trn_syms, xs, us, f̃s, c, α, ΔJ)
end
