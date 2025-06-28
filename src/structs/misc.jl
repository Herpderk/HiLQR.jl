"""
"""
mutable struct TemporaryArrays
    x::Vector{Float64}
    u::Vector{Float64}

    xx1::Matrix{Float64}
    xx2::Matrix{Float64}

    uu::Matrix{Float64}
    xu::Matrix{Float64}
    ux::Matrix{Float64}

    xx_hess::DiffResults.DiffResult
    uu_hess::DiffResults.DiffResult
end

function TemporaryArrays(
    nx::Int,
    nu::Int
)::TemporaryArrays
    x = zeros(nx)
    u = zeros(nu)
    xx1 = zeros(nx, nx)
    xx2 = zeros(nx, nx)
    uu = zeros(nu, nu)
    xu = zeros(nx, nu)
    ux = zeros(nu, nx)
    xx_hess = DiffResults.HessianResult(zeros(nx))
    uu_hess = DiffResults.HessianResult(zeros(nu))
    return TemporaryArrays(x, u, xx1, xx2, uu, xu, ux, xx_hess, uu_hess)
end
