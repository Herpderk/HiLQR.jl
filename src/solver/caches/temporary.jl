"""
"""
mutable struct TemporaryCache
    x::Vector{<:DiffFloat64}
    u::Vector{<:DiffFloat64}

    xx1::Matrix{Float64}
    xx2::Matrix{Float64}

    uu::Matrix{Float64}
    xu::Matrix{Float64}
    ux::Matrix{Float64}

    x_dual::AbstractVector

    xx_hess::DiffResults.DiffResult
    uu_hess::DiffResults.DiffResult
end

function TemporaryCache(
    nx::Int,
    nu::Int
)::TemporaryCache
    x = zeros(nx)
    u = zeros(nu)

    xx1 = zeros(nx, nx)
    xx2 = zeros(nx, nx)
    uu = zeros(nu, nu)
    xu = zeros(nx, nu)
    ux = zeros(nu, nx)

    x_dual = zeros(nx)

    xx_hess = DiffResults.HessianResult(zeros(nx))
    uu_hess = DiffResults.HessianResult(zeros(nu))

    return TemporaryCache(
        x,
        u,
        xx1,
        xx2,
        uu,
        xu,
        ux,
        x_dual,
        xx_hess,
        uu_hess
    )
end
