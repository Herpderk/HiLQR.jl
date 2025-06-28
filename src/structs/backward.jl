"""
"""
mutable struct FlowExpansion
    x::Matrix{Float64}
    u::Matrix{Float64}
end

function FlowExpansion(
    nx::Int,
    nu::Int
)::FlowExpansion
    Fx = zeros(nx, nx)
    Fu = zeros(nx, nu)
    return FlowExpansion(Fx, Fu)
end


"""
"""
mutable struct CostExpansion
    x::Vector{Float64}
    u::Vector{Float64}
    xx::Matrix{Float64}
    uu::Matrix{Float64}
end

function CostExpansion(
    nx::Int,
    nu::Int
)::CostExpansion
    Lx = zeros(nx)
    Lu = zeros(nu)
    Lxx = zeros(nx, nx)
    Luu = zeros(nu, nu)
    return CostExpansion(Lx, Lu, Lxx, Luu)
end


"""
"""
mutable struct ValueExpansion
    x::Vector{Float64}
    xx::Matrix{Float64}
end

function ValueExpansion(
    nx::Int
)::ValueExpansion
    Vx = zeros(nx)
    Vxx = zeros(nx, nx)
    return ValueExpansion(Vx, Vxx)
end


"""
"""
mutable struct ActionValueExpansion
    x::Vector{Float64}
    u::Vector{Float64}
    xx::Matrix{Float64}
    xu::Matrix{Float64}
    ux::Matrix{Float64}
    uu::Matrix{Float64}
    uu_lu::SparseArrays.UMFPACK.UmfpackLU{Float64, Int64}
end

function ActionValueExpansion(
    nx::Int,
    nu::Int
)::ActionValueExpansion
    Qx = zeros(nx)
    Qu = zeros(nu)
    Qxx = zeros(nx, nx)
    Qxu = zeros(nx, nu)
    Qux = zeros(nu, nx)
    Quu = zeros(nu, nu)
    Quu_lu = lu(sparse(rand(nu, nu)))
    return ActionValueExpansion(Qx, Qu, Qxx, Qxu, Qux, Quu, Quu_lu)
end


"""
"""
mutable struct BackwardTerms
    Fs::Vector{FlowExpansion}
    Ls::Vector{CostExpansion}
    Vs::Vector{ValueExpansion}
    Qs::Vector{ActionValueExpansion}
    Ks::Vector{VecOrMat{Float64}}
    ds::Vector{Vector{Float64}}
    μ::Matrix{Float64}
    ΔJ1::Float64
    ΔJ2::Float64
end

function BackwardTerms(
    nx::Int,
    nu::Int,
    N::Int
)::BackwardTerms
    Fs = [FlowExpansion(nx, nu) for k = 1:(N-1)]
    Ls = [CostExpansion(nx, nu) for k = 1:(N-1)]
    Vs = [ValueExpansion(nx) for k = 1:N]
    Qs = [ActionValueExpansion(nx, nu) for k = 1:(N-1)]
    Ks = [zeros(nu, nx) for k = 1:(N-1)]
    ds = [zeros(nu) for k = 1:(N-1)]
    μ = zeros(nu, nu)
    ΔJ1 = Inf
    ΔJ2 = Inf
    return BackwardTerms(Fs, Ls, Vs, Qs, Ks, ds, μ, ΔJ1, ΔJ2)
end
