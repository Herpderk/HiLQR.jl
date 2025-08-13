"""
    SaltationMatrix(flowI, flowJ, guard, reset)

Derives the saltation matrix function for a given hybrid transition. Assumes the following function arguments: `flow(x, u)`, `guard(x)`, and `reset(x)`.
"""
mutable struct SaltationMatrix
    flowI::Function
    flowJ::Function
    guard::Function
    reset::Function
    ẋI::Vector{<:DiffFloat64}
    ẋJ::Vector{<:DiffFloat64}
    ∇g::Vector{<:DiffFloat64}
    ∇R::Matrix{<:DiffFloat64}
    xtmp::Vector{<:DiffFloat64}
    function SaltationMatrix(
        flowI::Function,
        flowJ::Function,
        guard::Function,
        reset::Function,
        nx::Int
    )::SaltationMatrix
        ẋI = zeros(nx)
        ẋJ = zeros(nx)
        ∇g = zeros(nx)
        ∇R = zeros(nx, nx)
        xtmp = zeros(nx)
        return new(
            flowI,
            flowJ,
            guard,
            reset,
            ẋI,
            ẋJ,
            ∇g,
            ∇R,
            xtmp
        )
    end
end

"""
    compute_saltation_matrix!(Ξ, x, u)

Computes the saltation matrix for the corresponding hybrid transition in place.
"""
function (cache::SaltationMatrix)(
    Ξ::Matrix{<:DiffFloat64},
    x::Vector{<:DiffFloat64},
    u::Vector{<:DiffFloat64}
)::Nothing
    # Buffer arrays to be used for saltation matrix computation
    BLAS.copy!(cache.ẋI, cache.flowI(x, u))
    BLAS.copy!(cache.ẋJ, cache.flowJ(cache.reset(x), u))
    ForwardDiff.gradient!(cache.∇g, δx -> cache.guard(δx), x)
    ForwardDiff.jacobian!(cache.∇R, cache.reset, x)

    # Ξ = ∇R + (ẋJ - ∇R*ẋI) * ∇g' / (∇g'*ẋI)
    # (ẋJ - ∇R*ẋI) * ∇g'
    BLAS.copy!(cache.xtmp, cache.ẋJ)
    mul!(cache.xtmp, cache.∇R, cache.ẋI, -1.0, 1.0)
    mul!(Ξ, cache.xtmp, cache.∇g')

    # ... / ∇g'*ẋI
    rdiv!(Ξ, cache.∇g' *cache. ẋI)

    # ∇R + ...
    axpy!(1.0, cache.∇R, Ξ)
    return
end


"""
    Transition(flowI, flowJ, guard, reset)

Contains all hybrid system objects pertaining to a hybrid transition. Assumes the following function arguments: `flow(x, u)`, `reset(x)`, and `guard(x)`.
"""
struct Transition
    flowI::Function
    flowJ::Function
    guard::Function
    reset::Function
    saltationmatrix!::SaltationMatrix
end

function Transition(
    flowI::Function,
    flowJ::Function,
    guard::Function,
    reset::Function,
    nx::Int
)::Transition
    saltationmatrix! = SaltationMatrix(flowI, flowJ, guard, reset, nx)
    return Transition(flowI, flowJ, guard, reset, saltationmatrix!)
end

# Equality function for Transition
function ==(a::Transition, b::Transition)::Bool
    return (
        a.flowI === b.flowI &&
        a.flowJ === b.flowJ &&
        a.guard === b.guard &&
        a.reset === b.reset
    )
end

# Hash function for Transition
function hash(trn::Transition, h::UInt)
    h = hash(trn.flowI, h)
    h ⊻= hash(trn.flowJ, h)
    h ⊻= hash(trn.guard, h)
    h ⊻= hash(trn.reset, h)
    return h
end


"""
    HybridMode(flow, transitions=Dict())

Contains the flow and feasible transitions of the corresponding hybrid mode. Transitions and adjacent modes are stored as key-value pairs. Assumes the following arguments for flow: `flow(ẋ, x, u)`.
"""
mutable struct HybridMode
    flow::Function
    transitions::Dict{Transition, HybridMode}
end

function HybridMode(flow::Function)::HybridMode
    transitions = Dict{Transition, HybridMode}()
    return HybridMode(flow, transitions)
end


"""
    add_transition!(modeI, modeJ, transition)

Adds a given transition and adjacent mode J to the feasible transition dictionary of mode I.
"""
function add_transition!(
    modeI::HybridMode,
    modeJ::HybridMode,
    transition::Transition
)::Nothing
    modeI.transitions[transition] = modeJ
    return
end


"""
    HybridSystem(transitions, nx, nu)

Contains a given hybrid system's modes, transitions, and dimensions.
"""
mutable struct HybridSystem
    modes::Dict{Symbol, HybridMode}
    transitions::Dict{Symbol, Transition}
    nx::Int
    nu::Int
end

function HybridSystem(
    modes::Dict{Symbol, HybridMode},
    nx::Int,
    nu::Int
)::HybridSystem
    transitions = Dict(Tuple{Symbol, Transition}[])
    return HybridSystem(modes, transitions, nx, nu)
end
