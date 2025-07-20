"""
    SaltationMatrix(flowI, flowJ, reset, guard)

Derives the saltation matrix function for a given hybrid transition. Assumes the following function arguments: `flow(x, u)`, `reset(x)`, and `guard(x, u)`.
"""
struct SaltationMatrix
    flowI::Function
    flowJ::Function
    reset::Function
    guard::Function
    ẋI::Vector{<:DiffFloat64}
    ẋJ::Vector{<:DiffFloat64}
    xJ::Vector{<:DiffFloat64}
    ∇g::Vector{<:DiffFloat64}
    ∇R::Matrix{<:DiffFloat64}
    xtmp::Vector{<:DiffFloat64}
    function SaltationMatrix(
        flowI::Function,
        flowJ::Function,
        reset::Function,
        guard::Function,
        nx::Int
    )::SaltationMatrix
        ẋI = zeros(nx)
        ẋJ = zeros(nx)
        xJ = zeros(nx)
        ∇g = zeros(nx)
        ∇R = zeros(nx, nx)
        xtmp = zeros(nx)
        return new(
            flowI,
            flowJ,
            reset,
            guard,
            ẋI,
            ẋJ,
            xJ,
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
function (trn_cache::SaltationMatrix)(
    Ξ::Matrix{<:DiffFloat64},
    x::Vector{<:DiffFloat64},
    u::Vector{<:DiffFloat64}
)::Nothing
    # Buffer arrays to be used for saltation matrix computation
    BLAS.copy!(trn_cache.ẋI, trn_cache.flowI(x, u))
    BLAS.copy!(trn_cache.ẋJ, trn_cache.flowJ(x, u))
    BLAS.copy!(trn_cache.xJ, trn_cache.reset(x))
    ForwardDiff.gradient!(trn_cache.∇g, δx -> trn_cache.guard(δx, u), x)
    ForwardDiff.jacobian!(trn_cache.∇R, trn_cache.reset, trn_cache.xtmp, x)

    # Ξ = ∇R + (ẋJ - ∇R*ẋI) * ∇g' / (∇g'*ẋI)
    # (ẋJ - ∇R*ẋI) * ∇g'
    BLAS.copy!(trn_cache.xtmp, trn_cache.ẋJ)
    mul!(trn_cache.xtmp, trn_cache.∇R, trn_cache.ẋI, -1.0, 1.0)
    mul!(Ξ, trn_cache.xtmp, trn_cache.∇g')

    # ... / ∇g'*ẋI
    rdiv!(Ξ, trn_cache.∇g' * ẋI)

    # ∇R + ...
    axpy!(1.0, trn_cache.∇R, Ξ)
    return
end


"""
    Transition(flowI, flowJ, reset, guard)

Contains all hybrid system objects pertaining to a hybrid transition. Assumes the following function arguments: `flow(ẋ, x, u)`, `reset(xJ, x)`, and `guard(x, u)`.
"""
struct Transition
    flowI::Function
    flowJ::Function
    reset::Function
    guard::Function
    saltation::SaltationMatrix
end

function Transition(
    flowI::Function,
    flowJ::Function,
    reset::Function,
    guard::Function,
    nx::Int
)::Transition
    saltation = SaltationMatrix(flowI, flowJ, reset, guard, nx)
    return Transition(flowI, flowJ, reset, guard, saltation)
end

# Equality function for Transition
function ==(a::Transition, b::Transition)::Bool
    return (
        a.flowI === b.flowI &&
        a.flowJ === b.flowJ &&
        a.reset === b.reset &&
        a.guard === b.guard
    )
end

# Hash function for Transition
function hash(trn::Transition, h::UInt)
    h = hash(trn.flowI, h)
    h ⊻= hash(trn.flowJ, h)
    h ⊻= hash(trn.reset, h)
    h ⊻= hash(trn.guard, h)
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
    add_transition(modeI, modeJ, transition)

Adds a given transition and adjacent mode J to the feasible transition dictionary of mode I.
"""
function add_transition(
    modeI::HybridMode,
    modeJ::HybridMode,
    transition::Transition
)::Nothing
    modeI.transitions[transition] = modeJ
    return
end

"""
    add_transition!(modeI, modeJ, guard, reset)

Constructs a transition using the given guard, reset, and modes and adds it to the feasible transition dictionary of modeI.
"""
function add_transition(
    modeI::HybridMode,
    modeJ::HybridMode,
    reset::Function,
    guard::Function,
    nx::Int
)::Nothing
    transition = Transition(modeI.flow, modeJ.flow, reset, guard, nx)
    add_transition!(modeI, modeJ, transition)
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
