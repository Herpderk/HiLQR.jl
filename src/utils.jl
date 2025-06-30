# Convenience type for differentiable floating-point functions
const DiffFloat64 = Union{Float64, ForwardDiff.Dual}

# Convenience symbol for a "null" transition
const NULL_TRANSITION = :NULL_TRANSITION

# Equality function for HybridRobotDynamics.Transition
function ==(a::Transition, b::Transition)::Bool
    return (
        a.flow_I === b.flow_I &&
        a.flow_J === b.flow_J &&
        a.guard === b.guard &&
        a.reset === b.reset
    )
end

# Hash function for HybridRobotDynamics.Transition
function hash(trn::Transition, h::UInt)
    h = hash(trn.flow_I, h)
    h ⊻= hash(trn.flow_J, h)
    h ⊻= hash(trn.guard, h)
    h ⊻= hash(trn.reset, h)
    return h
end
