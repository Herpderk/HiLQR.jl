# Convenience type for differentiable floating-point functions
const DiffFloat64 = Union{Float64, ForwardDiff.Dual}

# Convenience symbol for a "null" transition
const NULL_TRANSITION = :NULL_TRANSITION
