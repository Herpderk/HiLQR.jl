module HiLQR

using LinearAlgebra
using SparseArrays
using ForwardDiff
using DiffResults
using Printf

using HybridRobotDynamics:
        ExplicitIntegrator,
        Transition,
        SaltationMatrix,
        HybridMode,
        HybridSystem

include("utils.jl")
include("objective.jl")
include("structs.jl")
include("expansion.jl")
include("solver/backward.jl")
include("solver/forward.jl")
include("solver/solver.jl")

end # module HiLQR
