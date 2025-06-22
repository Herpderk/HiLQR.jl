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
include("structs.jl")
include("solver/backward.jl")
include("solver/forward.jl")
include("solver/solver.jl")

end # module HiLQR
