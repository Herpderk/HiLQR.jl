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
include("backward.jl")
include("forward.jl")
include("solver.jl")

end # module HiLQR
