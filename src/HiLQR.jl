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
include("structs/cost.jl")
include("structs/backward.jl")
include("structs/forward.jl")
include("structs/misc.jl")
include("structs/solver.jl")
include("backward.jl")
include("forward.jl")
include("solver.jl")

end # module HiLQR
