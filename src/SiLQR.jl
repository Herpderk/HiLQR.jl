module SiLQR

using LinearAlgebra
using SparseArrays
using ForwardDiff
using DiffResults
using Printf

using HybridRobotDynamics: HybridSystem, Transition, ExplicitIntegrator

include("utils.jl")
include("objective.jl")
include("expansion.jl")
include("line_search.jl")
include("solver.jl")

end # module SiLQR
