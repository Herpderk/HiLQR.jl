module SiLQR

using LinearAlgebra
using ForwardDiff
using DiffResults
using HybridRobotDynamics

include("utils.jl")
include("objective.jl")
include("line_search.jl")
include("solver.jl")

end # module SiLQR
