module SiLQR

using LinearAlgebra
using ForwardDiff
using HybridRobotDynamics

include("utils.jl")
include("indexing.jl")
include("objective.jl")
include("line_search.jl")
include("solver.jl")

end # module SiLQR
