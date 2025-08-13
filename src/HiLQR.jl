module HiLQR

using LinearAlgebra
using SparseArrays
using ForwardDiff
using DiffResults
using RigidBodyDynamics
using Printf
using Plots
import Base: ==, hash

#= :
        ExplicitIntegrator,
        Transition,
        SaltationMatrix,
        HybridMode,
        HybridSystem =#

export
        HybridMode,
        Transition,
        add_transition!,
        HybridSystem,
        ProblemParameters,
        Solution,
        SolverCache,
        SolverOptions,
        solve!,
        solve,
        plot_2d_states

include("consts.jl")
include("cost.jl")
include("dynamics/RigidBodyDynamics.jl")
include("dynamics/hybrid_system.jl")
include("dynamics/integrators.jl")
include("plot.jl")
include("solver/caches/backward.jl")
include("solver/caches/forward.jl")
include("solver/caches/temporary.jl")
include("solver/interface.jl")
include("solver/backward_pass.jl")
include("solver/forward_pass.jl")
include("solver/main.jl")

end # module HiLQR
