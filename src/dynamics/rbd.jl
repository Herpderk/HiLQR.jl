"""
"""
mutable struct RigidBodyFlow
    state::MechanismState
    result::DynamicsResult
    ẋ::Vector{<:DiffFloat64}
end

function RigidBodyFlow(mechanism::Mechanism)::RigidBodyFlow
    state = MechanismState(mechanism)
    result = DynamicsResult(mechanism)
    ẋ = zeros(num_positions(mechanism) + num_velocities(mechanism))
    return RigidBodyFlow(state, result, ẋ)
end

"""
"""
function(cache::RigidBodyFlow)(
    x::Vector{<:DiffFloat64},
    u::Vector{<:DiffFloat64}
)::Vector{<:DiffFloat64}
    dynamics!(cache.ẋ, cache.result, cache.state, x, u)
    return cache.ẋ
end
