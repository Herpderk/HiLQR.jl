"""
"""
function get_flow_jacobians!(
    Qexp::ActionValueExpansion,
    params::Parameters,
    flow::Function,
    x::Vector{Float64},
    u::Vector{Float64}
)::Nothing
    ForwardDiff.jacobian!(
        Qexp.A, δx -> params.integrator(flow, δx, u, params.Δt), x
    )
    ForwardDiff.jacobian!(
        Qexp.B, δu -> params.integrator(flow, x, δu, params.Δt), u
    )
    return nothing
end

"""
"""
function update_backward_terms!(
    bwd::BackwardTerms,
    Qexp::ActionValueExpansion,
    tmp::TemporaryArrays,
    μ::Float64,
    k::Int
)::Nothing
    # Regularized Quu
    # Qexp.Quu_reg = Qexp.Quu + μ*I
    mul!(Qexp.Quu_reg, μ, I)
    Qexp.Quu_reg .+= Qexp.Quu
    tmp.lu = lu(sparse(Qexp.Quu_reg))

    # Feedback gains
    #bwd.Ks[k] .= Qexp.Quu_reg \ Qexp.Qux
    ldiv!(bwd.Ks[k], tmp.lu, Qexp.Qux)

    # Feedforward
    #bwd.ds[k] .= Qexp.Quu_reg \ Qexp.Qu
    ldiv!(bwd.ds[k], tmp.lu, Qexp.Qu)

    # Change in cost
    bwd.ΔJ += Qexp.Qu' * bwd.ds[k]
    return nothing
end

"""
"""
function backward_pass!(
    bwd::BackwardTerms,
    Jexp::CostExpansion,
    Qexp::ActionValueExpansion,
    tmp::TemporaryArrays,
    sol::Solution,
    params::Parameters,
    μ::Float64
    #sequence::Vector{TransitionTiming}, # TODO
)::Nothing
    bwd.ΔJ = 0.0
    tmp.x .= sol.xs[end] .- params.xrefs[end]
    expand_terminal_cost!(Qexp, Jexp, params.cost, tmp.x)

    for k = (params.N-1) : -1 : 1
        get_flow_jacobians!(
            Qexp, params, params.system.modes[params.mI].flow,
            sol.xs[k], sol.us[k]
        )
        tmp.x .= sol.xs[k] .- params.xrefs[k]
        tmp.u .= sol.us[k] .- params.urefs[k]
        expand_stage_cost!(Jexp, params.cost, tmp.x, tmp.u)
        expand_Q!(Qexp, Jexp, tmp, sol.f̂s[k])
        update_backward_terms!(bwd, Qexp, tmp, μ, k)
        expand_V!(Qexp, tmp, bwd.Ks[k], bwd.ds[k])
    end
    return nothing
end
