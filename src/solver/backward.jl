"""
"""
function expand_dynamics!(
    Qexp::ActionValueExpansion,
    tmp::TemporaryArrays,
    params::Parameters,
    trn::Union{Transition, Nothing},
    mode::HybridMode,
    x::Vector{Float64},
    u::Vector{Float64}
)::Nothing
    if typeof(trn) == Transition
        tmp.xx1 .= trn.saltation(x, u)

        # Hybrid dynamics jacobian wrt x: salt * A
        ForwardDiff.jacobian!(
            tmp.xx2, δx -> params.igtr(mode.flow, δx, u, params.Δt), x
        )
        mul!(Qexp.A, tmp.xx1, tmp.xx2)

        # Hybrid dynamics jacobian wrt u: salt * B
        ForwardDiff.jacobian!(
            tmp.xu, δu -> params.igtr(mode.flow, x, δu, params.Δt), u
        )
        mul!(Qexp.B, tmp.xx1, tmp.xu)

    else
        ForwardDiff.jacobian!(
            Qexp.A, δx -> params.igtr(mode.flow, δx, u, params.Δt), x
        )
        ForwardDiff.jacobian!(
            Qexp.B, δu -> params.igtr(mode.flow, x, δu, params.Δt), u
        )
    end
    return nothing
end

"""
"""
function update_gains!(
    K::VecOrMat{Float64},
    d::Vector{Float64},
    Qexp::ActionValueExpansion,
    tmp::TemporaryArrays,
    μ::Float64
)::Nothing
    # Regularized Quu: Qexp.Quu_reg = Qexp.Quu + μ*I
    mul!(Qexp.Quu_reg, μ, I)
    Qexp.Quu_reg .+= Qexp.Quu
    #tmp.lu = lu!(Qexp.Quu_reg)
    #tmp.lu = lu(sparse(Qexp.Quu_reg))
    tmp.qr = qr!(Qexp.Quu_reg)

    # Feedback gains: bwd.Ks[k] .= Qexp.Quu_reg \ Qexp.Qux
    ldiv!(K, tmp.qr, Qexp.Qux)

    # Feedforward gains: bwd.ds[k] .= Qexp.Quu_reg \ Qexp.Qu
    ldiv!(d, tmp.qr, Qexp.Qu)
    return nothing
end

"""
"""
function backward_pass!(
    fwd::ForwardTerms,
    bwd::BackwardTerms,
    tmp::TemporaryArrays,
    params::Parameters,
    μ::Float64
    #sequence::Vector{TransitionTiming}, # TODO
)::Nothing
    # Reference expansion structs
    Jexp = bwd.Jexp
    Qexp = bwd.Qexp

    # Reset predicted change in cost
    bwd.ΔJ = 0.0

    # Initialize value expansion
    tmp.x .= fwd.xs[end] .- params.xrefs[end]
    expand_terminal_cost!(Qexp, Jexp, params.cost, tmp.x)

    @inbounds for k = (params.N-1) : -1 : 1
        tmp.x .= fwd.xs[k] .- params.xrefs[k]
        tmp.u .= fwd.us[k] .- params.urefs[k]
        expand_stage_cost!(Jexp, params.cost, tmp.x, tmp.u)

        expand_dynamics!(
            Qexp, tmp, params,
            fwd.trns[k].val, fwd.modes[k],
            fwd.xs[k], fwd.us[k]
        )

        expand_Q!(Qexp, Jexp, tmp, fwd.f̃s[k])
        update_gains!(bwd.Ks[k], bwd.ds[k], Qexp, tmp, μ)
        expand_V!(Qexp, tmp, bwd.Ks[k], bwd.ds[k])

        # Change in cost
        bwd.ΔJ += Qexp.Qu' * bwd.ds[k]
    end
    return nothing
end
