"""
    rk4(x0, u0, Δt, flow)

Explicit integrator using the fourth order Runge-Kutta method. Assumes the following dynamics function: `flow(x, u)`
"""
function rk4(
    x0::Vector{<:DiffFloat64},
    u0::Vector{<:DiffFloat64},
    Δt::Float64,
    flow::Function
)::Vector{<:DiffFloat64}
    # Reference cache vectors
    #= k1 = cache.k1
    k2 = cache.k2
    k3 = cache.k3
    k4 = cache.k4
    tmp = cache.tmp =#

    # Get time step multiples
    #= t2 = T(Δt/2)
    t3 = T(Δt/3)
    t6 = T(Δt/6) =#

    k1 = flow(x0, u0)

    k2 = flow(x0 + Δt/2 * k1, u0)
    #= copy(tmp, x0)
    axpy(t2, k1, tmp)
    flow(k2, tmp, u0) =#

    k3 = flow(x0 + Δt/2*k2, u0)
    #= copy(tmp, x0)
    axpy(t2, k2, tmp)
    flow(k3, tmp, u0) =#

    k4 = flow(x0 + Δt*k3, u0)
    #= copy(tmp, x0)
    axpy(Δt, k3, tmp)
    flow(k4, tmp, u0) =#

    return x0 + Δt/6*(k1 + 2*k2 + 2*k3 + k4)
    #= copy(x1, x0)
    axpy(t6, k1, x1)
    axpy(t3, k2, x1)
    axpy(t3, k3, x1)
    axpy(t6, k4, x1) =#
end
