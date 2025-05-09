using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra
using HybridRobotDynamics
using SiLQR
using Plots
gr()

function animate_cartpole(xs::Vector{Vector{Float64}}, qmin, qmax, l, w, draw_walls=false; filename="cartpole.gif")
    h = 0.2  # cart height

    anim = Animation()  # initialize animation object

    for x in xs
        q_c, θ = x[1], x[2]

        # Cart body
        x_cart = [q_c - w/2, q_c + w/2, q_c + w/2, q_c - w/2, q_c - w/2]
        y_cart = [0, 0, h, h, 0]

        # Pole
        pole_x = [q_c, q_c + l * sin(θ)]
        pole_y = [h, h - l * cos(θ)]

        p = plot(; xlims=(qmin-1.0, qmax+1.0), ylims=(-0.5, l+h+0.5), legend=false, aspect_ratio=1)
        plot!(p, x_cart, y_cart, lw=2, fillalpha=0.4, color=:blue)
        plot!(p, pole_x, pole_y, lw=3, color=:black)
        scatter!(p, [q_c], [0], color=:black, markersize=4)

        if (draw_walls)
            vline!(p, [qmin, qmax], lw=1, lc=:red, linestyle=:dash)
        end

        frame(anim, p)
    end

    gif(anim, filename, fps=20)
end

"""
Cartpole Model
"""

function bouncing_cartpole_model(
    mc::Float64,
    mp::Float64,
    l::Float64, # pole length
    w::Float64, # cart width
    qmin::Float64,
    qmax::Float64,
    e::Float64,
    g::Float64 = 9.81
)::HybridSystem

    nx = 4  # state dimension
    nu = 1  # control input: force on cart

    function cartpole_flow(x::Vector, u::Vector)::Vector
        θ = x[2]
        θ̇ = x[4]

        M = [mc+mp mp*l*cos(θ); mp*l*cos(θ) mp*l^2]
        C = [0.0 -mp*l*θ̇*sin(θ); zeros(1, 2)]
        τ = [0, -mp*g*l*sin(θ)]
        B = [1.0, 0.0]

        q̇ = x[3:4]
        q̈ = M \ (τ + B*u[1] - C*q̇)
        return [q̇; q̈]
    end

    # Continuous flow
    flow = (x, u) -> cartpole_flow(x, u)

    free_mode = HybridMode(flow)
    modes = Dict(:free => free_mode)

    # --- Guards ---
    g_left_cart  = x -> (x[1] - w / 2) - qmin
    g_right_cart = x -> qmax - (x[1] + w / 2)

    function pole_tip_pos(x)
        x_cart = x[1]
        θ = x[2]
        return x_cart + l*sin(θ)
    end

    g_left_pole  = x -> pole_tip_pos(x) - qmin
    g_right_pole = x -> qmax - pole_tip_pos(x)

    # --- Resets (bounce by flipping velocity components) ---
    function bounce_cart_left(x)
        x_new = copy(x)
        x_new[1] = qmin + w/2 + 1e-9
        x_new[3] = -e * x[3]  # flip cart velocity
        return x_new
    end

    function bounce_cart_right(x)
        x_new = copy(x)
        x_new[1] = qmax - w/2 - 1e-9
        x_new[3] = -e * x[3]
        return x_new
    end

    function bounce_pole_left(x)
        x_new = copy(x)
        x_new[4] = -e * x[4]  # flip pole angular velocity
        return x_new
    end

    function bounce_pole_right(x)
        x_new = copy(x)
        x_new[4] = -e * x[4]
        return x_new
    end

    # --- Transitions ---
    transitions = Dict(
        :bounce_cart_left  => Transition(flow, flow, g_left_cart,  bounce_cart_left),
        :bounce_cart_right => Transition(flow, flow, g_right_cart, bounce_cart_right),
        :bounce_pole_left  => Transition(flow, flow, g_left_pole,  bounce_pole_left),
        :bounce_pole_right => Transition(flow, flow, g_right_pole, bounce_pole_right)
    )

    for (_, t) in transitions
        add_transition!(modes[:free], modes[:free], t)
    end

    return HybridSystem(nx, nu, transitions, modes)
end

"""
Solver Setup
"""

mc = 1.2
mp = 0.16
l = 0.55
w = 0.4
qmin = -0.8
qmax = 0.8
e = 0.5

# system = get_cartpole_model(mc, mp, l)
system = bouncing_cartpole_model(mc, mp, l, w, qmin, qmax, e)

# Stage and terminal costs
Q = 1e-4 * diagm([1e-2, 1.0, 1.0, 1.0])
R = 1e-6 * I(system.nu)
Qf = 2e+3 * Q
stage(x, u) = x'*Q*x + u'*R*u
terminal(x) = x'*Qf*x

# RK4 integrator
rk4 = ExplicitIntegrator(:rk4)

# Problem parameters
N = 50
Δt = 0.1
params = SiLQR.Parameters(system, stage, terminal, rk4, N, Δt)

# Reference trajectory
xref = [0.0, pi, 0.0, 0.0]
uref = zeros(system.nu)

params.xrefs = [xref for k = 1:N]
params.urefs = [uref for k = 1:(N-1)]

params.x0 = zeros(system.nx)
params.mI = :free

"""
Solve using SiLQR
"""

sol = SiLQR.Solution(params)
sol.xs .= [[qmin + w/2; zeros(3)] for k = 1:N]

cache = SiLQR.Cache(params)
@time SiLQR.solve!(sol, cache, params; multishoot=true, αmax=0.95, stat_tol=1e-8, max_iter=300)

animate_cartpole(sol.xs, qmin, qmax, l, w, true; filename="my_bouncing_cartpole.gif")

bxs = reduce(vcat, sol.xs)
plot_2d_states(
    N, system.nx, (1,2), bxs;
    xlim=(qmin,qmax), ylim=(-1,4),
    xlabel="x", ylabel="θ"
)

nothing
