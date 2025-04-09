"""
    TrajectoryCost(stage_cost, terminal_cost)

Callable struct containing a given problem's dimensions, indices, and cost functions.
"""
struct TrajectoryCost
    stage::Function
    terminal::Function
    function TrajectoryCost(
        stage_cost::Function,
        terminal_cost::Function
    )::TrajectoryCost
        annotated_stage_cost(
            x::Vector{<:DiffFloat64},
            u::Vector{<:DiffFloat64}
        ) = stage_cost(x, u)::DiffFloat64

        annotated_terminal_cost(
            x::Vector{<:DiffFloat64}
        ) = terminal_cost(x, u)::DiffFloat64

        return new(annotated_stage_cost, annotated_terminal_cost)
    end
end

"""
    cost(xrefs, urefs, xs, us)

Callable struct method for the `TrajectoryCost` struct that computes the accumulated cost over a trajectory given a sequence of references.
"""
function (cost::TrajectoryCost)(
    xrefs::Vector{Vector{Float64}},
    urefs::Vector{Vector{Float64}},
    xs::Vector{Vector{Float64}},
    us::Vector{Vector{Float64}}
)::Float64
    N = length(xs)
    Js = zeros(N)
    @simd for k = 1:(N-1)
        Js[k] = cost.stage(xs[k] - xrefs[k], us[k] - urefs[k])
    end
    Js[end] = cost.terminal(xs[end] - xrefs[end])
    return sum(Js)
end
