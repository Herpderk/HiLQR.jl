"""
    plot_2d_states(
        N, nx, vis_state_idx, xs;
        animate=false, title="System Trajectory",
        xlabel="x", ylabel="y", xlim=(0.0, 10.0), ylim=(0.0, 10.0),
        markershape=:none, markercolor=:blue, markersize=2.0,
        linecolor=:blue, linewidth=2.0
    )

Plots 2D trajectories of the states in corresponding to two given indices within a given primal trajectory.
"""
function plot_2d_states(
    xs::AbstractVector{<:AbstractVector},
    vis_state_idx::Tuple{Int, Int};
    title::String = "System Trajectory",
    xlabel::String = "x",
    ylabel::String = "y",
    xlim::Tuple{<:Real, <:Real} = (0.0, 10.0),
    ylim::Tuple{<:Real, <:Real} = (0.0, 10.0),
    markershape::Symbol = :none,
    markercolor::Symbol = :blue,
    markersize::Real = 2.0,
    linecolor::Symbol = :blue,
    linewidth::Real = 2.0
)::Nothing
    for i = vis_state_idx
        !(i in 1:length(xs[1])) ? error("invalid state index!") : nothing
    end
    vis_states = [[x[vis_state_idx[i]] for x in xs] for i in 1:2]
    plt = plot(
        vis_states...,
        xlim = xlim,
        ylim = ylim,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        markershape = markershape,
        markercolor = markercolor,
        markersize = markersize,
        linecolor = linecolor,
        linewidth = linewidth,
        legend = false,
    )
    display(plt)
    return
end
