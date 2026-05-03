# ─────────────────────────────────────────────────────────────────
# Visualization with CairoMakie
# ─────────────────────────────────────────────────────────────────

"""
    make_color_dict(cell_types) -> Dict

Map unique cell type labels to distinct colors using tab20 colormap.
Uses to_colormap (returns a Vector) to avoid integer-indexing a ColorGradient.
"""
function make_color_dict(cell_types::Vector)
    unique_types = unique(cell_types)
    n = length(unique_types)
    palette = to_colormap(:tab20)  # Vector{RGBAf} with 20 entries
    type_to_color = Dict(t => palette[mod1(i, length(palette))]
                         for (i, t) in enumerate(unique_types))
    return type_to_color, unique_types
end

"""
    plot_velocity_embedding(X_embed, V_embed, cell_types;
                            title="RNA Velocity", arrow_scale=0.3)

Scatter plot of cells in 2D embedding colored by cell type,
with RNA velocity arrows overlaid.
"""
function plot_velocity_embedding(X_embed::Matrix{Float32},
                                  V_embed::Matrix{Float32},
                                  cell_types::Vector;
                                  title::String="RNA Velocity",
                                  arrow_scale::Float32=0.3f0,
                                  subsample::Int=500)
    fig = Figure(size=(900, 700))
    ax = Axis(fig[1,1]; title=title,
              xlabel="Component 1", ylabel="Component 2")

    type_to_color, unique_types = make_color_dict(cell_types)
    colors = [type_to_color[t] for t in cell_types]

    scatter!(ax, X_embed[:, 1], X_embed[:, 2];
             color=colors, markersize=4, alpha=0.6)

    # Subsample arrows for clarity
    N = size(X_embed, 1)
    idx = randperm(N)[1:min(subsample, N)]
    V_scale = V_embed[idx, :] .* arrow_scale
    arrows!(ax,
            X_embed[idx, 1], X_embed[idx, 2],
            V_scale[:, 1],   V_scale[:, 2];
            arrowsize=8, linewidth=0.5, color=(:black, 1))

    elements = [MarkerElement(color=type_to_color[t], marker=:circle)
                for t in unique_types]
    Legend(fig[1,2], elements, string.(unique_types); framevisible=false)
    return fig
end

"""
    plot_phase_portrait(u, s, v; gene_name="", γ=nothing)

Plot the (u, s) phase portrait for a single gene with velocity coloring.
"""
function plot_phase_portrait(u::Vector{Float32}, s::Vector{Float32},
                              v::Vector{Float32};
                              gene_name::String="",
                              γ::Union{Float32, Nothing}=nothing)
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1,1];
              title="Phase portrait: $gene_name",
              xlabel="Spliced (s)", ylabel="Unspliced (u)")

    vlim = quantile(abs.(v), 0.96f0)
    vlim = vlim > 0 ? vlim : 1f0
    scatter!(ax, s, u; color=v, colormap=:viridis,
             colorrange=(-vlim, vlim), markersize=4)
    Colorbar(fig[1,2]; colormap=:viridis, limits=(-vlim, vlim), label="velocity")

    if !isnothing(γ)
        s_range = range(0, maximum(s)*1.1, length=100)
        lines!(ax, collect(s_range), γ .* s_range;
               color=:black, linewidth=2, linestyle=:dash,
               label="steady state (γ=$(round(γ, digits=2)))")
        axislegend(ax)
    end
    return fig
end

"""
    plot_trajectories_pca(Z_pca, trajectories, cell_types;
                           pc1=1, pc2=2, title="Neural ODE Trajectories")

Plot neural ODE trajectories overlaid on the PCA scatter plot.
"""
function plot_trajectories_pca(Z_pca::Matrix{Float32},
                                trajectories::Array{Float32, 3},
                                cell_types::Vector;
                                pc1::Int=1, pc2::Int=2,
                                title::String="Neural ODE Trajectories")
    K = size(trajectories, 1)
    fig = Figure(size=(900, 700))
    ax = Axis(fig[1,1]; title=title,
              xlabel="PC$pc1", ylabel="PC$pc2")

    type_to_color, unique_types = make_color_dict(cell_types)
    colors = [type_to_color[t] for t in cell_types]
    scatter!(ax, Z_pca[:, pc1], Z_pca[:, pc2];
             color=colors, markersize=3, alpha=0.4)

    for k in 1:K
        traj = trajectories[k, :, :]  # (d, n_steps)
        lines!(ax, traj[pc1, :], traj[pc2, :];
               color=(:darkorange, 0.8), linewidth=2)
        scatter!(ax, [traj[pc1, 1]], [traj[pc2, 1]];
                 color=:darkorange, markersize=8, marker=:circle)
        scatter!(ax, [traj[pc1, end]], [traj[pc2, end]];
                 color=:red, markersize=8, marker=:utriangle)
    end

    elements = [MarkerElement(color=type_to_color[t], marker=:circle)
                for t in unique_types]
    Legend(fig[1,2], elements, string.(unique_types); framevisible=false)
    return fig
end

"""
    plot_loss_history(loss_history; title="Training Loss")
"""
function plot_loss_history(loss_history::Vector{Float32};
                            title::String="Neural ODE Training Loss")
    fig = Figure(size=(700, 400))
    ax = Axis(fig[1,1]; title=title,
              xlabel="Epoch", ylabel="MSE Loss", yscale=log10)
    lines!(ax, 1:length(loss_history), loss_history; color=:steelblue, linewidth=2)
    return fig
end
