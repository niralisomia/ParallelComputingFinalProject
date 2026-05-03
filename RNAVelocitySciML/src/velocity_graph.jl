# ─────────────────────────────────────────────────────────────────
# Velocity graph: transition probabilities via cosine similarity
# ─────────────────────────────────────────────────────────────────

"""
    build_velocity_graph(X, V; n_neighbors=30, scale=10.0)

Compute transition probability matrix T where T[i,j] is proportional to
exp( cosine_similarity(V_i, X_j - X_i) ) for neighbors j of cell i.

This encodes whether the velocity vector of cell i points toward cell j.
Returns a sparse N×N transition matrix (all non-negative entries).

Implementation note — `use_negative_cosines`:
scVelo separates this into two steps: (1) a signed cosine similarity graph
π_ij = cos(V_i, X_j - X_i), then (2) a softmax kernel T̃_ij = exp(π_ij/σ).
With `use_negative_cosines=True` (scVelo default), negative cosines contribute
a *repulsive* signal: a neighbor the velocity points *away from* gets weight
exp(-|cos|/σ) < 1/n_neighbors, actively pushing the transition toward other cells.
Here we combine both steps into T[i,j] ∝ exp(scale·cos_sim), so negative
cosines yield tiny positive weights (~0.007 for cos=-0.5) rather than
contributing repulsive signal. The result is slightly less directional at
progenitor cluster edges but functionally equivalent for typical datasets.
If you need signed-cosine behavior (e.g., for CellRank), split into:
  (1) cos_graph = cosine_similarity_matrix(X, V)
  (2) T = exp.(cos_graph ./ σ)  with cos_graph ≤ 0 set to −∞ or kept signed.
"""
function build_velocity_graph(X::Matrix{Float32}, V::Matrix{Float32};
                               n_neighbors::Int=30, scale::Float64=10.0)
    N = size(X, 1)
    kdtree = KDTree(X')
    idxs, _ = knn(kdtree, X', n_neighbors+1, true)

    Is = Int[]
    Js = Int[]
    Vs_sp = Float32[]

    for i in 1:N
        vi = V[i, :]
        vi_norm = norm(vi)
        vi_norm < 1f-10 && continue

        neighbors = filter(j -> j != i, idxs[i])
        weights = Float32[]
        for j in neighbors
            dij = X[j, :] - X[i, :]
            dij_norm = norm(dij)
            cos_sim = dij_norm < 1f-10 ? 0f0 : dot(vi, dij) / (vi_norm * dij_norm)
            push!(weights, exp(Float32(scale) * cos_sim))
        end

        w_sum = sum(weights)
        w_sum < 1f-10 && continue
        weights ./= w_sum  # normalize to probabilities

        for (k, j) in enumerate(neighbors)
            push!(Is, i)
            push!(Js, j)
            push!(Vs_sp, weights[k])
        end
    end

    return sparse(Is, Js, Vs_sp, N, N)
end

"""
    project_velocity_embedding(X_embed, T_graph) -> V_embed::Matrix{Float32}

Project velocity arrows into a 2D embedding (e.g. tSNE, UMAP).

Matches scVelo's `velocity_embedding.py` exactly:

    V_emb[i] = Σ_j (π̃_ij − p̄_i) · δ̃_ij

where δ̃_ij = (X_j - X_i) / ‖X_j - X_i‖  (unit displacement)
and   p̄_i  = mean(π̃_ij for j ∈ neighbors(i))  (neighborhood mean weight)

The mean-subtraction is the *density-drift correction*: it cancels the
systematic bias that arises when a cell's embedding neighborhood is
asymmetric (boundary cells, cells near cluster edges). Without it, arrows
at cluster boundaries point toward the denser region regardless of velocity
— an artifact of the local kNN geometry, not biology.
"""
function project_velocity_embedding(X_embed::Matrix{Float32},
                                     T_graph::SparseMatrixCSC{Float32})
    N = size(X_embed, 1)
    d = size(X_embed, 2)
    V_embed = zeros(Float32, N, d)

    # First pass: accumulate unit-displacement vectors and weights per cell
    # to compute the neighborhood mean weight p̄_i
    rows, cols, vals = findnz(T_graph)

    # Count neighbors and sum weights per cell (for mean computation)
    weight_sum = zeros(Float32, N)    # Σ_j π̃_ij
    n_neighbors = zeros(Int, N)       # number of actual neighbors
    disp_sum = zeros(Float32, N, d)   # Σ_j δ̃_ij (sum of unit displacements)

    for k in eachindex(rows)
        i, j = rows[k], cols[k]
        disp = X_embed[j, :] .- X_embed[i, :]
        dn = norm(disp)
        dn < 1f-10 && continue
        unit_disp = disp ./ dn
        weight_sum[i]    += vals[k]
        n_neighbors[i]   += 1
        disp_sum[i, :]   .+= unit_disp
        V_embed[i, :]    .+= vals[k] .* unit_disp
    end

    # Second pass: subtract density-drift correction p̄_i * Σ_j δ̃_ij
    # p̄_i = weight_sum[i] / n_neighbors[i]  (mean transition probability)
    for i in 1:N
        n_neighbors[i] == 0 && continue
        p_bar = weight_sum[i] / n_neighbors[i]
        V_embed[i, :] .-= p_bar .* disp_sum[i, :]
    end

    return V_embed
end


"""
    compute_velocity_consistency(T_graph, cell_types) -> Dict

For each cell type, compute the fraction of transitions that go *within*
the same cell type vs. to other types. Low within-type consistency suggests
high differentiation flux (cells are leaving that state).
"""
function compute_velocity_consistency(T_graph::SparseMatrixCSC{Float32},
                                       cell_types::Vector)
    unique_types = unique(cell_types)
    consistency = Dict{String, Float32}()
    N = size(T_graph, 1)

    for ct in unique_types
        mask = cell_types .== ct
        cell_idx = findall(mask)
        isempty(cell_idx) && continue

        within = 0f0
        total  = 0f0
        for i in cell_idx
            for j in 1:N
                w = T_graph[i, j]
                w == 0 && continue
                total += w
                cell_types[j] == ct && (within += w)
            end
        end
        consistency[string(ct)] = total > 0 ? within / total : 0f0
    end
    return consistency
end

"""
    plot_velocity_consistency(cons_normal, cons_aml; title=...)

Bar chart comparing differentiation consistency across cell types.
"""
function plot_velocity_consistency(cons_normal::Dict, cons_aml::Dict;
                                    title::String="Velocity Consistency: Normal vs AML")
    shared_types = sort(intersect(keys(cons_normal), keys(cons_aml)))
    isempty(shared_types) && return Figure()

    x = 1:length(shared_types)
    y_normal = [get(cons_normal, ct, NaN32) for ct in shared_types]
    y_aml    = [get(cons_aml,    ct, NaN32) for ct in shared_types]

    fig = Figure(size=(900, 500))
    ax = Axis(fig[1,1];
              title=title,
              xlabel="Cell Type",
              ylabel="Within-type transition probability",
              xticks=(collect(x), shared_types),
              xticklabelrotation=π/4)

    barplot!(ax, collect(x) .- 0.2, y_normal; width=0.35, color=:steelblue, label="Normal BM")
    barplot!(ax, collect(x) .+ 0.2, y_aml;    width=0.35, color=:firebrick, label="AML")
    axislegend(ax)
    return fig
end

"""
    velocity_length(V) -> Vector{Float32}

L2 norm of the velocity vector for each cell.
Cells with near-zero velocity are at steady state; high-velocity cells are
actively transitioning. Reported by scVelo as a standard diagnostic.
"""
velocity_length(V::Matrix{Float32}) = vec(sqrt.(sum(V .^ 2, dims=2)))

"""
    velocity_confidence(V, T_graph) -> Vector{Float32}

For each cell i, compute the median cosine similarity between its own velocity
vector and the velocity vectors of its neighbors (as defined by T_graph).

High confidence → neighboring cells agree on direction (coherent local dynamics).
Low confidence  → velocity is noisy or the cell is at a branching decision point.

This is the primary uncertainty proxy reported by scVelo and used by DeepVelo
for quantitative benchmarking — required for any parity claim.
"""
function velocity_confidence(V::Matrix{Float32},
                              T_graph::SparseMatrixCSC{Float32})
    N = size(V, 1)
    conf = zeros(Float32, N)
    rows, cols, _ = findnz(T_graph)

    # Group neighbor indices by source cell
    neighbor_lists = [Int[] for _ in 1:N]
    for k in eachindex(rows)
        push!(neighbor_lists[rows[k]], cols[k])
    end

    for i in 1:N
        vi = V[i, :]
        ni = norm(vi)
        ni < 1f-10 && continue
        sims = Float32[]
        for j in neighbor_lists[i]
            vj = V[j, :]
            nj = norm(vj)
            nj < 1f-10 && continue
            push!(sims, dot(vi, vj) / (ni * nj))
        end
        isempty(sims) && continue
        conf[i] = median(sims)
    end
    return conf
end
