# ─────────────────────────────────────────────────────────────────
# Data structures & preprocessing
# ─────────────────────────────────────────────────────────────────

struct AnnData
    X::Matrix{Float32}          # (cells × genes) spliced counts
    layers::Dict{String, Matrix{Float32}}  # e.g. "spliced", "unspliced"
    obs::Dict{String, Vector}   # cell metadata
    var::Dict{String, Vector}   # gene metadata
    obsm::Dict{String, Matrix{Float32}}   # embeddings (umap, pca, ...)
    uns::Dict{String, Any}      # unstructured metadata
end

function AnnData(X::Matrix{Float32};
        layers=Dict{String,Matrix{Float32}}(),
        obs=Dict{String,Vector}(),
        var=Dict{String,Vector}(),
        obsm=Dict{String,Matrix{Float32}}(),
        uns=Dict{String,Any}())
    AnnData(X, layers, obs, var, obsm, uns)
end

n_cells(adata::AnnData) = size(adata.X, 1)
n_genes(adata::AnnData) = size(adata.X, 2)

"""
    normalize_total!(adata; target_sum=1f4)

Library-size normalize each cell so counts sum to `target_sum`, then log1p transform.
Applied in-place to both spliced and unspliced layers.
"""
function normalize_total!(adata::AnnData; target_sum::Float32=1f4)
    for key in ("spliced", "unspliced")
        haskey(adata.layers, key) || continue
        M = adata.layers[key]           # cells × genes
        row_sums = sum(M, dims=2)       # cells × 1
        row_sums[row_sums .== 0] .= 1f0
        adata.layers[key] = (M ./ row_sums) .* target_sum
    end
    # Also normalize X
    row_sums = sum(adata.X, dims=2)
    row_sums[row_sums .== 0] .= 1f0
    adata.X .= (adata.X ./ row_sums) .* target_sum
    return adata
end

"""
    log1p_transform!(adata)

Apply log(1+x) to all count matrices.
"""
function log1p_transform!(adata::AnnData)
    adata.X .= log1p.(adata.X)
    for key in keys(adata.layers)
        adata.layers[key] .= log1p.(adata.layers[key])
    end
    return adata
end

"""
    filter_genes(adata; min_shared_counts=30, n_top_genes=2000)

Keep only genes detected in both spliced and unspliced layers with sufficient counts,
then select the top highly-variable genes by variance.
"""
function filter_genes(adata::AnnData; min_shared_counts::Int=30, n_top_genes::Int=2000)
    s = get(adata.layers, "spliced",  adata.X)
    u = get(adata.layers, "unspliced", adata.X)

    # Keep genes where both layers have sufficient total counts
    s_total = vec(sum(s, dims=1))
    u_total = vec(sum(u, dims=1))
    shared_mask = (s_total .>= min_shared_counts) .& (u_total .>= min_shared_counts)

    # Among those, pick top by variance in spliced layer
    shared_idx = findall(shared_mask)
    vars = vec(var(s[:, shared_idx], dims=1))
    top_k = min(n_top_genes, length(shared_idx))
    top_idx = shared_idx[sortperm(vars, rev=true)[1:top_k]]

    new_layers = Dict(k => v[:, top_idx] for (k, v) in adata.layers)
    new_X = adata.X[:, top_idx]
    new_var = Dict(k => v[top_idx] for (k, v) in adata.var)

    return AnnData(new_X; layers=new_layers, obs=adata.obs, var=new_var,
                   obsm=adata.obsm, uns=adata.uns)
end
