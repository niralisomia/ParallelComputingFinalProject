#!/usr/bin/env julia
"""
RNA velocity analysis — Setty et al. 2019 bone marrow dataset.

Run from the project root with:
    julia --project=RNAVelocity --threads=auto RNAVelocity/scripts/bonemarrow_analysis.jl

Prerequisites:
    1. julia setup.jl            (install packages, once)
    2. python3 download_data.py  (download bonemarrow.h5ad, once)
"""

# Absolute paths — computed before any Pkg calls so they work regardless of
# the working directory from which Julia is invoked.
SCRIPT_DIR  = @__DIR__                     # …/RNAVelocity/scripts
PKG_DIR     = dirname(SCRIPT_DIR)          # …/RNAVelocity  (contains Project.toml)
PROJECT_DIR = dirname(PKG_DIR)             # …/Project      (contains bonemarrow.h5ad)

using Pkg
Pkg.activate(PKG_DIR)   # activate env using absolute path → deps (Muon, CairoMakie …) available

using Muon
using SparseArrays

for _f in ["preprocessing.jl", "velocity.jl", "graph.jl", "plotting.jl"]
    include(joinpath(PKG_DIR, "src", _f))
end

println("Threads: $(Threads.nthreads())")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

data_path = joinpath(PROJECT_DIR, "bonemarrow.h5ad")
isfile(data_path) || error("Data not found: $data_path\nRun `python3 download_data.py` first.")

println("\n── Loading data ──────────────────────────────────────────────")
adata = readh5ad(data_path)

n_cells, n_genes = size(adata.X)
println("Cells × genes : $n_cells × $n_genes")
println("Layers        : $(collect(keys(adata.layers)))")
println("Embeddings    : $(collect(keys(adata.obsm)))")
println("Obs columns   : $(names(adata.obs))")

spliced   = Matrix{Float64}(adata.layers["spliced"])
unspliced = Matrix{Float64}(adata.layers["unspliced"])
tsne      = Matrix{Float64}(adata.obsm["X_tsne"])

cell_types = String.(adata.obs[!, "clusters"])
println("Types : $(unique(cell_types))")

# Differentiation order: stem cells → progenitors → committed lineages
CT_ORDER = [
    "HSC_1", "HSC_2",       # haematopoietic stem cells (most primitive)
    "Precursors",            # early multipotent progenitors
    "CLP",                   # common lymphoid progenitor (lymphoid branch)
    "Ery_1", "Ery_2",       # erythroid lineage → red blood cells
    "Mega",                  # megakaryocytes → platelets (erythroid-related)
    "Mono_1", "Mono_2",     # monocytes (myeloid branch)
    "DCs",                   # dendritic cells (myeloid branch)
]

# Colors: grey for stem cells, purple for progenitors, red for erythroid,
#         pink for megakaryocyte, orange for myeloid, green for DCs
CT_COLORS = Dict(
    "HSC_1"      => "#252525",   # dark grey
    "HSC_2"      => "#737373",   # mid grey
    "Precursors" => "#bdbdbd",   # light grey
    "CLP"        => "#6a51a3",   # purple
    "Ery_1"      => "#cb181d",   # dark red
    "Ery_2"      => "#fb6a4a",   # salmon
    "Mega"       => "#e377c2",   # pink
    "Mono_1"     => "#238b45",   # dark green
    "Mono_2"     => "#74c476",   # light green
    "DCs"        => "#2171b5",   # blue
)

# Gene names — Muon.jl 0.2+ exposes var_names; fall back to first var column
gene_names = String.(adata.var_names)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

println("\n── Preprocessing ─────────────────────────────────────────────")

# Normalise + log-transform spliced counts for HVG selection and PCA
S_norm = normalize_per_cell(spliced)
log1p_transform!(S_norm)

# Select highly variable genes
print("  HVG selection...")
hvg_idx = select_hvg(S_norm; n_top_genes=2000)
println(" $(length(hvg_idx)) genes selected")

spliced_hvg   = spliced[:, hvg_idx]
unspliced_hvg = unspliced[:, hvg_idx]
hvg_names     = gene_names[hvg_idx]

# PCA (30 components, used for kNN)
print("  PCA (30 PCs)...")
pca = compute_pca(S_norm[:, hvg_idx]; n_pcs=30)
println(" done  size=$(size(pca))")

# kNN graph
print("  kNN graph (k=30)...")
knn_idx, _ = compute_neighbors(pca; n_neighbors=30)
conn       = build_connectivities(knn_idx, n_cells)
println(" done  nnz=$(nnz(conn))")

# First-order moments (smoothed spliced / unspliced)
print("  Computing moments...")
Ms, Mu = compute_moments(spliced_hvg, unspliced_hvg, conn)
println(" done")

# ─────────────────────────────────────────────────────────────────────────────
# 2b. Preprocessing visualisation
# ─────────────────────────────────────────────────────────────────────────────

println("\n── Preprocessing visualisation ───────────────────────────────")
mkpath(joinpath(PROJECT_DIR, "output"))

# PCA scatter coloured by cell type — shows that PCA separates cell identities
plot_pca(pca, cell_types;
    order    = CT_ORDER,
    colors   = CT_COLORS,
    title    = "PCA — bone marrow cell types",
    filename = joinpath(PROJECT_DIR, "output", "pca_cell_types.png"),
)
println("  Saved: output/pca_cell_types.png")

# t-SNE coloured by cell type — same colours as PCA for direct comparison
plot_cell_types(tsne, cell_types;
    order    = CT_ORDER,
    colors   = CT_COLORS,
    xlabel   = "t-SNE 1",
    ylabel   = "t-SNE 2",
    title    = "t-SNE — bone marrow cell types",
    filename = joinpath(PROJECT_DIR, "output", "tsne_cell_types.png"),
)
println("  Saved: output/tsne_cell_types.png")

# Moments comparison for ELANE (neutrophil marker):
# left panel  = raw spliced counts  (noisy, many zeros)
# right panel = smoothed Ms         (kNN-averaged, much smoother)
let gene = "ELANE"
    g_idx = findfirst(g -> occursin(gene, g), hvg_names)
    if !isnothing(g_idx)
        plot_moments_comparison(
            tsne,
            spliced_hvg[:, g_idx],
            Ms[:, g_idx];
            gene_name = gene,
            filename  = joinpath(PROJECT_DIR, "output", "moments_$(gene).png"),
        )
        println("  Saved: output/moments_$(gene).png")
    else
        println("  $gene not in HVG set — skipping moments plot")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Velocity estimation (steady-state model)
# ─────────────────────────────────────────────────────────────────────────────

println("\n── Velocity estimation (steady-state) ────────────────────────")
print("  Fitting γ per gene ($(length(hvg_idx)) genes, $(Threads.nthreads()) threads)...")
γ        = fit_gamma(Ms, Mu; quantile_fit=0.95)
velocity = compute_velocity_steadystate(Ms, Mu, γ)
n_active = sum(vec(maximum(abs.(velocity); dims=1)) .> 1e-10)
println(" done")
println("  Active genes : $n_active / $(length(hvg_idx))")
println("  γ range      : [$(round(minimum(γ),digits=3)), $(round(maximum(γ),digits=3))]")

plot_gamma_distribution(γ;
    filename = joinpath(PROJECT_DIR, "output", "gamma_distribution.png"),
)
println("  Saved: output/gamma_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Velocity graph
# ─────────────────────────────────────────────────────────────────────────────

println("\n── Velocity graph ────────────────────────────────────────────")
print("  Building transition matrix...")
T = velocity_graph(velocity, Ms, knn_idx)
println(" done  nnz=$(nnz(T))")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Project to UMAP embedding
# ─────────────────────────────────────────────────────────────────────────────

println("\n── Embedding projection ──────────────────────────────────────")
V_embed = velocity_embedding(T, tsne)
v_speed = vec(sqrt.(sum(V_embed .^ 2; dims=2)))
println("  Speed range  : [$(round(minimum(v_speed),digits=4)), $(round(maximum(v_speed),digits=4))]")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualise
# ─────────────────────────────────────────────────────────────────────────────

println("\n── Plotting ──────────────────────────────────────────────────")

out_dir = joinpath(PROJECT_DIR, "output")
mkpath(out_dir)

# Cell-type map
plot_cell_types(tsne, cell_types;
    order    = CT_ORDER,
    colors   = CT_COLORS,
    xlabel   = "t-SNE 1",
    ylabel   = "t-SNE 2",
    title    = "Bone marrow — cell types (Setty 2019)",
    filename = joinpath(out_dir, "cell_types.png"),
)
println("  Saved: output/cell_types.png")

# Velocity embedding
plot_velocity_embedding(tsne, V_embed;
    title       = "RNA Velocity — bone marrow (Setty 2019)",
    arrow_scale = 0.4,
    subsample   = 0.25,
    filename    = joinpath(out_dir, "velocity_embedding.png"),
)
println("  Saved: output/velocity_embedding.png")

# Phase portraits for a few marker genes
for (gene, annotation) in [("GATA1", "erythroid"), ("MPO", "myeloid"), ("ELANE", "neutrophil")]
    idx = findfirst(g -> occursin(gene, g), hvg_names)
    if isnothing(idx)
        println("  $gene not in HVG set — skipping")
        continue
    end
    plot_phase_portrait(
        Ms[:, idx], Mu[:, idx], velocity[:, idx];
        γ         = γ[idx],
        gene_name = "$gene ($annotation)",
        filename  = joinpath(out_dir, "phase_$(gene).png"),
    )
    println("  Saved: output/phase_$(gene).png")
end

println("\nDone. Output written to: $out_dir")
