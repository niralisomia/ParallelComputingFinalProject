# ─────────────────────────────────────────────────────────────────
# Full RNA velocity pipeline
# ─────────────────────────────────────────────────────────────────

"""
    run_rna_velocity!(adata; ...)

Top-level RNA velocity runner. Adds:
  - adata.layers["velocity"]: (N × G) velocity matrix
  - adata.uns["velocity_params"]: per-gene (α, β, γ, t_switch)
  - adata.uns["velocity_genes"]: Bool mask of genes that passed quality filters

Options:
  - `pool_key`: obsm key for PCA coordinates used for kNN pooling (La Manno 2018).
                If present, u and s are smoothed over k nearest neighbors before
                fitting γ, dramatically reducing sparsity noise.
  - `pool_k`:   number of neighbors for pooling (paper uses 30–550)
  - `min_gamma`, `min_r`: gene quality filters (paper: γ≥0.1, Spearman≥0 in R impl)
"""
function run_rna_velocity!(adata::AnnData;
                           model::Symbol=:steady_state,
                           pool_key::Union{Nothing, String}="X_pca",
                           pool_k::Int=30,
                           min_gamma::Float32=0.01f0,
                           min_r::Float32=0.1f0,
                           use_pseudotime::Bool=true,
                           pseudotime_key::Union{Nothing, String}=nothing,
                           max_cells_per_gene::Int=1200,
                           n_iter::Int=6,
                           n_t::Int=60,
                           gd_steps::Int=20,
                           lr::Float32=3f-2,
                           t_end::Float32=20f0)
    s = adata.layers["spliced"]
    u = adata.layers["unspliced"]
    G = n_genes(adata)
    N = n_cells(adata)

    # ── kNN pooling (La Manno 2018) ───────────────────────────────
    s_fit = s
    u_fit = u
    if !isnothing(pool_key) && haskey(adata.obsm, pool_key)
        X_pca = adata.obsm[pool_key][:, 1:min(50, size(adata.obsm[pool_key], 2))]
        println("Applying kNN pooling (k=$pool_k) using embedding '$pool_key'...")
        s_fit = knn_pool(s, X_pca; k=pool_k)
        u_fit = knn_pool(u, X_pca; k=pool_k)
        println("Pooling complete.")
    else
        println("No PCA embedding found for pooling — fitting on raw counts.")
    end

    # ── Estimate γ on pooled data ─────────────────────────────────
    println("Estimating γ for $G genes...")
    γ_all, gene_mask = estimate_gammas(s_fit, u_fit;
                                       min_gamma=min_gamma, min_r=min_r)
    n_good = sum(gene_mask)
    println("  $n_good / $G genes passed quality filters (γ≥$min_gamma, Spearman≥$min_r)")

    V = zeros(Float32, N, G)
    # Pre-fill with a zero/NaN sentinel so filtered genes have a valid entry
    params_list = [(α=NaN32, β=1f0, γ=0f0, t_switch=NaN32) for _ in 1:G]

    # ── Optional pseudotime supervision for dynamical fitting ─────
    t_obs_all = nothing
    if model == :dynamical && use_pseudotime
        key = isnothing(pseudotime_key) ?
              (haskey(adata.obs, "palantir_pseudotime") ? "palantir_pseudotime" : nothing) :
              pseudotime_key
        if !isnothing(key) && haskey(adata.obs, key)
            pt_raw = Float32.(adata.obs[key])
            valid = .!isnan.(pt_raw)
            if any(valid)
                pmin = minimum(pt_raw[valid])
                pmax = maximum(pt_raw[valid])
                if pmax > pmin
                    t_obs_all = similar(pt_raw)
                    t_obs_all .= 0f0
                    t_obs_all[valid] .= ((pt_raw[valid] .- pmin) ./ (pmax - pmin)) .* t_end
                    println("Using pseudotime supervision from obs['$key'].")
                end
            end
        end
    end

    # One RNG per thread — avoids a data race on the global rng when the
    # dynamical model subsamples cells inside the parallel gene loop.
    thread_rngs = [Random.MersenneTwister(42 + t) for t in 1:Threads.maxthreadid()]

    println("Computing velocity for $G genes across $N cells (model=$model, $(Threads.nthreads()) threads)...")
    prog = Progress(G; desc="  genes: ")
    Threads.@threads for g in 1:G
        sg = s_fit[:, g]
        ug = u_fit[:, g]
        γ_g = γ_all[g]

        if model == :steady_state
            V[:, g] = ug .- γ_g .* sg
            params_list[g] = (α=NaN32, β=1f0, γ=γ_g, t_switch=NaN32)

        elseif model == :dynamical
            trng = thread_rngs[Threads.threadid()]
            idx_fit = N > max_cells_per_gene ? randperm(trng, N)[1:max_cells_per_gene] : collect(1:N)
            ug_fit_g = ug[idx_fit]
            sg_fit_g = sg[idx_fit]
            t_fit = isnothing(t_obs_all) ? nothing : t_obs_all[idx_fit]

            p = try
                fit_gene_params_em(ug_fit_g, sg_fit_g;
                                   n_iter=n_iter, n_t=n_t,
                                   gd_steps=gd_steps, lr=lr,
                                   t_end=t_end, t_obs=t_fit)
            catch
                (α=mean(ug_fit_g), β=1f0, γ=γ_g, t_switch=t_end * 0.35f0)
            end
            V[:, g] = compute_velocity_dynamical(ug, sg, p)
            params_list[g] = p

        else
            error("Unknown model: $model. Use :steady_state or :dynamical.")
        end
        next!(prog)
    end

    # Zero out velocities for genes that failed quality filters
    V[:, .!gene_mask] .= 0f0

    adata.layers["velocity"] = V
    adata.uns["velocity_params"] = params_list
    adata.uns["velocity_genes"] = gene_mask
    println("Done. Velocity stored in adata.layers[\"velocity\"].")
    return adata
end
