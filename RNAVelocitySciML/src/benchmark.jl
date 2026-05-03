# ─────────────────────────────────────────────────────────────────
# Benchmarking utilities
# ─────────────────────────────────────────────────────────────────

"""
    benchmark_velocity(adata; n_runs=3, model=:steady_state)

Time the RNA velocity estimation on a dataset, reporting:
  - Wall time (seconds)
  - Allocations (bytes)
  - Throughput (cells/sec, genes/sec)
"""
function benchmark_velocity(adata::AnnData; n_runs::Int=3, model::Symbol=:steady_state)
    times = Float64[]
    allocs = Int[]

    for run in 1:n_runs
        # Deep copy to avoid side effects
        adata_copy = AnnData(
            copy(adata.X);
            layers=Dict(k => copy(v) for (k,v) in adata.layers),
            obs=adata.obs, var=adata.var, obsm=adata.obsm, uns=adata.uns
        )
        result = @timed run_rna_velocity!(adata_copy; model=model)
        push!(times, result.time)
        push!(allocs, result.bytes)
    end

    N, G = n_cells(adata), n_genes(adata)
    println("\n=== Benchmark Results (model=$model) ===")
    println("  Cells: $N | Genes: $G")
    println("  Wall time: $(round(mean(times), digits=2)) ± $(round(std(times), digits=2)) s")
    println("  Memory:    $(round(mean(allocs)/1e6, digits=1)) MB")
    println("  Throughput: $(round(N*G/mean(times)/1e6, digits=2)) M cell-gene pairs/sec")
    return (times=times, allocs=allocs)
end

"""
    cosine_similarity_score(V1, V2)

Mean cosine similarity between two velocity matrices — used to compare
our Julia implementation against scVelo reference output.
"""
function cosine_similarity_score(V1::Matrix{Float32}, V2::Matrix{Float32})
    @assert size(V1) == size(V2)
    N = size(V1, 1)
    sims = Float32[]
    for i in 1:N
        n1, n2 = norm(V1[i,:]), norm(V2[i,:])
        if n1 > 1f-10 && n2 > 1f-10
            push!(sims, dot(V1[i,:], V2[i,:]) / (n1 * n2))
        end
    end
    return mean(sims)
end
