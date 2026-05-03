# ─────────────────────────────────────────────────────────────────
# Synthetic data generation
# ─────────────────────────────────────────────────────────────────

"""
    generate_synthetic_dataset(; n_cells=500, n_genes=100,
                                n_groups=3, noise_level=0.1)

Generate a synthetic single-cell RNA dataset with known ground-truth
RNA velocity, to test our parameter estimation pipeline.

Each gene has independently sampled (α, β, γ) and cells are sampled
at random times along the ODE trajectory.
"""
function generate_synthetic_dataset(; n_cells::Int=500, n_genes::Int=100,
                                      noise_level::Float64=0.1,
                                      seed::Int=42,
                                      t_end::Float32=20f0)
    rng_local = Random.MersenneTwister(seed)

    # Sample random rate parameters for each gene
    α_true = rand(rng_local, Uniform(0.5, 3.0), n_genes) .|> Float32
    β_true = rand(rng_local, Uniform(0.3, 1.5), n_genes) .|> Float32
    γ_true = rand(rng_local, Uniform(0.1, 1.0), n_genes) .|> Float32
    t_switch_true = rand(rng_local, Uniform(3.0, 8.0), n_genes) .|> Float32

    S = zeros(Float32, n_cells, n_genes)
    U = zeros(Float32, n_cells, n_genes)
    V_true = zeros(Float32, n_cells, n_genes)

    # One shared developmental time per cell — biologically, each cell has a
    # single age and all genes are evaluated at that same time point.
    t_cells = rand(rng_local, Uniform(0, t_end), n_cells) .|> Float32

    for g in 1:n_genes
        for i in 1:n_cells
            t = t_cells[i]
            if t <= t_switch_true[g]
                # Induction phase
                u = analytical_u(t, α_true[g], β_true[g], 0f0)
                s = analytical_s(t, α_true[g], β_true[g], γ_true[g], 0f0, 0f0)
                v = β_true[g] * u - γ_true[g] * s
            else
                # Repression phase: α=0, start from switch point
                u_sw = analytical_u(t_switch_true[g], α_true[g], β_true[g], 0f0)
                s_sw = analytical_s(t_switch_true[g], α_true[g], β_true[g], γ_true[g], 0f0, 0f0)
                τ = t - t_switch_true[g]
                u = analytical_u(τ, 0f0, β_true[g], u_sw)
                s = analytical_s(τ, 0f0, β_true[g], γ_true[g], u_sw, s_sw)
                v = β_true[g] * u - γ_true[g] * s
            end
            U[i, g] = max(0f0, u + Float32(randn(rng_local) * noise_level * u))
            S[i, g] = max(0f0, s + Float32(randn(rng_local) * noise_level * s))
            V_true[i, g] = v
        end
    end

    adata = AnnData(
        S;
        layers=Dict("spliced" => S, "unspliced" => U),
        obs=Dict{String,Vector}("cell_time" => t_cells),
        uns=Dict{String,Any}(
            "true_params" => (α=α_true, β=β_true, γ=γ_true, t_switch=t_switch_true),
            "true_velocity" => V_true
        )
    )
    return adata
end
