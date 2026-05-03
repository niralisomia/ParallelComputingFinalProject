# ─────────────────────────────────────────────────────────────────
# Gene-wise parameter estimation
# ─────────────────────────────────────────────────────────────────

"""
    knn_pool(M, X_pca; k=30) -> M_pooled::Matrix{Float32}

kNN cell pooling (La Manno 2018): for each cell, replace its count vector
with the mean across its k nearest neighbours (including itself).
This dramatically reduces per-gene sparsity before fitting γ.

M: (N × G) count matrix (spliced or unspliced)
X_pca: (N × d) PCA coordinates used to define cell neighbourhoods
"""
function knn_pool(M::Matrix{Float32}, X_pca::Matrix{Float32}; k::Int=30)
    N = size(M, 1)
    kdtree = KDTree(X_pca')
    idxs, _ = knn(kdtree, X_pca', k, true)
    M_pooled = similar(M)
    for i in 1:N
        M_pooled[i, :] = vec(mean(M[idxs[i], :], dims=1))
    end
    return M_pooled
end

# Helper: compute percentile
percentile_(x::AbstractVector, p::Float64) = quantile(x, p/100)

"""
    spearman_cor(x, y) -> Float32

Spearman rank correlation between two vectors.
"""
function spearman_cor(x::Vector{Float32}, y::Vector{Float32})
    n = length(x)
    n < 3 && return 0f0
    rx = Float32.(invperm(sortperm(x)))
    ry = Float32.(invperm(sortperm(y)))
    return Float32(cor(rx, ry))
end

"""
    estimate_gammas(s, u; hi_pct=95, lo_pct=5, min_gamma, min_r) -> (γ, mask)

Steady-state slope estimation matching La Manno 2018 Supplementary Note 2 §2.

The phase portrait of u vs s has two steady-state anchors:
  - Upper-right corner (high s + u): cells near the induction steady state
  - Origin (low s + u):             cells near the repression steady state

Using BOTH anchor regions for the OLS gives an unbiased γ estimate even for
transient genes that are never close to just one steady state. Using only the
upper quantile (as many implementations do) systematically underestimates γ
because the regression loses the origin-anchoring lower-quantile cells.

Quality filters:
  - min_gamma: discard genes with γ < min_gamma (degenerate / near-zero slope)
  - min_r:     discard genes with Spearman(s, u) < min_r (no phase-space structure)

Returns (γ::Vector{Float32}, good::BitVector).
"""
function estimate_gammas(s::Matrix{Float32}, u::Matrix{Float32};
                         hi_pct::Float64=95.0,
                         lo_pct::Float64=5.0,
                         min_gamma::Float32=0.01f0,
                         min_r::Float32=0.1f0)
    G = size(s, 2)
    γ = zeros(Float32, G)
    good = trues(G)
    for g in 1:G
        sg = s[:, g]
        ug = u[:, g]

        # Gene quality: Spearman correlation between s and u
        r = spearman_cor(sg, ug)
        if r < min_r
            good[g] = false
            continue
        end

        # Select cells in BOTH extreme quantiles of (s + u):
        #   upper quantile → near induction steady state (upper-right of phase portrait)
        #   lower quantile → near repression steady state (near origin)
        # This two-anchor fit is the velocyto convention; it prevents γ bias on
        # transient genes that have no cells near one of the two steady states.
        total = sg .+ ug
        hi_thresh = percentile_(total, hi_pct)
        lo_thresh = percentile_(total, lo_pct)
        mask = (total .>= hi_thresh) .| (total .<= lo_thresh)
        sum(mask) < 5 && (mask = trues(length(sg)))

        ss = sg[mask]
        uu = ug[mask]
        # OLS through origin for u ≈ γ*s: γ = (s'·s)⁻¹ (s'·u)
        denom = dot(ss, ss)
        γ_g = denom > 0 ? dot(ss, uu) / denom : 0f0

        # Gene quality: minimum gamma
        if γ_g < min_gamma
            good[g] = false
            continue
        end
        γ[g] = γ_g
    end
    return γ, good
end

"""
    compute_velocity_steady_state(s, u, γ) -> V::Matrix{Float32}

RNA velocity under steady-state model:
    v_g(cell) = u_g - γ_g * s_g

Returns a (cells × genes) velocity matrix.
Positive v means gene g is being upregulated in that cell;
negative v means downregulation.
"""
function compute_velocity_steady_state(s::Matrix{Float32}, u::Matrix{Float32},
                                        γ::Vector{Float32})
    return u .- s .* γ'  # (N × G)
end
