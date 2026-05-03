# ─────────────────────────────────────────────────────────────────
# PCA projection of cells and velocities
# ─────────────────────────────────────────────────────────────────

"""
    compute_pca(X; n_components=50) -> (Z, pca_model)

Fit PCA on the gene expression matrix X (N × G) and return:
  - Z: (N × d) PCA coordinates
  - pca_model: fitted MultivariateStats PCA object
"""
function compute_pca(X::Matrix{Float32}; n_components::Int=50)
    # MultivariateStats expects (features × samples)
    pca_model = fit(PCA, X'; maxoutdim=n_components, pratio=0.99)
    Z = predict(pca_model, X')'  # (N × d)
    return Matrix{Float32}(Z), pca_model
end

"""
    project_velocity_pca(V, pca_model) -> V_pca::Matrix{Float32}

Project the high-dimensional velocity matrix V (N × G) into PCA space
using the linear transformation W of the fitted PCA model.
Since PCA is linear, velocity transforms as: v_pca = W' * v_gene
"""
function project_velocity_pca(V::Matrix{Float32}, pca_model)
    W = projection(pca_model)  # G × d loading matrix
    return (V * W)             # (N × d)
end
