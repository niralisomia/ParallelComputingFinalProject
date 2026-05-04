# ─────────────────────────────────────────────────────────────────
# Data loading: read .h5ad (AnnData HDF5 format)
# ─────────────────────────────────────────────────────────────────

"""
    read_csr_matrix(grp) -> Matrix{Float32}

Read a CSR sparse matrix stored in h5ad format (data/indices/indptr)
and return a dense Float32 matrix (cells × genes).
"""
function read_csr_matrix(grp::HDF5.Group)
    data    = Float32.(read(grp["data"]))
    indices = Int.(read(grp["indices"])) .+ 1   # 0-based → 1-based
    indptr  = Int.(read(grp["indptr"]))  .+ 1

    n_rows = length(indptr) - 1

    # Read n_cols from the HDF5 shape attribute (reliable even if last col is all-zero)
    attr = HDF5.attributes(grp)
    n_cols = if haskey(attr, "shape")
        Int(read(attr["shape"])[2])
    elseif haskey(attr, "h5sparse_shape")
        Int(read(attr["h5sparse_shape"])[2])
    else
        isempty(indices) ? 0 : maximum(indices)
    end

    M = zeros(Float32, n_rows, n_cols)
    for row in 1:n_rows
        for k in indptr[row]:(indptr[row+1]-1)
            M[row, indices[k]] = data[k]
        end
    end
    return M
end

"""
    read_h5ad_matrix(f, key) -> Matrix{Float32}

Read a matrix from an h5ad file — handles both dense arrays and CSR sparse groups.
"""
function read_h5ad_matrix(f, key::String)
    obj = f[key]
    if obj isa HDF5.Group
        # Sparse CSR format: has data/indices/indptr
        if haskey(obj, "data") && haskey(obj, "indptr")
            return read_csr_matrix(obj)
        else
            error("Unknown group format at key: $key")
        end
    else
        return Float32.(read(obj))
    end
end

"""
    decode_categorical(f_obs, key) -> Vector{String}

Decode a pandas Categorical column stored in h5ad obs:
codes are Int8/Int16 indices into __categories/<key>.
"""
function decode_categorical(f_obs, key::String)
    codes = Int.(read(f_obs[key])) .+ 1   # 0-based → 1-based
    cats  = String.(read(f_obs["__categories"][key]))
    return cats[codes]
end

"""
    load_h5ad(path) -> AnnData

Read an AnnData .h5ad file into our Julia AnnData struct.
Handles: dense and CSR sparse matrices, categorical obs fields.
"""
function load_h5ad(path::String)
    isfile(path) || error("File not found: $path")
    h5open(path, "r") do f
        # ── Main matrix X ────────────────────────────────────────
        X = read_h5ad_matrix(f, "X")
        # h5ad stores rows=cells, cols=genes — no transpose needed for CSR

        # ── Layers ───────────────────────────────────────────────
        layers = Dict{String, Matrix{Float32}}()
        if haskey(f, "layers")
            for key in keys(f["layers"])
                try
                    layers[key] = read_h5ad_matrix(f["layers"], key)
                catch e
                    @warn "Could not read layer '$key': $e"
                end
            end
        end
        !haskey(layers, "spliced") && (layers["spliced"] = X)

        # ── Cell metadata (obs) ───────────────────────────────────
        obs = Dict{String, Vector}()
        if haskey(f, "obs")
            f_obs = f["obs"]
            cats_keys = haskey(f_obs, "__categories") ? keys(f_obs["__categories"]) : String[]
            for key in keys(f_obs)
                key == "__categories" && continue
                try
                    if key in cats_keys
                        obs[key] = decode_categorical(f_obs, key)
                    else
                        obs[key] = read(f_obs[key])
                    end
                catch end
            end
        end

        # ── Gene metadata (var) ───────────────────────────────────
        var = Dict{String, Vector}()
        if haskey(f, "var")
            for key in keys(f["var"])
                key == "__categories" && continue
                try; var[key] = read(f["var"][key]); catch end
            end
        end

        # ── Embeddings (obsm) ─────────────────────────────────────
        # HDF5.jl reads Python row-major arrays transposed (C→Fortran order).
        # Python stores obsm as (n_cells × n_components); Julia reads (n_components × n_cells).
        # We transpose back so obsm[key] is always (n_cells × n_components).
        obsm = Dict{String, Matrix{Float32}}()
        if haskey(f, "obsm")
            for key in keys(f["obsm"])
                try
                    M = Float32.(read(f["obsm"][key]))
                    if ndims(M) == 2
                        # After HDF5.jl read: shape is (n_components × n_cells) → transpose
                        obsm[key] = size(M, 1) < size(M, 2) ? Matrix(M') : M
                    else
                        obsm[key] = reshape(M, :, 1)
                    end
                catch end
            end
        end

        return AnnData(X; layers=layers, obs=obs, var=var, obsm=obsm)
    end
end
