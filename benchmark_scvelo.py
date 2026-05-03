#!/usr/bin/env python3
"""
benchmark_scvelo.py

Benchmarks scVelo (Bergen et al. 2020) steady-state and dynamical models
on the same bone marrow dataset used in the Julia notebook (Setty et al. 2019).

Saves timing + quality metrics to models/scvelo_benchmark.jl so they can be
loaded directly with Julia's include() for the comparison plots.

Usage:
    pip install scvelo anndata h5py scipy
    python benchmark_scvelo.py [--dyn]   # --dyn enables the slow dynamical model
"""

import argparse, time, os, sys, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import anndata as ad
    import scanpy as sc
    import scvelo as scv
    from scipy.sparse import issparse
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nRun: pip install scvelo anndata h5py scipy")


def compute_consistency(adata_fit, cell_type_key):
    """
    Fraction of transitions that stay within the same cell type,
    computed from scVelo's velocity graph (same definition as Julia notebook).
    """
    vg = adata_fit.uns.get("velocity_graph", None)
    if vg is None:
        return {}
    T = vg.toarray() if issparse(vg) else np.array(vg)
    labels = adata_fit.obs[cell_type_key].values
    unique = sorted(set(labels))
    result = {}
    for ct in unique:
        mask = labels == ct
        idx  = np.where(mask)[0]
        if len(idx) == 0:
            continue
        T_sub = T[idx]
        total  = T_sub.sum()
        if total < 1e-12:
            continue
        within = T_sub[:, mask].sum()
        result[str(ct)] = float(within / total)
    return result


def dict_to_julia(d):
    if not d:
        return 'Dict{String,Float64}()'
    pairs = ", ".join(f'"{k}" => {v:.6f}' for k, v in d.items())
    return f"Dict({pairs})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dyn", action="store_true", help="Also run the dynamical model (~15-60 min)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Threads for scVelo recover_dynamics")
    args = parser.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_PATH = Path.home() / "18337_ParallelComputing/RNAVelocity/data/bone_marrow.h5ad"
    OUT_PATH  = Path("models/scvelo_benchmark.jl")

    if not DATA_PATH.exists():
        sys.exit(f"Data file not found: {DATA_PATH}\nRun the Julia notebook first to download it.")

    os.makedirs("models", exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    adata_raw = ad.read_h5ad(DATA_PATH)
    print(f"  {adata_raw.n_obs} cells x {adata_raw.n_vars} genes")

    cell_type_key = "cell_type" if "cell_type" in adata_raw.obs.columns else \
                    "clusters"  if "clusters"  in adata_raw.obs.columns else \
                    adata_raw.obs.columns[0]
    print(f"  Cell type key: '{cell_type_key}'")
    print(f"  Cell types: {sorted(adata_raw.obs[cell_type_key].unique())}")

    # ── Preprocessing ──────────────────────────────────────────────────────────
    print("\nPreprocessing...")
    adata_prep = adata_raw.copy()
    t0 = time.perf_counter()
    scv.pp.filter_and_normalize(adata_prep, min_shared_counts=20)
    sc.pp.pca(adata_prep, n_comps=30)
    sc.pp.neighbors(adata_prep, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata_prep)
    prep_time = time.perf_counter() - t0
    print(f"  Preprocessing: {prep_time:.1f}s  ({adata_prep.n_obs} cells x {adata_prep.n_vars} genes after filter)")

    # ── Steady-state model ─────────────────────────────────────────────────────
    print("\nRunning scVelo steady-state model...")
    adata_ss = adata_prep.copy()
    t0 = time.perf_counter()
    scv.tl.velocity(adata_ss, mode="stochastic")
    scv.tl.velocity_graph(adata_ss, n_jobs=1)
    ss_time = time.perf_counter() - t0
    print(f"  Steady-state: {ss_time:.1f}s")

    scv.tl.velocity_confidence(adata_ss)
    ss_conf        = float(np.nanmedian(adata_ss.obs["velocity_confidence"].values))
    ss_consistency = compute_consistency(adata_ss, cell_type_key)
    print(f"  Median confidence: {ss_conf:.3f}")
    print(f"  Consistency: { {k: round(v, 3) for k, v in ss_consistency.items()} }")

    # Save velocity matrix + γ + gene names for direct Julia comparison
    try:
        import h5py
        V_ss = adata_ss.layers["velocity"]
        if issparse(V_ss): V_ss = V_ss.toarray()
        with h5py.File("models/scvelo_ss_comparison.h5", "w") as fh:
            fh.create_dataset("velocity",   data=V_ss.astype(np.float32))
            fh.create_dataset("gamma",      data=adata_ss.var["velocity_gamma"].fillna(0).values.astype(np.float32))
            fh.create_dataset("gene_names", data=np.array(adata_ss.var_names.tolist(), dtype="S"))
            fh.create_dataset("cell_types", data=np.array(adata_ss.obs[cell_type_key].tolist(), dtype="S"))
        print("  Saved models/scvelo_ss_comparison.h5 (velocity, gamma, gene_names)")
    except Exception as e:
        print(f"  Warning: could not save H5 comparison file: {e}")

    # ── Dynamical model (optional) ─────────────────────────────────────────────
    dyn_time        = None
    dyn_conf        = None
    dyn_consistency = {}

    if args.dyn:
        print(f"\nRunning scVelo dynamical model (n_jobs={args.n_jobs}) — this may take 15–60 min...")
        adata_dyn = adata_prep.copy()
        t0 = time.perf_counter()
        scv.tl.recover_dynamics(adata_dyn, n_jobs=args.n_jobs)
        scv.tl.velocity(adata_dyn, mode="dynamical")
        scv.tl.velocity_graph(adata_dyn, n_jobs=1)
        dyn_time = time.perf_counter() - t0
        print(f"  Dynamical: {dyn_time:.1f}s  ({dyn_time/60:.1f} min)")

        scv.tl.velocity_confidence(adata_dyn)
        dyn_conf        = float(np.nanmedian(adata_dyn.obs["velocity_confidence"].values))
        dyn_consistency = compute_consistency(adata_dyn, cell_type_key)
        print(f"  Median confidence: {dyn_conf:.3f}")
        print(f"  Consistency: { {k: round(v, 3) for k, v in dyn_consistency.items()} }")
    else:
        print("\nSkipping dynamical model (pass --dyn to enable).")

    # ── Write Julia-parseable results ──────────────────────────────────────────
    with open(OUT_PATH, "w") as f:
        f.write("# Auto-generated by benchmark_scvelo.py — do not edit manually\n")
        f.write(f"# scVelo version: {scv.__version__}\n\n")
        f.write(f"const scvelo_prep_time_s      = {prep_time:.3f}\n")
        f.write(f"const scvelo_ss_time_s        = {ss_time:.3f}\n")
        f.write(f"const scvelo_dyn_time_s       = {dyn_time if dyn_time is not None else 'NaN'}\n")
        f.write(f"const scvelo_ss_conf          = {ss_conf:.6f}\n")
        f.write(f"const scvelo_dyn_conf         = {dyn_conf if dyn_conf is not None else 'NaN'}\n")
        f.write(f"const scvelo_n_cells          = {adata_prep.n_obs}\n")
        f.write(f"const scvelo_n_genes_filtered = {adata_prep.n_vars}\n")
        f.write(f"const scvelo_ss_consistency   = {dict_to_julia(ss_consistency)}\n")
        f.write(f"const scvelo_dyn_consistency  = {dict_to_julia(dyn_consistency)}\n")

    print(f"\nResults written to {OUT_PATH}")
    print("\n" + "="*55)
    print("scVelo benchmark summary")
    print("="*55)
    print(f"  Dataset:       {adata_prep.n_obs} cells x {adata_prep.n_vars} genes")
    print(f"  Preprocessing: {prep_time:.1f}s")
    print(f"  Steady-state:  {ss_time:.1f}s")
    print(f"  Dynamical:     {f'{dyn_time:.1f}s' if dyn_time else 'not run (use --dyn)'}")
    print(f"  SS confidence: {ss_conf:.3f}")
    print(f"  Dyn confidence:{f' {dyn_conf:.3f}' if dyn_conf else ' N/A'}")
    print("="*55)
    print("Next: run the Julia benchmark cells in the notebook to generate comparison plots.")


if __name__ == "__main__":
    main()
