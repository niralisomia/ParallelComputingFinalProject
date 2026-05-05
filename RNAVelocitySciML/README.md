# RNAVelocitySciML

RNA velocity estimation and neural ODE tooling in Julia ([SciML](https://sciml.ai/) / Lux).

**18.337/6.7320 final project —** Nirali Somia, Emilie de Vet, Antonio Rios

## What this is

The package implements RNA velocity and related analysis in Julia: steady-state (La Manno *et al.*, 2018) and dynamical (Bergen *et al.*, 2020) models, kNN pooling and gene-wise fitting, velocity graphs, PCA projection, and a Lux neural ODE for a learned vector field in latent space. It can load `.h5ad`, run the velocity pipeline, and plot with CairoMakie.

**RNA velocity in brief:** unspliced pre-mRNA ($u$) and spliced mRNA ($s$) are produced sequentially. The relationship between $u$ and $s$ per cell and per gene encodes whether that gene is being up- or down-regulated, yielding a velocity vector in expression space that approximates near-future state.

For the full narrative, equations, and validation workflows (synthetic data, bone marrow, neural ODE experiments), see the project notebook `Synthetic_and_Bone_Marrow_data.ipynb` at the repository root.

## Install from GitHub

In Julia, add the package from this repo using the **`subdir`** keyword (the Julia package lives in the `RNAVelocitySciML` folder, not at the repo root):

```julia
using Pkg
Pkg.activate()   # or Pkg.activate("path/to/your/environment")
Pkg.add(url="https://github.com/niralisomia/ParallelComputingFinalProject.git", subdir="RNAVelocitySciML")
```

Then:

```julia
using RNAVelocitySciML
```

`Pkg` will clone the repository, resolve dependencies from `RNAVelocitySciML/Project.toml`, and precompile. To pin a branch or revision, add e.g. `rev="main"` to `Pkg.add` (see the [Pkg docs](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-a-package-in-a-subdirectory-of-a-repository)).

### Optional: develop a local clone

If you have the repository checked out locally:

```julia
using Pkg
Pkg.develop(path="/absolute/path/to/ParallelComputingFinalProject/RNAVelocitySciML")
```

## Documentation in code

Exported functions and types are documented with docstrings in `src/*.jl`. In the Julia REPL, use `?function_name` (e.g. `?run_rna_velocity!`) to read them.

## Requirements

Julia 1.9+ and the dependencies listed in `Project.toml` / `Manifest.toml` (SciML stack, Lux, HDF5, MultivariateStats, CairoMakie, etc.).
