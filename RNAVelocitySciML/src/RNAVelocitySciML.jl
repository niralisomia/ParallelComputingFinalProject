"""
RNA velocity estimation and neural ODE tooling (SciML / Lux), extracted from the
course project notebook for reuse as a normal Julia package.
"""
module RNAVelocitySciML

using Random
using LinearAlgebra
using SparseArrays
using Statistics
using Logging

using Distributions
using DifferentialEquations
using OrdinaryDiffEq
using SciMLSensitivity
using Lux
using Optimisers
using Zygote
using ComponentArrays
using MultivariateStats
using NearestNeighbors
using HDF5
using CairoMakie
import CairoMakie: Axis
import HDF5
using ProgressMeter

export AnnData, n_cells, n_genes
export normalize_total!, log1p_transform!, filter_genes
export knn_pool, spearman_cor, estimate_gammas, compute_velocity_steady_state
export RNAVelocityParams, rna_ode!, solve_gene_trajectory
export steady_state_u, steady_state_s, analytical_u, analytical_s, predict_us
export assign_latent_times, fit_gene_params_em, compute_velocity_dynamical
export run_rna_velocity!
export build_velocity_graph, project_velocity_embedding
export compute_velocity_consistency, plot_velocity_consistency
export velocity_length, velocity_confidence
export compute_pca, project_velocity_pca
export build_velocity_network, VelocityField, train_neural_ode, simulate_trajectories
export read_csr_matrix, read_h5ad_matrix, decode_categorical, load_h5ad
export benchmark_velocity, cosine_similarity_score
export make_color_dict, plot_velocity_embedding, plot_phase_portrait
export plot_trajectories_pca, plot_loss_history
export generate_synthetic_dataset

include("ann_data.jl")
include("gene_estimation.jl")
include("dynamical_ode.jl")
include("pipeline.jl")
include("velocity_graph.jl")
include("pca.jl")
include("neural_velocity.jl")
include("io.jl")
include("benchmark.jl")
include("plots.jl")
include("synthetic.jl")

end # module
