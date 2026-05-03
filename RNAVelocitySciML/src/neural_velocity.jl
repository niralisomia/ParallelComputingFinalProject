# ─────────────────────────────────────────────────────────────────
# Neural ODE model (Lux) + training + trajectory simulation
# ─────────────────────────────────────────────────────────────────

"""
    build_velocity_network(d; hidden=128, depth=3) -> model

Build a Lux.jl neural network f_θ: R^d → R^d that parameterizes the
continuous velocity field in PCA space.

Architecture: d → hidden → hidden → ... → d, with tanh activations.
tanh is used (not ReLU) to produce smooth, bounded vector fields.
"""
function build_velocity_network(d::Int; hidden::Int=128, depth::Int=3)
    layers = Any[Dense(d, hidden, tanh)]
    for _ in 2:depth
        push!(layers, Dense(hidden, hidden, tanh))
    end
    push!(layers, Dense(hidden, d))  # linear output
    return Chain(layers...)
end

"""
    VelocityField

Callable struct wrapping the Lux network and its parameters for use as
an ODE right-hand side: dz/dt = f_θ(z).
"""
struct VelocityField{M, PS, ST}
    model::M
    ps::PS
    st::ST
end

function (vf::VelocityField)(z, p, t)
    # z: state vector (d,), p: ComponentArray of params, t: time
    y, _ = vf.model(reshape(z, :, 1), p, vf.st)
    return vec(y)
end

"""
    train_neural_ode(Z_pca, V_pca; d_pca=30, n_epochs=200, lr=1e-3,
                     batch_size=256, hidden=128, depth=3, rng=Random.default_rng())

Train the neural velocity field f_θ: R^d → R^d to match RNA velocity
vectors projected into PCA space.

Loss: MSE between f_θ(z_i) and v̂_i for each cell i.

Returns the trained (model, ps, st) and loss history.
"""
function train_neural_ode(Z_pca::Matrix{Float32}, V_pca::Matrix{Float32};
                           n_epochs::Int=200,
                           lr::Float64=1e-3,
                           batch_size::Int=256,
                           hidden::Int=128,
                           depth::Int=3,
                           rng::AbstractRNG=Random.default_rng())
    N, d = size(Z_pca)
    @assert size(V_pca) == (N, d) "Z_pca and V_pca must have same shape"

    # Build model
    model = build_velocity_network(d; hidden=hidden, depth=depth)
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps)

    # Optimizer
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)

    loss_history = Float32[]

    # Z: (d × N) for Lux (features × batch)
    Z_T = Z_pca'  # d × N
    V_T = V_pca'  # d × N

    # Number of gradient steps per epoch = ceil(N / batch_size)
    n_batches = max(1, ceil(Int, N / batch_size))

    println("Training neural velocity field: d=$d, N=$N, epochs=$n_epochs, batches/epoch=$n_batches")
    @showprogress for epoch in 1:n_epochs
        # Shuffle once per epoch, then iterate over all cells in order
        perm = randperm(rng, N)
        epoch_loss = 0f0

        for b in 1:n_batches
            lo = (b - 1) * batch_size + 1
            hi = min(b * batch_size, N)
            idx = perm[lo:hi]
            Z_batch = Z_T[:, idx]
            V_batch = V_T[:, idx]

            loss, grads = Zygote.withgradient(ps) do p
                V_pred, _ = model(Z_batch, p, st)
                mean((V_pred .- V_batch).^2)
            end

            opt_state, ps = Optimisers.update!(opt_state, ps, grads[1])
            epoch_loss += loss
        end

        push!(loss_history, epoch_loss / n_batches)
    end

    println("Training complete. Final loss: $(loss_history[end])")
    return model, ps, st, loss_history
end

"""
    simulate_trajectories(z0_batch, model, ps, st;
                          tspan=(0f0, 10f0), n_steps=100,
                          solver=Tsit5())

Integrate the learned vector field f_θ forward in time from initial
conditions z0_batch (a K × d matrix of starting PCA coordinates).

Returns a (K × d × n_steps) array of trajectory coordinates.
Uses the adjoint method (InterpolatingAdjoint) for memory-efficient
gradient flow through the ODE solver.
"""
function simulate_trajectories(z0_batch::Matrix{Float32},
                                model, ps, st;
                                tspan::Tuple=(0f0, 10f0),
                                n_steps::Int=100,
                                solver=Tsit5())
    K, d = size(z0_batch)
    t_save = range(tspan[1], tspan[2], length=n_steps)
    trajectories = zeros(Float32, K, d, n_steps)

    for k in 1:K
        z0 = z0_batch[k, :]

        function ode_rhs!(dz, z, p, t)
            y, _ = model(reshape(z, d, 1), p, st)
            dz .= vec(y)
        end

        prob = ODEProblem(ode_rhs!, z0, tspan, ps)
        sol = solve(prob, solver;
                    saveat=collect(t_save),
                    sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

        for (ti, t) in enumerate(t_save)
            trajectories[k, :, ti] = sol(t)
        end
    end
    return trajectories
end
