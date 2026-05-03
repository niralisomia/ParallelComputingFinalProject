# ─────────────────────────────────────────────────────────────────
# Dynamical model (Bergen et al. 2020 — scVelo-style) + EM fitting
# ─────────────────────────────────────────────────────────────────

"""
    RNAVelocityParams

Holds gene-wise rate parameters estimated under the full dynamical model.
"""
struct RNAVelocityParams
    α::Vector{Float32}   # transcription rates
    β::Vector{Float32}   # splicing rates
    γ::Vector{Float32}   # degradation rates
    t_::Vector{Float32}  # switching times (induction→repression)
end

"""
    rna_ode!(du, u, p, t)

Single-gene RNA velocity ODE system for one gene:
    du/dt = α(t) - β*u
    ds/dt = β*u  - γ*s
where α(t) = α during induction phase (t < t_switch), 0 during repression.
"""
function rna_ode!(du, u_state, p, t)
    α, β, γ, t_switch = p
    α_t = t < t_switch ? α : 0f0
    du[1] = α_t - β * u_state[1]   # d(unspliced)/dt
    du[2] = β * u_state[1] - γ * u_state[2]  # d(spliced)/dt
end

"""
    solve_gene_trajectory(α, β, γ, t_switch; tspan=(0f0, 20f0), n_points=200)

Integrate the ODE for a single gene and return the (u, s) trajectory.
"""
function solve_gene_trajectory(α::Float32, β::Float32, γ::Float32, t_switch::Float32;
                                tspan=(0f0, 20f0), n_points::Int=200)
    u0 = [0f0, 0f0]  # start from zero
    p  = (α, β, γ, t_switch)
    prob = ODEProblem(rna_ode!, u0, tspan, p)
    sol  = solve(prob, Tsit5(); saveat=range(tspan[1], tspan[2], length=n_points))
    return sol
end

"""
    steady_state_u(α, β)  -> u*
    steady_state_s(α, β, γ) -> s*
"""
steady_state_u(α, β) = α / β
steady_state_s(α, β, γ) = α / γ

# Numerically stable helpers
softplus(x) = log1p(exp(-abs(x))) + max(x, 0f0)
sigmoid(x) = 1f0 / (1f0 + exp(-x))

"""
    analytical_u(t, α, β, u0)
    analytical_s(t, α, β, γ, u0, s0)

Closed-form solution of:
  du/dt = α - βu
  ds/dt = βu - γs
"""
analytical_u(t, α, β, u0) = u0 * exp(-β*t) + (α/β) * (1 - exp(-β*t))

function analytical_s(t, α, β, γ, u0, s0)
    if abs(β - γ) < 1f-6
        return s0 * exp(-γ*t) + (α/γ) * (1 - exp(-γ*t)) +
               β * (u0 - α/β) * t * exp(-γ*t)
    else
        c1 = β * (u0 - α/β) / (γ - β)
        return (s0 - α/γ - c1) * exp(-γ*t) + c1 * exp(-β*t) + α/γ
    end
end

"""
    predict_us(t, α, β, γ, t_switch)

Piecewise trajectory with induction (α>0) then repression (α=0).
"""
function predict_us(t::Float32, α::Float32, β::Float32, γ::Float32, t_switch::Float32)
    if t <= t_switch
        u = analytical_u(t, α, β, 0f0)
        s = analytical_s(t, α, β, γ, 0f0, 0f0)
    else
        u_sw = analytical_u(t_switch, α, β, 0f0)
        s_sw = analytical_s(t_switch, α, β, γ, 0f0, 0f0)
        τ = t - t_switch
        u = analytical_u(τ, 0f0, β, u_sw)
        s = analytical_s(τ, 0f0, β, γ, u_sw, s_sw)
    end
    return u, s
end

"""
    assign_latent_times(u_obs, s_obs, α, β, γ, t_switch; n_t=60, t_end=20)

Fast E-step: assign each cell to nearest point on a precomputed trajectory grid.
"""
function assign_latent_times(u_obs::Vector{Float32}, s_obs::Vector{Float32},
                             α::Float32, β::Float32, γ::Float32, t_switch::Float32;
                             n_t::Int=60, t_end::Float32=20f0)
    N = length(u_obs)
    t_grid = collect(range(0f0, t_end, length=2*n_t))

    u_grid = similar(t_grid)
    s_grid = similar(t_grid)
    for k in eachindex(t_grid)
        u_grid[k], s_grid[k] = predict_us(t_grid[k], α, β, γ, t_switch)
    end

    t_latent = zeros(Float32, N)
    phases = fill(:induction, N)

    for i in 1:N
        ui, si = u_obs[i], s_obs[i]
        best_k = 1
        best_d = Inf32
        for k in eachindex(t_grid)
            d = (ui - u_grid[k])^2 + (si - s_grid[k])^2
            if d < best_d
                best_d = d
                best_k = k
            end
        end
        t_latent[i] = t_grid[best_k]
        phases[i] = t_latent[i] <= t_switch ? :induction : :repression
    end
    return t_latent, phases
end

"""
    fit_gene_params_em(u_obs, s_obs; ...)

Stabilized EM-style fit for (α, β, γ, t_switch):
- Optional supervised latent time from pseudotime (`t_obs`)
- Bounded parameterization via softplus/sigmoid to avoid collapse
- Lightweight gradient M-step on reconstruction error in (u,s)
"""
function fit_gene_params_em(u_obs::Vector{Float32}, s_obs::Vector{Float32};
                            n_iter::Int=6,
                            n_t::Int=60,
                            gd_steps::Int=20,
                            lr::Float32=3f-2,
                            t_end::Float32=20f0,
                            t_obs::Union{Nothing, Vector{Float32}}=nothing)
    N = length(u_obs)

    # Initialize from moments
    α0 = clamp(mean(u_obs), 1f-3, 50f0)
    β0 = 1f0
    γ0 = clamp(sum(u_obs .* s_obs) / max(1f-6, sum(u_obs .^ 2)), 1f-3, 10f0)
    t0 = isnothing(t_obs) ? (0.35f0 * t_end) : clamp(median(t_obs), 1f0, t_end - 1f0)

    # Raw unconstrained params
    raw = Float32[log(exp(α0)-1f0), log(exp(β0)-1f0), log(exp(γ0)-1f0), 0f0]

    unpack(rawv) = begin
        α = softplus(rawv[1]) + 1f-4
        β = softplus(rawv[2]) + 1f-4
        γ = softplus(rawv[3]) + 1f-4
        # keep away from hard bounds to prevent degeneracy
        t_switch = 1f0 + (t_end - 2f0) * sigmoid(rawv[4])
        α, β, γ, t_switch
    end

    # Use external pseudotime if provided; otherwise do EM latent-time assignment
    t_lat = isnothing(t_obs) ? fill(t0, N) : clamp.(t_obs, 0f0, t_end)

    function reconstruction_loss(rawv, t_vec)
        α, β, γ, t_switch = unpack(rawv)
        acc = 0f0
        @inbounds for i in 1:N
            up, sp = predict_us(t_vec[i], α, β, γ, t_switch)
            du = up - u_obs[i]
            ds = sp - s_obs[i]
            acc += du*du + ds*ds
        end
        acc / N
    end

    for _ in 1:n_iter
        if isnothing(t_obs)
            α, β, γ, t_switch = unpack(raw)
            t_lat, _ = assign_latent_times(u_obs, s_obs, α, β, γ, t_switch; n_t=n_t, t_end=t_end)
        end

        # M-step: optimize params with fixed latent times
        for _ in 1:gd_steps
            grads = Zygote.gradient(rv -> reconstruction_loss(rv, t_lat), raw)[1]
            grads = clamp.(grads, -10f0, 10f0)
            raw .-= lr .* grads
        end
    end

    α, β, γ, t_switch = unpack(raw)
    return (α=Float32(α), β=Float32(β), γ=Float32(γ), t_switch=Float32(t_switch))
end

"""
    compute_velocity_dynamical(u_obs, s_obs, params)
"""
function compute_velocity_dynamical(u_obs::Vector{Float32}, s_obs::Vector{Float32},
                                    params::NamedTuple)
    return params.β .* u_obs .- params.γ .* s_obs
end
