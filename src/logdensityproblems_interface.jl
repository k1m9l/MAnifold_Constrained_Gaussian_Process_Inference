# src/logdensityproblems_interface.jl
module LogDensityProblemsInterface

# Import the core MAGI likelihood function
using ..Likelihoods: log_likelihood_and_gradient_banded
# Import types needed from other modules
using ..GaussianProcess: GPCov
using ..ODEModels
# Import the interface definition package
using LogDensityProblems
using LinearAlgebra

export MagiTarget

"""
    MagiTarget

A struct that bundles all necessary components for the MAGI model and implements
the `LogDensityProblems.jl` interface.

This acts as an adapter, allowing standard MCMC samplers (like `AdvancedHMC.jl`)
that expect the `LogDensityProblems` interface to work with the MAGI log-posterior
density function (`log_likelihood_and_gradient_banded`). It holds the data,
pre-calculated GP covariances, ODE functions, and fixed hyperparameters needed
to evaluate the log-posterior and its gradient.

# Fields
- `yobs::Matrix{T}`: Matrix of observations y(τ) (size: `n_times` × `n_dims`). Use `NaN` for missing.
- `gp_cov_all_dims::GPCovVec`: Vector of pre-calculated `GPCov` structs (one per dimension `d`), containing banded approximations of kernel matrices C⁻¹, K⁻¹, etc.
- `ode_f!::OdeFunc`: The ODE function f(x, θ, t), computing ẋ. Form: `f!(du, u, p, t)`.
- `ode_dfdx!::OdeJacX`: The ODE state Jacobian ∂f/∂x. Form: `dfdx!(J, u, p, t)`.
- `ode_dfdp::OdeJacP`: The ODE parameter Jacobian ∂f/∂θ. Form: `dfdp(u, p, t)` (returns matrix).
- `σ::Vector{T}`: Vector of observation noise standard deviations σ = (σ₁, ..., σ_D).
- `prior_temperature::Vector{T}`: Vector of tempering factors β = [β_deriv, β_level, β_obs].
- `n_times::Int`: Number of time points `n` in the discretization grid `I`.
- `n_dims::Int`: Number of system dimensions `D`.
- `n_params_ode::Int`: Number of ODE parameters `k` in θ.
"""
struct MagiTarget{T <: Real, GPCovVec <: AbstractVector{GPCov}, OdeFunc, OdeJacX, OdeJacP}
    yobs::Matrix{T}             # Observations y(τ)
    gp_cov_all_dims::GPCovVec   # Precomputed GP covariances (C⁻¹, K⁻¹, mϕ) per dim
    ode_f!::OdeFunc             # ODE function f(x, θ, t)
    ode_dfdx!::OdeJacX          # ODE Jacobian ∂f/∂x
    ode_dfdp::OdeJacP           # ODE Jacobian ∂f/∂θ
    σ::Vector{T}                # Noise standard deviations σ_d
    prior_temperature::Vector{T}# Tempering factors β
    n_times::Int                # Number of discretization points n
    n_dims::Int                 # Number of dimensions D
    n_params_ode::Int           # Number of ODE parameters k
end

"""
    LogDensityProblems.dimension(target::MagiTarget)

Returns the total dimension of the parameter vector sampled by the MCMC algorithm.

This is the combined dimension of the vectorized latent states `vec(x(I))` and the
ODE parameters `θ`. Dimension = (n_times × n_dims) + n_params_ode.
"""
function LogDensityProblems.dimension(target::MagiTarget)
    # Total dimension = (dimension of xlatent) + (dimension of θ)
    return target.n_times * target.n_dims + target.n_params_ode
end

"""
    LogDensityProblems.capabilities(::Type{<:MagiTarget})

Specifies that the `MagiTarget` struct can provide both the log-density value
and its gradient (LogDensityOrder{1}).

This is necessary for gradient-based MCMC samplers like HMC and NUTS, which require
the gradient ∇L to simulate Hamiltonian dynamics.
"""
function LogDensityProblems.capabilities(::Type{<:MagiTarget})
    # Signals that both logdensity and logdensity_and_gradient are available
    return LogDensityProblems.LogDensityOrder{1}()
end

"""
    _unpack_params(target::MagiTarget, params::AbstractVector{T}) where {T}

Helper function to unpack the flat parameter vector used by the sampler
into the structured `xlatent` matrix and `θ` vector needed by the likelihood function.

The MCMC sampler works with a single, flat vector `params` containing all variables
(latent states and ODE parameters). This function reverses the concatenation
`[vec(xlatent); θ]` performed when setting up the initial parameters for the sampler,
making the variables usable by the `log_likelihood_and_gradient_banded` function.

# Arguments
- `target`: The `MagiTarget` instance (used to get dimensions `n_times`, `n_dims`).
- `params`: The flat parameter vector `[vec(xlatent); θ]` provided by the sampler at a given MCMC step.

# Returns
- `Tuple{Matrix{T}, AbstractVector{T}}`: A tuple containing:
    - `xlatent`: The latent states x(I) reshaped into a matrix (n_times × n_dims).
    - `θ`: The ODE parameter vector θ.
"""
function _unpack_params(target::MagiTarget, params::AbstractVector{T}) where {T}
    # Calculate the number of elements corresponding to xlatent
    n_xlatent_elements = target.n_times * target.n_dims
    # Extract the portion corresponding to vec(xlatent) using a view for efficiency
    xlatent_flat = @view params[1:n_xlatent_elements]
    # Extract the portion corresponding to θ using a view
    θ = @view params[(n_xlatent_elements + 1):end]
    # Reshape the flat xlatent vector back into a matrix (n_times × n_dims)
    # Note: reshape shares memory with the underlying view
    xlatent = reshape(xlatent_flat, target.n_times, target.n_dims)
    return xlatent, θ
end

"""
    LogDensityProblems.logdensity(target::MagiTarget, params::AbstractVector{T}) where {T}

Calculates the MAGI log-posterior density L(xlatent, θ | y, ...) for the
given combined parameter vector `params`.

This function fulfills the `logdensity` requirement of the `LogDensityProblems`
interface. It unpacks the `params` vector into `xlatent` and `θ`, then calls the
core `log_likelihood_and_gradient_banded` function (from the `Likelihoods` module),
and returns only the log-density value L.

# Arguments
- `target`: The `MagiTarget` instance containing data, ODE functions, etc.
- `params`: The flat parameter vector `[vec(xlatent); θ]` provided by the sampler.

# Returns
- `Float64`: The log-posterior density value L.
"""
function LogDensityProblems.logdensity(target::MagiTarget, params::AbstractVector{T}) where {T}
    # Convert flat parameter vector back to structured xlatent matrix and θ vector
    xlatent, θ = _unpack_params(target, params)

    # Call the main likelihood function, ignoring the returned gradient
    ll, _ = log_likelihood_and_gradient_banded(
        xlatent,
        θ,
        target.σ,
        target.yobs,
        target.gp_cov_all_dims,
        target.ode_f!,
        target.ode_dfdx!,
        target.ode_dfdp;
        prior_temperature = target.prior_temperature
    )
    return ll # Return only the log-likelihood value
end

"""
    LogDensityProblems.logdensity_and_gradient(target::MagiTarget, params::AbstractVector{T}) where {T}

Calculates both the MAGI log-posterior density L(xlatent, θ | y, ...) and its
gradient ∇L with respect to the combined parameter vector `params`.

This function fulfills the `logdensity_and_gradient` requirement of the
`LogDensityProblems` interface (LogDensityOrder{1}). It unpacks the `params`
vector into `xlatent` and `θ`, then calls the core
`log_likelihood_and_gradient_banded` function (from the `Likelihoods` module),
and returns both the log-density value L and the gradient vector ∇L. The gradient
vector is structured as `[vec(∂L/∂xlatent); ∂L/∂θ]`.

# Arguments
- `target`: The `MagiTarget` instance containing data, ODE functions, etc.
- `params`: The flat parameter vector `[vec(xlatent); θ]` provided by the sampler.

# Returns
- `Tuple{Float64, Vector{T}}`: A tuple containing:
    1.  `log_likelihood`: The log-posterior density value L.
    2.  `gradient`: The gradient vector ∇L.
"""
function LogDensityProblems.logdensity_and_gradient(target::MagiTarget, params::AbstractVector{T}) where {T}
    # Convert flat parameter vector back to structured xlatent matrix and θ vector
    xlatent, θ = _unpack_params(target, params)

    # Call the main likelihood function to get both value and gradient
    ll, grad = log_likelihood_and_gradient_banded(
        xlatent,
        θ,
        target.σ,
        target.yobs,
        target.gp_cov_all_dims,
        target.ode_f!,
        target.ode_dfdx!,
        target.ode_dfdp;
        prior_temperature = target.prior_temperature
    )
    # Return both log-likelihood value and the gradient vector
    return ll, grad
end

end # module LogDensityProblemsInterface
