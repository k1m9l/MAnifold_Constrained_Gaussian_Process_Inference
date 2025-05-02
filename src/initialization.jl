# -*- coding: utf-8 -*-
# src/initialization.jl

"""
    Initialization

Provides functions for initializing hyperparameters required by the MAGI algorithm,
specifically focusing on estimating Gaussian Process (GP) kernel hyperparameters (ϕ)
and observation noise standard deviations (σ) by optimizing the marginal likelihood
of the observed data for each dimension independently.
"""
module Initialization

# Dependencies needed for this module
using Optim                  # For optimization algorithms (Nelder-Mead)
using LinearAlgebra          # For matrix operations (Diagonal, dot, cholesky, logdet)
using KernelFunctions        # For GP kernel definitions and kernelmatrix calculation
using PositiveFactorizations # For robust Cholesky decomposition (cholesky(Positive, ...))
using Statistics             # For var, median used in initial guesses
using Logging                # For @warn, @error messages

# Access modules from the parent MagiJl module
using ..Kernels              # To use kernel creation functions like create_matern52_kernel

# Export the main optimization function
export optimize_gp_hyperparameters

# --------------------------------------------------------------------------
# Objective Function: Negative Log Marginal Likelihood
# --------------------------------------------------------------------------
"""
    negative_log_marginal_likelihood(
        log_params::Vector{Float64},
        y_obs_dim::Vector{Float64},
        t_obs::Vector{Float64},
        kernel_type::String,
        jitter::Float64 = 1e-6
    )

Calculates the negative log marginal likelihood (NLML) for a single-dimensional
Gaussian Process (GP) model, ignoring any ODE constraints. This is used for
initializing GP hyperparameters (ϕ) and observation noise (σ).

Assumes the observations `y` for a single dimension follow the model:
  y ~ GP(0, Kϕ) + N(0, σ²I)
where Kϕ is the covariance matrix derived from the chosen kernel with
hyperparameters ϕ = [variance (σϕ²); lengthscale (ℓ)], and σ² is the observation
noise variance.

The marginal likelihood integrates out the latent GP function values. Its
negative logarithm (up to constants) is proportional to:
  NLML ∝ log|Kϕ + σ²I| + yᵀ(Kϕ + σ²I)⁻¹y

This function computes this value, taking log-transformed parameters as input
for unconstrained optimization.

# Arguments
- `log_params::Vector{Float64}`: Vector containing log-transformed hyperparameters:
    `[log(variance), log(lengthscale), log(σ)]`. Using logs ensures positivity.
- `y_obs_dim::Vector{Float64}`: Vector of observations `y` for the current dimension. `NaN` values are handled.
- `t_obs::Vector{Float64}`: Vector of observation times `t`.
- `kernel_type::String`: Specifies the GP kernel ("matern52", "rbf", etc.).
- `jitter::Float64`: Small value added to the diagonal of the covariance matrix for numerical stability during inversion.

# Returns
- `Float64`: The calculated negative log marginal likelihood value. Returns `Inf` if parameters are invalid or numerical issues occur.

# Notes
- Handles missing values (`NaN`) in `y_obs_dim` by subsetting the data and covariance matrix.
- Uses `PositiveFactorizations.jl` for robust Cholesky decomposition.
"""
function negative_log_marginal_likelihood(log_params::Vector{Float64},
                                          y_obs_dim::Vector{Float64},
                                          t_obs::Vector{Float64},
                                          kernel_type::String,
                                          jitter::Float64 = 1e-6)

    # Transform log-parameters back to original positive scale
    variance = exp(log_params[1])    # σϕ² (Signal variance)
    lengthscale = exp(log_params[2]) # ℓ (Lengthscale)
    σ = exp(log_params[3])           # σ (Noise standard deviation)
    σ² = σ^2                         # σ² (Noise variance)

    # Basic parameter validity checks
    if !isfinite(variance) || !isfinite(lengthscale) || !isfinite(σ) || variance <= 0 || lengthscale <= 0 || σ <= 0
        return Inf # Invalid parameters lead to infinite penalty
    end

    # --- Handle NaNs ---
    # Find indices of valid (non-NaN) observations
    valid_indices = findall(!isnan, y_obs_dim)
    if isempty(valid_indices)
        # If no data for this dimension, likelihood is undefined. Return Inf.
        return Inf
    end
    # Subset data and time points to only include valid observations
    y_subset = y_obs_dim[valid_indices]
    t_subset = t_obs[valid_indices]
    n_valid = length(y_subset) # Number of valid observations N_d
    # -------------------

    # --- Create Kernel ---
    local kernel # Ensure kernel is in scope outside try block
    try
        # Use helper functions from the Kernels module to create the kernel object
        if kernel_type == "matern52"
            kernel = Kernels.create_matern52_kernel(variance, lengthscale)
        elseif kernel_type == "rbf"
            kernel = Kernels.create_rbf_kernel(variance, lengthscale)
        # Add other kernel types here if needed
        else
            @warn "Unsupported kernel type '$kernel_type' in optimization. Defaulting to matern52."
            kernel = Kernels.create_matern52_kernel(variance, lengthscale)
        end
    catch e
         @error "Error creating kernel:" exception=(e, catch_backtrace()) variance=variance lengthscale=lengthscale
         return Inf # Penalize kernel creation errors
    end
    # -------------------

    # --- Calculate Covariance Matrix Kϕ + σ²I ---
    try
        # Kϕ: Covariance matrix from the GP kernel for valid time points
        Kϕ_mat = kernelmatrix(kernel, t_subset) # Renamed to avoid clash with Kϕ notation meaning K parameterized by ϕ

        # K_full = Kϕ_mat + σ²I + jitter*I : Add noise variance and jitter to the diagonal
        # Ensure K_full is explicitly Float64 if kernelmatrix returns other types
        K_full_base = Kϕ_mat + Diagonal(fill(σ² + jitter, n_valid))
        # Ensure symmetry and Float64 type for numerical stability and compatibility
        K_full = Symmetric(Matrix{Float64}(K_full_base))

        # --- Cholesky Decomposition ---
        # Use PositiveFactorizations for robustness against near non-PSD matrices
        # chol_K = cholesky(K_full) -> L such that LLᵀ = K_full
        chol_K = cholesky(PositiveFactorizations.Positive, K_full)

        # --- Calculate Log Determinant & Quadratic Term ---
        # log|K_full| = log|LLᵀ| = 2 * log|L| = 2 * sum(log(diag(L)))
        log_det_K = logdet(chol_K)

        # Quadratic term: yᵀ * K_full⁻¹ * y
        # Solve K_full * α = y efficiently using Cholesky: L Lᵀ α = y
        # Forward substitution: L z = y
        # Backward substitution: Lᵀ α = z  => α = Lᵀ \\ (L \\ y) = K_full \\ y
        α = chol_K \ y_subset
        quad_term = dot(y_subset, α) # yᵀ * α = yᵀ * K_full⁻¹ * y

        # --- Negative Log Marginal Likelihood ---
        # Formula: 0.5 * (log|K_full| + yᵀ K_full⁻¹ y + N_d * log(2π))
        # (Ignoring constant term -0.5 * N_d * log(2π) doesn't affect optimization minimum)
        # We include it here for correctness of the value.
        neg_log_lik = 0.5 * (log_det_K + quad_term + n_valid * log(2.0 * π))

        # Check for non-finite results which can crash optimizer
        if !isfinite(neg_log_lik)
             @warn "NLML calculation resulted in non-finite value." log_params=log_params
             return Inf # Penalize non-finite results heavily
        end

        return neg_log_lik

    catch e
        # Handle potential errors during kernelmatrix or cholesky (e.g., PosDefException)
        # These often occur with extreme hyperparameter values.
        # Return Inf to indicate these parameters are invalid/unstable for the optimizer.
         if e isa PosDefException || contains(string(lowercase(string(e))), "cholesky") || contains(string(lowercase(string(e))), "positive definite")
             # This commonly happens with extreme lengthscales or variances
             # println("Cholesky failed for params: ", exp.(log_params)) # Debug print
             return Inf # Penalize non-positive definite matrices
         else
             # Log unexpected errors
             @error "Unexpected error in negative_log_marginal_likelihood calculation:" exception=(e, catch_backtrace()) params=exp.(log_params)
             return Inf # Penalize other errors
         end
    end
end

# --------------------------------------------------------------------------
# Optimization Function for GP Hyperparameters
# --------------------------------------------------------------------------
"""
    optimize_gp_hyperparameters(
        y_obs_dim::Vector{Float64},
        t_obs::Vector{Float64},
        kernel_type::String,
        initial_log_params::Vector{Float64};
        jitter::Float64 = 1e-6,
        optim_options = Optim.Options(iterations = 100, show_trace=false)
    )

Optimizes GP hyperparameters (variance σϕ², lengthscale ℓ) and observation noise (σ)
for a single dimension by minimizing the negative log marginal likelihood (NLML)
calculated by `negative_log_marginal_likelihood`.

Uses the Nelder-Mead algorithm (gradient-free) from `Optim.jl`.

# Arguments
- `y_obs_dim::Vector{Float64}`: Observations `y` for the current dimension.
- `t_obs::Vector{Float64}`: Observation times `t`.
- `kernel_type::String`: Specifies the GP kernel ("matern52", "rbf", etc.).
- `initial_log_params::Vector{Float64}`: Vector of initial guesses for the log-transformed
    hyperparameters `[log(variance), log(lengthscale), log(σ)]`.
- `jitter::Float64`: Jitter value passed to `negative_log_marginal_likelihood`.
- `optim_options::Optim.Options`: Options for the `Optim.optimize` function (e.g., iterations, tolerance).

# Returns
- `Vector{Float64}`: Vector containing the optimized hyperparameters in their
    original scale: `[optimized_variance, optimized_lengthscale, optimized_sigma]`.
    Returns the exponentiated initial guess if optimization yields invalid parameters.
"""
function optimize_gp_hyperparameters(y_obs_dim::Vector{Float64},
                                     t_obs::Vector{Float64},
                                     kernel_type::String,
                                     initial_log_params::Vector{Float64};
                                     jitter::Float64 = 1e-6,
                                     optim_options = Optim.Options(iterations = 100, show_trace=false)) # Example options

    # Define the objective function closure for Optim.jl
    # This captures the fixed arguments (y_obs_dim, t_obs, etc.)
    objective = log_p -> negative_log_marginal_likelihood(log_p, y_obs_dim, t_obs, kernel_type, jitter)

    # Perform optimization using Nelder-Mead (a gradient-free method suitable when gradients are unavailable or complex)
    # Optimizes over the log-transformed parameters `log_p`.
    # Box constraints could be added using Fminbox(NelderMead()) if needed,
    # but optimizing log-params often keeps them in valid (positive) range after exp().
    result = Optim.optimize(objective,
                            initial_log_params, # Starting point in log-space
                            NelderMead(),       # Optimization algorithm
                            optim_options)      # Optimizer settings

    # Check convergence status
    if !Optim.converged(result)
        @warn "GP hyperparameter optimization did not converge for a dimension. Using best found parameters." minimum=Optim.minimum(result) params=exp.(Optim.minimizer(result))
    end

    # Get the optimized log-parameters that minimize the NLML
    optimized_log_params = Optim.minimizer(result)
    # Transform back to the original parameter scale (variance, lengthscale, sigma)
    optimized_params = exp.(optimized_log_params) # [variance, lengthscale, σ]

    # Add a final check for potentially unreasonable values after optimization
    if any(x -> !isfinite(x) || x <= 0, optimized_params)
         @error "Optimization resulted in invalid parameters: $(optimized_params). Check initial guesses and data."
         # Fallback to initial guess (in original scale) if optimization fails badly
         # Returning initial guess might be safer than erroring out completely.
         @warn "Falling back to initial hyperparameter guess due to invalid optimization result."
         return exp.(initial_log_params)
    end

    # Return the optimized parameters [variance, lengthscale, σ]
    return optimized_params
end


end # module Initialization