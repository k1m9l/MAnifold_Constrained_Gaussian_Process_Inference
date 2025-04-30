# src/initialization.jl

module Initialization

# Dependencies needed for this module
using Optim
using LinearAlgebra
using KernelFunctions
using PositiveFactorizations # For robust Cholesky
using Statistics           # For var, median used in initial guesses within solve_magi
using Logging              # For @warn, @error

# Access modules from the parent MagiJl module
using ..Kernels            # To use kernel creation functions like create_matern52_kernel

# Export the main optimization function
export optimize_gp_hyperparameters

# --------------------------------------------------------------------------
# Objective Function: Negative Log Marginal Likelihood
# --------------------------------------------------------------------------
"""
    negative_log_marginal_likelihood(log_params, y_obs_dim, t_obs, kernel_type, jitter)

Calculates the negative log marginal likelihood for a single dimension GP model.
Assumes y_obs ~ GP(0, K_phi) + N(0, sigma^2*I).

Arguments:
- log_params: Vector containing [log(variance), log(lengthscale), log(sigma)].
- y_obs_dim: Vector of observations for the current dimension (NaNs are handled).
- t_obs: Vector of observation times.
- kernel_type: String specifying the kernel ("matern52", "rbf", etc.).
- jitter: Small value added to the diagonal for numerical stability.

Returns:
- Negative log marginal likelihood value (Float64), or Inf if parameters are invalid.
"""
function negative_log_marginal_likelihood(log_params::Vector{Float64},
                                          y_obs_dim::Vector{Float64},
                                          t_obs::Vector{Float64},
                                          kernel_type::String,
                                          jitter::Float64 = 1e-6)

    # Transform parameters back to original scale
    variance = exp(log_params[1])
    lengthscale = exp(log_params[2])
    sigma = exp(log_params[3]) # Standard deviation sigma
    sigma_sq = sigma^2         # Variance sigma^2

    # Basic parameter validity checks
    if !isfinite(variance) || !isfinite(lengthscale) || !isfinite(sigma) || variance <= 0 || lengthscale <= 0 || sigma <= 0
        return Inf # Invalid parameters
    end

    # --- Handle NaNs ---
    valid_indices = findall(!isnan, y_obs_dim)
    if isempty(valid_indices)
        # If no data, likelihood is undefined (or constant). Return Inf to avoid selection.
        return Inf
    end
    y_subset = y_obs_dim[valid_indices]
    t_subset = t_obs[valid_indices]
    n_valid = length(y_subset)
    # -------------------

    # --- Create Kernel ---
    local kernel # Ensure kernel is in scope
    try
        # Use functions from the Kernels module
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

    # --- Calculate Covariance Matrix K_phi + sigma^2*I ---
    try
        K_phi = kernelmatrix(kernel, t_subset)
        # Add observation noise variance and jitter for numerical stability
        # Ensure K_full is explicitly Float64 if kernelmatrix returns other types
        K_full_base = K_phi + Diagonal(fill(sigma_sq + jitter, n_valid))
        K_full = Symmetric(Matrix{Float64}(K_full_base)) # Ensure symmetry and Float64

        # --- Cholesky Decomposition ---
        # Use PositiveFactorizations for robustness against near non-PSD matrices
        chol_K = cholesky(PositiveFactorizations.Positive, K_full)

        # --- Calculate Log Determinant & Quadratic Term ---
        # logdet(K_full) = 2 * sum(log.(diag(chol_K.L)))
        log_det_K = logdet(chol_K)
        # Solve K_full * alpha = y_subset efficiently using Cholesky: alpha = K_full \ y_subset
        alpha = chol_K \ y_subset
        quad_term = dot(y_subset, alpha) # y' * inv(K_full) * y

        # --- Negative Log Marginal Likelihood ---
        # Formula: 0.5 * (log|K_full| + y' * inv(K_full) * y + n * log(2*pi))
        neg_log_lik = 0.5 * (log_det_K + quad_term + n_valid * log(2.0 * pi))

        # Check for non-finite results which can crash optimizer
        if !isfinite(neg_log_lik)
             # Penalize non-finite results heavily
             return Inf
        end

        return neg_log_lik

    catch e
        # Handle potential errors during kernelmatrix or cholesky (e.g., PosDefError)
        # Return Inf to indicate these parameters are invalid/unstable
         if e isa PosDefException || contains(string(lowercase(string(e))), "cholesky") || contains(string(lowercase(string(e))), "positive definite")
             # This commonly happens with extreme lengthscales or variances
             # println("Cholesky failed for params: ", exp.(log_params)) # Debug print
             return Inf # Penalize non-positive definite matrices
         else
             @error "Unexpected error in neg_log_lik:" exception=(e, catch_backtrace()) params=exp.(log_params)
             return Inf # Penalize other errors
         end
    end
end

# --------------------------------------------------------------------------
# Optimization Function
# --------------------------------------------------------------------------
"""
    optimize_gp_hyperparameters(y_obs_dim, t_obs, kernel_type, initial_log_params; ...)

Optimizes GP hyperparameters (variance, lengthscale, sigma) for a single dimension
by minimizing the negative log marginal likelihood.

Arguments:
- y_obs_dim: Vector of observations for the current dimension.
- t_obs: Vector of observation times.
- kernel_type: String specifying the kernel type.
- initial_log_params: Vector of initial guesses for [log(var), log(len), log(sigma)].
- jitter: Jitter value for numerical stability (optional).
- optim_options: Optim.Options for the optimizer (optional).

Returns:
- Vector containing optimized [variance, lengthscale, sigma].
"""
function optimize_gp_hyperparameters(y_obs_dim::Vector{Float64},
                                     t_obs::Vector{Float64},
                                     kernel_type::String,
                                     initial_log_params::Vector{Float64};
                                     jitter::Float64 = 1e-6,
                                     optim_options = Optim.Options(iterations = 100, show_trace=false)) # Example options

    # Define the objective function closure for Optim.jl
    objective = log_p -> negative_log_marginal_likelihood(log_p, y_obs_dim, t_obs, kernel_type, jitter)

    # Perform optimization using Nelder-Mead (gradient-free)
    # Box constraints could be added using Fminbox(NelderMead()) if needed,
    # but optimizing log-params often keeps them in valid range.
    result = Optim.optimize(objective,
                            initial_log_params,
                            NelderMead(), # Gradient-free method
                            optim_options)

    if !Optim.converged(result)
        @warn "GP hyperparameter optimization did not converge for a dimension. Using best found parameters." minimum=Optim.minimum(result) params=exp.(Optim.minimizer(result))
    end

    # Return optimized parameters transformed back to original scale
    optimized_log_params = Optim.minimizer(result)
    optimized_params = exp.(optimized_log_params) # [variance, lengthscale, sigma]

    # Add a check for potentially unreasonable values after optimization
    if any(x -> !isfinite(x) || x <= 0, optimized_params)
         @error "Optimization resulted in invalid parameters: $(optimized_params). Check initial guesses and data."
         # Fallback to initial guess (in original scale) or error?
         # Returning initial guess might be safer than erroring out completely.
         return exp.(initial_log_params)
    end


    return optimized_params
end


end # module Initialization
