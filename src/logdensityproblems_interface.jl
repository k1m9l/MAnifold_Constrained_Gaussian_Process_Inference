# src/logdensityproblems_interface.jl
# UPDATED to handle fixed vs. sampled sigma

module LogDensityProblemsInterface

# Import the core MAGI likelihood function
using ..Likelihoods: log_likelihood_and_gradient_banded
# Import types needed from other modules
using ..GaussianProcess: GPCov
# using ..ODEModels # Not strictly needed here if types aren't used directly
# Import the interface definition package
using LogDensityProblems
using LinearAlgebra
using Logging # For warnings

export MagiTarget

"""
    MagiTarget

A struct that bundles all necessary components for the MAGI model and implements
the `LogDensityProblems.jl` interface. Includes a flag to indicate if sigma is fixed.

# Fields
# ... (descriptions for yobs, gp_cov_all_dims, ode_f!, etc. remain the same) ...
- `sigma_init::Vector{T}`: Vector of initial/fixed observation noise standard deviations σ.
- `prior_temperature::Vector{T}`: Vector of tempering factors β = [β_deriv, β_level, β_obs].
- `n_times::Int`: Number of time points `n`.
- `n_dims::Int`: Number of system dimensions `D`.
- `n_params_ode::Int`: Number of ODE parameters `k`.
- `sigma_is_fixed::Bool`: Flag indicating if sigma is fixed (true) or sampled (false).
"""
struct MagiTarget{T <: Real, GPCovVec <: AbstractVector{GPCov}, OdeFunc, OdeJacX, OdeJacP}
    yobs::Matrix{T}
    gp_cov_all_dims::GPCovVec
    ode_f!::OdeFunc
    ode_dfdx!::OdeJacX
    ode_dfdp::OdeJacP
    sigma_init::Vector{T} # Store the initial/fixed sigma value
    prior_temperature::Vector{T}
    n_times::Int
    n_dims::Int
    n_params_ode::Int
    sigma_is_fixed::Bool # <-- NEW FLAG
end

"""
    LogDensityProblems.dimension(target::MagiTarget)

Returns the total dimension of the parameter vector being sampled.
Dynamically adjusts based on whether sigma is fixed.
"""
function LogDensityProblems.dimension(target::MagiTarget)
    # Base dimension for xlatent and theta
    dim = target.n_times * target.n_dims + target.n_params_ode
    # Add dimensions for log_sigma only if sigma is NOT fixed
    if !target.sigma_is_fixed
        dim += target.n_dims
    end
    return dim
end

"""
    LogDensityProblems.capabilities(::Type{<:MagiTarget})

Specifies that `MagiTarget` provides log-density and gradient (LogDensityOrder{1}).
"""
function LogDensityProblems.capabilities(::Type{<:MagiTarget})
    return LogDensityProblems.LogDensityOrder{1}()
end

"""
    _unpack_params(target::MagiTarget, params::AbstractVector{T}) where {T}

Helper function to unpack the flat parameter vector `params` provided by the sampler.
Returns `(xlatent, theta, log_sigma)` if sigma is sampled,
or `(xlatent, theta, nothing)` if sigma is fixed.
"""
function _unpack_params(target::MagiTarget, params::AbstractVector{T}) where {T}
    n_xlatent_elements = target.n_times * target.n_dims
    n_theta_elements = target.n_params_ode
    n_sigma_elements = target.n_dims

    xlatent_flat = @view params[1:n_xlatent_elements]
    theta_flat = @view params[(n_xlatent_elements + 1):(n_xlatent_elements + n_theta_elements)]

    xlatent = reshape(xlatent_flat, target.n_times, target.n_dims)

    if target.sigma_is_fixed
        # If sigma is fixed, params only contains x and theta.
        # The dimension check ensures params has the correct length.
        @assert length(params) == n_xlatent_elements + n_theta_elements "Parameter vector length mismatch for fixed sigma case"
        return xlatent, theta_flat, nothing # Return nothing for log_sigma placeholder
    else
        # If sigma is sampled, params contains x, theta, and log_sigma.
        expected_len = n_xlatent_elements + n_theta_elements + n_sigma_elements
        @assert length(params) == expected_len "Parameter vector length mismatch when unpacking log_sigma"
        log_sigma_flat = @view params[(n_xlatent_elements + n_theta_elements + 1):end]
        return xlatent, theta_flat, log_sigma_flat
    end
end

"""
    LogDensityProblems.logdensity(target::MagiTarget, params::AbstractVector{T}) where {T}

Calculates the MAGI log-posterior density L(...) for the given parameter vector `params`.
Uses the fixed sigma from `target.sigma_init` if `target.sigma_is_fixed` is true,
otherwise transforms the sampled `log_sigma` from `params`. Includes log Jacobian term
for the transformation when sigma is sampled.
"""
function LogDensityProblems.logdensity(target::MagiTarget, params::AbstractVector{T}) where {T}
    # Check dimension reported by target matches input params length
     if length(params) != LogDensityProblems.dimension(target)
          @error "Dimension mismatch in logdensity!" expected=LogDensityProblems.dimension(target) got=length(params) sigma_fixed=target.sigma_is_fixed
          return -Inf
      end

    xlatent, theta, log_sigma_or_nothing = _unpack_params(target, params)
    local sigma::Vector{T}
    log_prior_contribution = zero(T) # For log Jacobian / explicit prior

    if target.sigma_is_fixed
        sigma = target.sigma_init # Use the fixed value stored in target
        # Basic check for validity of fixed sigma
        if any(s -> !isfinite(s) || s <= 0, sigma)
             @error "Fixed sigma value stored in MagiTarget is invalid!" sigma=sigma
             return -Inf
         end
    else
        # Sigma is sampled, transform from log_sigma
        log_sigma = log_sigma_or_nothing::AbstractVector{T}
        # Prevent issues with extreme log_sigma values during exp transformation
        # Clamp log_sigma to avoid Inf/zero sigma, adjust bounds as needed
        clamped_log_sigma = clamp.(log_sigma, -15.0, 15.0) # Example bounds
        sigma = exp.(clamped_log_sigma)
        if any(s -> !isfinite(s) || s <= 0, sigma) # Should not happen with clamp
            @warn "Transformed sigma is invalid after clamping." log_sigma=log_sigma sigma=sigma
            return -Inf # Invalid parameters
        end
        # Add log Jacobian determinant for log_sigma -> sigma transform
        # log Jacobian = sum(log(d(sigma)/d(log_sigma))) = sum(log(sigma)) = sum(log_sigma)
        # Use the clamped value for stability
        log_prior_contribution = sum(clamped_log_sigma) # Assumes uniform prior on log_sigma -> 1/sigma prior on sigma
        # Add explicit log prior terms for log_sigma here if desired
    end

    # Call the main likelihood function (ensure it handles potential zero sigma gracefully if needed)
    ll_likelihood, _ = log_likelihood_and_gradient_banded(
        xlatent, theta, sigma, target.yobs, target.gp_cov_all_dims,
        target.ode_f!, target.ode_dfdx!, target.ode_dfdp;
        prior_temperature = target.prior_temperature
    )

    # Add log jacobian / prior contribution if sigma was sampled
    total_ll = ll_likelihood
    if !target.sigma_is_fixed
         total_ll += log_prior_contribution
    end

    if !isfinite(total_ll)
         @warn "Log density calculation resulted in non-finite value." params=params total_ll=total_ll ll_likelihood=ll_likelihood log_prior=log_prior_contribution
         return -Inf
    end

    return total_ll
end


"""
    LogDensityProblems.logdensity_and_gradient(target::MagiTarget, params::AbstractVector{T}) where {T}

Calculates the MAGI log-posterior L and its gradient ∇L w.r.t. `params`.
Handles fixed vs. sampled sigma, including transformations and Jacobian terms.
The returned gradient has the dimension reported by `LogDensityProblems.dimension(target)`.
"""
function LogDensityProblems.logdensity_and_gradient(target::MagiTarget, params::AbstractVector{T}) where {T}
    # Check dimension reported by target matches input params length
    total_dim = LogDensityProblems.dimension(target)
    if length(params) != total_dim
        @error "Dimension mismatch in logdensity_and_gradient!" expected=total_dim got=length(params) sigma_fixed=target.sigma_is_fixed
        return -Inf, fill(NaN, total_dim) # Return NaN gradient on dimension error
    end

    xlatent, theta, log_sigma_or_nothing = _unpack_params(target, params)
    local sigma::Vector{T}
    log_prior_contribution = zero(T)
    grad_log_jacobian = zeros(T, target.n_dims) # Placeholder if sigma fixed

    if target.sigma_is_fixed
        sigma = target.sigma_init # Use the fixed value
        # Basic check for validity
        if any(s -> !isfinite(s) || s <= 0, sigma)
             @error "Fixed sigma value stored in MagiTarget is invalid!" sigma=sigma
             return -Inf, fill(NaN, total_dim)
         end
    else
        # Sigma is sampled
        log_sigma = log_sigma_or_nothing::AbstractVector{T}
        # Clamp log_sigma to avoid Inf/zero sigma
        clamped_log_sigma = clamp.(log_sigma, -15.0, 15.0) # Example bounds
        sigma = exp.(clamped_log_sigma)
        if any(s -> !isfinite(s) || s <= 0, sigma)
            @warn "Transformed sigma is invalid after clamping." log_sigma=log_sigma sigma=sigma
            return -Inf, fill(zero(T), total_dim) # Return zero grad
        end
        log_prior_contribution = sum(clamped_log_sigma)
        # Gradient of log Jacobian: d(sum(log_sigma))/d(log_sigma_p) = 1 for each p
        grad_log_jacobian = ones(T, target.n_dims)
        # Add gradient of explicit log prior terms for log_sigma here if desired
    end

    # Call the main likelihood function
    # Ensure log_likelihood_and_gradient_banded returns a gradient vector of size
    # n_x + n_theta + n_sigma (it calculates dL_likelihood/dsigma)
    ll_likelihood, grad_likelihood = log_likelihood_and_gradient_banded(
         xlatent, theta, sigma, target.yobs, target.gp_cov_all_dims,
         target.ode_f!, target.ode_dfdx!, target.ode_dfdp;
         prior_temperature = target.prior_temperature
    )

    # Check if likelihood gradient calculation failed
    if !isfinite(ll_likelihood) || !all(isfinite, grad_likelihood)
        @warn "Likelihood or its gradient calculation resulted in non-finite value(s)." ll=ll_likelihood
        # Consider inspecting grad_likelihood components if debugging needed
        return -Inf, fill(zero(T), total_dim)
    end

    # Initialize final gradient vector with the correct dimension based on the target flag
    final_grad = zeros(T, total_dim)
    n_x_theta = target.n_times * target.n_dims + target.n_params_ode

    # Assign gradients for x and theta (always present)
    final_grad[1:n_x_theta] = @view grad_likelihood[1:n_x_theta]

    # Calculate total log likelihood and handle sigma gradient based on flag
    total_ll = ll_likelihood
    if !target.sigma_is_fixed
        total_ll += log_prior_contribution

        # Extract grad_sigma (dL_likelihood / dsigma) from likelihood gradient result
        # Assuming grad_likelihood has size n_x + n_theta + n_sigma
        if length(grad_likelihood) != (n_x_theta + target.n_dims)
             @error "Gradient from likelihood function has unexpected size!" expected=n_x_theta + target.n_dims got=length(grad_likelihood)
             return -Inf, fill(NaN, total_dim)
        end
        grad_sigma_likelihood = @view grad_likelihood[(n_x_theta + 1):end]

        # Apply chain rule for log_sigma: dL/d(log_sigma) = dL/dsigma * sigma
        grad_log_sigma_likelihood = grad_sigma_likelihood .* sigma # Element-wise multiplication

        # Add gradient of log Jacobian / prior term
        # This gradient component only exists if sigma is NOT fixed
        final_grad[(n_x_theta + 1):end] = grad_log_sigma_likelihood + grad_log_jacobian

    end
    # Note: If sigma_is_fixed, the elements of final_grad beyond n_x_theta remain zero,
    # and total_dim ensures final_grad has the correct (smaller) size.

    # Final check for finite values in the assembled gradient
    if !all(isfinite, final_grad)
        @warn "Final gradient calculation resulted in non-finite value(s)."
        # Inspect final_grad components if debugging needed
        return total_ll, fill(zero(T), total_dim) # Return zero grad
    end

    return total_ll, final_grad
end

end # module LogDensityProblemsInterface