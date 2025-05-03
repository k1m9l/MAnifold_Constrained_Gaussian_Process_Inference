# src/likelihoods.jl

module Likelihoods

using LinearAlgebra
using BandedMatrices
# Import types/modules from your own package needed here
using ..GaussianProcess: GPCov

export log_likelihood_and_gradient_banded

"""
    log_likelihood_and_gradient_banded(
        xlatent::AbstractMatrix{T},
        theta::AbstractVector{T},
        sigma::AbstractVector{T},
        yobs::AbstractMatrix{T},
        gp_cov_all_dims::Vector{GPCov},
        ode_f!::Function,
        ode_dfdx!::Function, # Jacobian dF/dX (in-place)
        ode_dfdp::Function; # Jacobian dF/dTheta (returns matrix)
        prior_temperature::AbstractVector{T} = [1.0, 1.0, 1.0]
    ) where {T<:Real}

Calculates the log-likelihood AND its gradient w.r.t. xlatent (vectorized)
and theta using banded matrix approximations. Corrected gradient accumulation
and NaN handling.

Arguments:
- xlatent: Matrix of latent states (n_times x n_dims)
- theta: Vector of ODE parameters (n_params)
- sigma: Vector of noise std dev per dimension (n_dims)
- yobs: Matrix of observations (n_times x n_dims), NaN for missing
- gp_cov_all_dims: Vector of pre-calculated GPCov structs (one per dim)
- ode_f!: The ODE function `f!(du, u, p, t)`
- ode_dfdx!: The ODE state Jacobian function `dfdx!(J, u, p, t)`
- ode_dfdp: The ODE parameter Jacobian function `dfdp(u, p, t)` (returns matrix)
- prior_temperature: Vector of temperatures [Deriv, Level, Obs] (Indices: 1=Deriv, 2=Level, 3=Obs)

Returns:
- A tuple `(log_likelihood::Float64, gradient::Vector{Float64})`
"""
function log_likelihood_and_gradient_banded(
        xlatent::AbstractMatrix{T},
        theta::AbstractVector{T},
        sigma::AbstractVector{T},
        yobs::AbstractMatrix{T},
        gp_cov_all_dims::Vector{GPCov},
        ode_f!::Function,
        ode_dfdx!::Function, # Takes J, u, p, t
        ode_dfdp::Function;  # Takes u, p, t, returns matrix
        prior_temperature::AbstractVector{T} = [1.0, 1.0, 1.0]
    ) where {T<:Real}

    n_times, n_dims = size(xlatent)
    n_params = length(theta)
    ll = zero(T) # Initialize log-likelihood
    # Resize gradient vector (now includes space for sigma gradients)
    total_grad_elements = n_times * n_dims + n_params + n_dims # <-- Added n_dims
    grad = zeros(T, total_grad_elements)

    # --- Argument Checks ---
    if size(yobs) != (n_times, n_dims)
        error("Dimensions of yobs $(size(yobs)) do not match xlatent $(size(xlatent))")
    end
    if length(sigma) != n_dims
        error("Length of sigma ($(length(sigma))) does not match number of dimensions ($n_dims)")
    end
    if length(gp_cov_all_dims) != n_dims
        error("Length of gp_cov_all_dims ($(length(gp_cov_all_dims))) does not match number of dimensions ($n_dims)")
    end
     if length(prior_temperature) != 3
        error("Length of prior_temperature ($(length(prior_temperature))) must be 3.")
    end


    # --- Pre-calculate ODE derivatives (fderiv) ---
    fderiv = similar(xlatent)
    u_temp = zeros(T, n_dims)
    du_temp = zeros(T, n_dims)
    if isempty(gp_cov_all_dims) || isempty(gp_cov_all_dims[1].tvec)
        error("GPCov structs must contain the time vector 'tvec'.")
    end
    tvec = gp_cov_all_dims[1].tvec
    if length(tvec) != n_times
        error("Length of tvec in GPCov ($(length(tvec))) does not match n_times from xlatent ($(n_times)).")
    end

    try
        for i in 1:n_times
            u_temp .= @view xlatent[i, :]
            # Use try-catch block for ODE function call
            ode_f!(du_temp, u_temp, theta, tvec[i])
            fderiv[i, :] .= du_temp
        end
    catch e
        @error "Error during ODE function (ode_f!) evaluation:" exception=(e, catch_backtrace()) theta=theta time_index=i state=u_temp
        rethrow(e)
    end


    sigma_sq = sigma.^2

    # --- Intermediate storage ---
    Kinv_fitDerivError_all = similar(xlatent)
    Cinv_x_all = similar(xlatent)
    fitLevelError_all = similar(xlatent) # Stores (x - y)
    idx_finite_all = BitMatrix(undef, n_times, n_dims) # Store indices of finite obs

    # --- Calculate Likelihood Value (Store intermediates) ---
    for pdim in 1:n_dims
        # Check gp_cov dimensions before accessing fields
        gp_cov = gp_cov_all_dims[pdim]
        if size(gp_cov.mphiBand, 1) != n_times || size(gp_cov.KinvBand, 1) != n_times || size(gp_cov.CinvBand, 1) != n_times
           error("Pre-calculated GP covariance matrices for dimension $pdim have incorrect size. Expected $n_times, got sizes: mphiBand=$(size(gp_cov.mphiBand)), KinvBand=$(size(gp_cov.KinvBand)), CinvBand=$(size(gp_cov.CinvBand))")
        end

        xlatent_dim = @view xlatent[:, pdim]
        fderiv_dim = @view fderiv[:, pdim]
        yobs_dim = @view yobs[:, pdim]

        fitLevelError = xlatent_dim .- yobs_dim # This is (x - y)
        idx_finite = isfinite.(yobs_dim)
        idx_finite_all[:, pdim] .= idx_finite # Store for gradient calculation
        fitLevelError[.!idx_finite] .= zero(T) # Zero out error for NaN observations
        nobs_dim = sum(idx_finite)
        fitLevelError_all[:, pdim] .= fitLevelError # Store (x - y), potentially zeroed for NaNs

        mphi_x = gp_cov.mphiBand * xlatent_dim
        fitDerivError = fderiv_dim .- mphi_x # This is (f - mphi*x)

        Kinv_fitDerivError = gp_cov.KinvBand * fitDerivError # Kinv*(f-mphi*x)
        Cinv_x = gp_cov.CinvBand * xlatent_dim              # Cinv*x
        Kinv_fitDerivError_all[:, pdim] .= Kinv_fitDerivError
        Cinv_x_all[:, pdim] .= Cinv_x

        # Log Likelihood Calculation
        # Observation term: Only include finite observations
        ll_obs = -0.5 * dot(fitLevelError[idx_finite], fitLevelError[idx_finite]) / sigma_sq[pdim]
        if nobs_dim > 0 # Avoid log(0) if no observations
            ll_obs -= 0.5 * nobs_dim * log(2.0 * pi * sigma_sq[pdim])
        end
        ll += ll_obs / prior_temperature[3]

        # Derivative term
        ll_deriv = -0.5 * dot(fitDerivError, Kinv_fitDerivError)
        ll += ll_deriv / prior_temperature[1]

        # Level term
        ll_level = -0.5 * dot(xlatent_dim, Cinv_x)
        ll += ll_level / prior_temperature[2]
    end

    # --- Calculate Gradient ---
    grad_x = @view grad[1:(n_times * n_dims)]
    grad_theta = @view grad[(n_times * n_dims + 1):(n_times * n_dims + n_params)]
    grad_sigma = @view grad[(n_times * n_dims + n_params + 1):end] # New view for sigma gradients
    grad_x_mat = reshape(grad_x, n_times, n_dims)

    grad_x_mat .= zero(T)
    grad_theta .= zero(T)
    grad_sigma .= zero(T) # Initialize sigma gradients

    J_state_temp = zeros(T, n_dims, n_dims)
    J_param_temp = zeros(T, n_dims, n_params)

    # Calculate gradient contributions dimension by dimension
    for pdim in 1:n_dims
        gp_cov = gp_cov_all_dims[pdim]
        Kinv_fitDerivError = @view Kinv_fitDerivError_all[:, pdim] # Kinv*(f-mphi*x) for dim p
        Cinv_x = @view Cinv_x_all[:, pdim]                         # Cinv*x for dim p
        fitLevelError = @view fitLevelError_all[:, pdim]           # (x-y) for dim p (NaNs are 0)
        idx_finite = @view idx_finite_all[:, pdim]                 # Finite indices for this dim

        # 1. Contribution from Observation term (dL_obs / dx_pdim) = -(x-y)/sigma^2
        #    Only add contribution for non-NaN observations
        for i in 1:n_times
            if idx_finite[i] # Check if observation is finite
                 grad_x_mat[i, pdim] -= (fitLevelError[i] / sigma_sq[pdim]) / prior_temperature[3]
            end
        end

        # 2. Contribution from Level term (dL_level / dx_pdim) = -Cinv*x
        #    This term is always present, regardless of observation status
        for i in 1:n_times
             grad_x_mat[i, pdim] -= Cinv_x[i] / prior_temperature[2]
        end

        # 3. Contribution from Derivative term (dL_deriv / dx and dL_deriv / dtheta)
        # Term 1: Contribution to dL/dx_pdim from mphi term = mphi'*Kinv*(f-mphi*x)
        #    This term is always present
        mphi_term = gp_cov.mphiBand' * Kinv_fitDerivError
        for i in 1:n_times
             grad_x_mat[i, pdim] += mphi_term[i] / prior_temperature[1]
        end

        # Term 2 & 3: Contributions involving ODE Jacobians
        #    These terms are always present
        for i in 1:n_times # Iterate through time
            u_temp .= @view xlatent[i, :] # Get current state
            KFE_i_scaled = Kinv_fitDerivError[i] / prior_temperature[1] # Pre-scale

            try
                ode_dfdx!(J_state_temp, u_temp, theta, tvec[i])
                J_param_temp .= ode_dfdp(u_temp, theta, tvec[i])
            catch e
                 @error "Error during ODE Jacobian evaluation:" exception=(e, catch_backtrace()) theta=theta time_index=i state=u_temp
                rethrow(e)
            end


            # Accumulate gradient contributions
            # dL/dx_j contribution from dF_p/dx_j
            for j in 1:n_dims
                 grad_x_mat[i, j] -= J_state_temp[pdim, j] * KFE_i_scaled
            end

            # dL/dtheta_k contribution from dF_p/dtheta_k
            for k in 1:n_params
                 grad_theta[k] -= J_param_temp[pdim, k] * KFE_i_scaled
            end
        end

        # --- NEW: Calculate Gradient w.r.t. sigma ---
        # Contribution from Observation term to dL/dsigma_p
        # dL/dsigma_p = (1/beta_3) * [ SSE_p / sigma_p^3 - Np / sigma_p ]
        # where SSE_p = dot(fitLevelError[idx_finite], fitLevelError[idx_finite])
        #       Np = sum(idx_finite)
        if sigma[pdim] > 0 # Avoid division by zero
            sse_p = zero(T)
            np = 0
            for i in 1:n_times
                if idx_finite[i]
                    sse_p += fitLevelError[i]^2
                    np += 1
                end
            end

            # Only calculate if there are observations for this dimension
            if np > 0
                # grad_L_p_wrt_sigma_p = (sse_p / sigma[pdim]^3 - np / sigma[pdim]) / prior_temperature[3]
                # Simplified:
                grad_L_p_wrt_sigma_p = (sse_p / sigma_sq[pdim] - np) / (sigma[pdim] * prior_temperature[3])
                grad_sigma[pdim] += grad_L_p_wrt_sigma_p
            end
        end
    end

    # Check for non-finite values in the final gradient
    if !all(isfinite, grad)
        @warn "Non-finite values detected in the calculated gradient." gradient=grad
        # Optionally replace NaN/Inf with zero or throw an error depending on desired behavior
        # grad[.!isfinite.(grad)] .= 0.0
    end

    return ll, grad # Return both value and gradient vector
end


end # module Likelihoodså