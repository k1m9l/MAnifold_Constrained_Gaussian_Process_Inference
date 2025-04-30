"""
    MagiJl

MAnifold-constrained Gaussian process Inference (MAGI) in Julia.
"""
module MagiJl

# =========
# External Dependencies
# =========
using LinearAlgebra
using KernelFunctions
using BandedMatrices
using PositiveFactorizations
using LogDensityProblems
using AdvancedHMC
using Logging
using Statistics # For var, median, abs
using Optim      # Now needed here if called directly, or within Initialization

# =========
# Includes: Order matters if files depend on each other
# =========
include("ode_models.jl")
include("kernels.jl")
include("gaussian_process.jl")
include("likelihoods.jl")
include("logdensityproblems_interface.jl")
include("samplers.jl")
include("initialization.jl") # <<<--- INCLUDE THE NEW FILE

# =========
# Usings: Bring submodules into the main MagiJl scope
# =========
using .ODEModels
using .Kernels
using .GaussianProcess
using .Likelihoods
using .LogDensityProblemsInterface
using .Samplers
using .Initialization # <<<--- USE THE NEW MODULE

# =========
# Exports: Make functions/types available to users via 'using MagiJl'
# =========

# From ode_models.jl
export OdeSystem
export fn_ode!, hes1_ode!, hes1log_ode!, hes1log_ode_fixg!, hes1log_ode_fixf!, hiv_ode!, ptrans_ode!
export fn_ode_dx!, fn_ode_dtheta, hes1_ode_dx!, hes1_ode_dtheta

# From kernels.jl
export create_rbf_kernel, create_matern52_kernel, create_general_matern_kernel

# From gaussian_process.jl
export GPCov, calculate_gp_covariances!

# From likelihoods.jl
export log_likelihood_and_gradient_banded

# From LogDensityProblemsInterface
export MagiTarget

# From Samplers
export run_nuts_sampler

# From Initialization (Optional - only if users need direct access)
# export optimize_gp_hyperparameters

# Main function
export solve_magi


"""
    solve_magi(y_obs, t_obs, ode_system, config; ...)
Main function to run the MAGI algorithm.
... (docstring remains the same) ...
"""
function solve_magi(
    y_obs::Matrix{Float64},
    t_obs::Vector{Float64},
    ode_system::OdeSystem,
    config::Dict{Symbol,Any}=Dict{Symbol,Any}();
    initial_params=nothing
)

    @info "Starting MAGI solver..."
    println("Input Data: $(size(y_obs, 1)) time points, $(size(y_obs, 2)) dimensions.")
    println("Time points from $(minimum(t_obs)) to $(maximum(t_obs)).")
    println("Using ODE System with $(ode_system.thetaSize) parameters.")
    println("Config: ", config)

    # --- 1 & 2: Extract Info & Config ---
    kernel_type = get(config, :kernel, "matern52")
    niter_hmc = get(config, :niterHmc, 20000)
    burnin_ratio = get(config, :burninRatio, 0.5)
    step_size_factor = get(config, :stepSizeFactor, 0.01)
    band_size = get(config, :bandSize, 20)
    prior_temperature = get(config, :priorTemperature, [1.0, 1.0, 1.0])
    sigma_exogenous = get(config, :sigma, Float64[])
    phi_exogenous = get(config, :phi, Matrix{Float64}(undef, 0, 0))
    x_init_exogenous = get(config, :xInit, Matrix{Float64}(undef, 0, 0))
    theta_init_exogenous = get(config, :thetaInit, Float64[])
    target_accept_ratio = get(config, :targetAcceptRatio, 0.8)
    jitter = get(config, :jitter, 1e-6)

    # Dimensions
    n_times = length(t_obs)
    n_dims = size(y_obs, 2)
    n_params_ode = ode_system.thetaSize

    # --- 3. Initialize GP Hyperparameters (phi) & Observation Noise (sigma) ---
    local phi_all_dimensions::Matrix{Float64}
    local sigma::Vector{Float64} # Define sigma here

    if isempty(phi_exogenous) || isempty(sigma_exogenous)
        println("Optimizing GP hyperparameters (phi and sigma) using marginal likelihood...")
        phi_all_dimensions = zeros(2, n_dims)
        sigma = zeros(n_dims) # Initialize sigma vector
        optim_opts = Optim.Options(
            iterations = get(config, :gpOptimIterations, 100),
            show_trace = get(config, :gpOptimShowTrace, false),
            f_tol = get(config, :gpOptimFTol, 1e-8), # Add tolerance options
            g_tol = get(config, :gpOptimGTol, 1e-8)  # g_tol might not be used by NelderMead
        )

        for dim in 1:n_dims
            println("  Optimizing dimension $dim...")
            y_dim = y_obs[:, dim]

            # Initial guess (log-transformed) using heuristics
            log_var_guess, log_len_guess, log_sigma_guess = 0.0, 0.0, 0.0
            valid_y = filter(!isnan, y_dim)
            if !isempty(valid_y) && length(valid_y) > 1 # Need >1 point for variance/MAD
                var_y = var(valid_y; corrected=true)
                data_range = maximum(valid_y) - minimum(valid_y)
                time_range = maximum(t_obs) - minimum(t_obs)
                mad_val = median(abs.(valid_y .- median(valid_y))) * 1.4826

                log_var_guess = log(max(var_y, 1e-4))
                log_len_guess = log(max(time_range / 10.0, 1e-2))
                log_sigma_guess = log(max(mad_val, 1e-3 * data_range, 1e-4))
            else
                # Default guesses if no data or single point
                log_var_guess = log(1.0)
                log_len_guess = log(max((maximum(t_obs) - minimum(t_obs)) / 10.0, 1e-2))
                log_sigma_guess = log(0.1)
            end
            initial_log_params = [log_var_guess, log_len_guess, log_sigma_guess]
            println("    Initial guess [log(var), log(len), log(sigma)]: ", round.(initial_log_params, digits=3))

            # Call the optimization function from the Initialization module
            optimized_params = Initialization.optimize_gp_hyperparameters( # <<<--- Use Initialization. prefix
                y_dim,
                t_obs,
                kernel_type,
                initial_log_params;
                jitter=jitter,
                optim_options=optim_opts
            )
            println("    Optimized [var, len, sigma]: ", round.(optimized_params, digits=4))

            # Store results
            phi_all_dimensions[1, dim] = optimized_params[1] # variance
            phi_all_dimensions[2, dim] = optimized_params[2] # lengthscale
            sigma[dim] = optimized_params[3]                # sigma
        end
        println("Optimization complete.")

    # Handle cases where only one is provided (optional, could error instead)
    elseif isempty(phi_exogenous) && !isempty(sigma_exogenous)
         error("If providing :sigma exogenously, must also provide :phi.")
    elseif !isempty(phi_exogenous) && isempty(sigma_exogenous)
         error("If providing :phi exogenously, must also provide :sigma.")
    else
        # Both phi and sigma provided exogenously
         if size(phi_exogenous) != (2, n_dims)
             error("Provided :phi matrix has wrong dimensions. Expected (2, $n_dims), got $(size(phi_exogenous)).")
         end
         if length(sigma_exogenous) != n_dims
             error("Provided :sigma vector has wrong length. Expected $n_dims, got $(length(sigma_exogenous)).")
         end
         phi_all_dimensions = phi_exogenous
         sigma = sigma_exogenous
     end
    # Print the final phi/sigma being used
    println("Using phi (GP hyperparameters):\n", round.(phi_all_dimensions, digits=4))
    println("Using sigma (observation noise SD): ", round.(sigma, digits=4))

    # --- 3. Initialize Latent States (x) ---
    # ... (x_init logic remains the same) ...
    local x_init::Matrix{Float64}
    if isempty(x_init_exogenous)
        x_init = zeros(n_times, n_dims)
        for dim in 1:n_dims
            valid_indices = findall(!isnan, y_obs[:, dim])
            if isempty(valid_indices)
                x_init[:, dim] .= 0.0 # Fill with zeros if no data
                @warn "No observations found for dimension $dim. Initializing x with zeros."
                continue
            end

            valid_times = t_obs[valid_indices]
            valid_values = y_obs[valid_indices, dim]

            for i in 1:n_times
                t = t_obs[i]
                # Manual linear interpolation/extrapolation
                if t <= valid_times[1]
                    x_init[i, dim] = valid_values[1]
                elseif t >= valid_times[end]
                    x_init[i, dim] = valid_values[end]
                else
                    # Find interval
                    idx_lower = findlast(valid_times .<= t)
                    # Handle cases where t might be exactly on a valid_time
                    if t == valid_times[idx_lower]
                         x_init[i, dim] = valid_values[idx_lower]
                         continue
                    end
                    idx_upper = idx_lower + 1 # Since t is between first and last
                    # Boundary check for upper index (can happen with floating point)
                    if idx_upper > length(valid_times)
                        x_init[i, dim] = valid_values[end]
                        continue
                    end

                    t_lower = valid_times[idx_lower]
                    t_upper = valid_times[idx_upper]
                    v_lower = valid_values[idx_lower]
                    v_upper = valid_values[idx_upper]

                    # Avoid division by zero if time points are identical
                    if t_upper == t_lower
                         x_init[i, dim] = v_lower
                    else
                         weight = (t - t_lower) / (t_upper - t_lower)
                         x_init[i, dim] = (1 - weight) * v_lower + weight * v_upper
                    end
                end
            end
        end
    else
         if size(x_init_exogenous) != (n_times, n_dims)
             error("Provided :xInit matrix has wrong dimensions. Expected ($n_times, $n_dims), got $(size(x_init_exogenous)).")
         end
        x_init = x_init_exogenous
    end

    # --- 3. Initialize ODE Parameters (theta) ---
    # ... (theta_init logic remains the same) ...
    local theta_init::Vector{Float64}
    if isempty(theta_init_exogenous)
        theta_init = zeros(n_params_ode)
        for i in 1:n_params_ode
            lb = ode_system.thetaLowerBound[i]
            ub = ode_system.thetaUpperBound[i]
            if isfinite(lb) && isfinite(ub)
                theta_init[i] = (lb + ub) / 2.0
            elseif isfinite(lb)
                theta_init[i] = lb + 1.0 # Start slightly above lower bound
            elseif isfinite(ub)
                theta_init[i] = ub - 1.0 # Start slightly below upper bound
            else
                theta_init[i] = 0.0 # Default if unbounded (consider 1.0?)
            end
            # Ensure init is within bounds if they are finite
             theta_init[i] = clamp(theta_init[i], lb, ub)
        end
    else
         if length(theta_init_exogenous) != n_params_ode
             error("Provided :thetaInit vector has wrong length. Expected $n_params_ode, got $(length(theta_init_exogenous)).")
         end
        theta_init = theta_init_exogenous
        # Check if provided init respects bounds
        if any(theta_init .< ode_system.thetaLowerBound) || any(theta_init .> ode_system.thetaUpperBound)
             @warn "Provided :thetaInit contains values outside the specified bounds. Clamping initial values."
             theta_init = clamp.(theta_init, ode_system.thetaLowerBound, ode_system.thetaUpperBound)
        end
    end
    println("Initial theta: ", round.(theta_init, digits=4))


    # --- 4. Calculate GPCov Structs ---
    # ... (GPCov calculation logic remains the same, uses the determined phi_all_dimensions) ...
    @info "Calculating GP Covariance structures..."
    cov_all_dimensions = Vector{GPCov}(undef, n_dims)
    actual_band_size = min(band_size, n_times - 1) # Ensure bandsize isn't too large
    println("Using Band Size: $actual_band_size (Requested: $band_size)")

    for dim in 1:n_dims
        cov_all_dimensions[dim] = GPCov()
        local kernel # Ensure kernel is defined in this scope
        phi_dim = phi_all_dimensions[:, dim]
        var = phi_dim[1]
        len = phi_dim[2]

        if kernel_type == "matern52"
            kernel = Kernels.create_matern52_kernel(var, len)
        elseif kernel_type == "rbf"
            kernel = Kernels.create_rbf_kernel(var, len)
        # Add other kernel types here if needed
        else
            @warn "Unsupported kernel type '$kernel_type'. Defaulting to matern52."
            kernel = Kernels.create_matern52_kernel(var, len)
        end

        try
             calculate_gp_covariances!(
                 cov_all_dimensions[dim],
                 kernel,
                 phi_dim,
                 t_obs,
                 actual_band_size; # Use adjusted band size
                 complexity=2,      # Assume complexity 2 for derivatives
                 jitter=jitter      # Pass jitter value
             )
        catch e
             @error "Failed to calculate GP covariances for dimension $dim." phi=phi_dim kernel=kernel bandsize=actual_band_size jitter=jitter
             rethrow(e)
         end
    end
    @info "GP Covariance calculation complete."


    # --- 5. Define Log Posterior Target ---
    # ... (MagiTarget setup remains the same, uses the determined sigma) ...
    if !(typeof(prior_temperature) <: AbstractVector && length(prior_temperature) == 3)
         @warn "priorTemperature should be a vector of 3 floats [Deriv, Level, Obs]. Using [$prior_temperature, $prior_temperature, $prior_temperature]."
         prior_temps = fill(Float64(prior_temperature[1]), 3) # Assume scalar if not vector
    else
         prior_temps = convert(Vector{Float64}, prior_temperature)
    end
    println("Using Prior Temperatures [Deriv, Level, Obs]: ", prior_temps)

    target = MagiTarget(
        y_obs,
        cov_all_dimensions,
        ode_system.fOde,
        ode_system.fOdeDx,
        ode_system.fOdeDtheta,
        sigma, # Use the determined sigma
        prior_temps,
        n_times,
        n_dims,
        n_params_ode
    )

    # --- 6. Initialize Full Parameter Vector for Sampler ---
    # ... (params_init logic remains the same) ...
    local params_init::Vector{Float64}
    if initial_params === nothing
        params_init = vcat(vec(x_init), theta_init)
    else
         if length(initial_params) != (n_times * n_dims + n_params_ode)
             error("Provided initial_params vector has wrong length. Expected $(n_times * n_dims + n_params_ode), got $(length(initial_params)).")
         end
        params_init = initial_params
        # Optional: Check if the theta part of initial_params respects bounds
        theta_part_init = initial_params[(n_times * n_dims + 1):end]
         if any(theta_part_init .< ode_system.thetaLowerBound) || any(theta_part_init .> ode_system.thetaUpperBound)
             @warn "Theta part of provided initial_params contains values outside bounds. Sampler might struggle or reject states."
         end
    end
    total_params_dim = length(params_init)
    println("Total number of parameters being sampled: $total_params_dim")


    # --- 7. Run Sampler ---
    # ... (run_nuts_sampler call remains the same) ...
    @info "Setting up and running NUTS sampler..."
    n_adapts = Int(floor(niter_hmc * burnin_ratio))
    n_samples_total = niter_hmc # Total iterations passed to sampler

    # Use step_size_factor directly as initial step size for AdvancedHMC > 0.4
    initial_step_size = step_size_factor # Can be scalar or vector matching params_init length

    # Use the dedicated sampler function
    # run_nuts_sampler returns chain (Vector{Vector}) and stats (Vector{NamedTuple})
    chain, stats = run_nuts_sampler(
        target,
        params_init;
        n_samples = n_samples_total,
        n_adapts = n_adapts,
        target_accept_ratio = target_accept_ratio,
        initial_step_size = initial_step_size
    )

    if chain === nothing || isempty(chain)
        error("Sampling failed or returned no samples.")
    end
    @info "Sampling completed."

    # --- 8. Process Results ---
    # ... (Result processing logic remains the same) ...
    @info "Processing MCMC samples..."
    # Convert chain (Vector{Vector{Float64}}) to Matrix{Float64} [param_dim, n_samples_post_burnin]
    # Note: Since drop_warmup=true was used, the chain already excludes warmup.
    samples_post_burnin_matrix = hcat(chain...) # Convert Vector{Vector} to Matrix
    n_samples_post_burnin = size(samples_post_burnin_matrix, 2)

    if n_samples_post_burnin <= 0
         error("Sampler returned no post-burn-in samples.")
    end

    # Extract components
    x_indices = 1:(n_times * n_dims)
    theta_indices = (n_times * n_dims + 1):(total_params_dim) # End index is total dim

    # Reshape x samples to 3D array [sample_idx, time, dimension]
    x_samples_flat = samples_post_burnin_matrix[x_indices, :] # Shape [n_x_params, n_samples_post_burnin]
    # Need to transpose flat samples before reshaping
    x_samples = Array{Float64, 3}(undef, n_samples_post_burnin, n_times, n_dims)

    for i in 1:n_samples_post_burnin
        # This line should now work correctly
        x_samples[i, :, :] = reshape(x_samples_flat[:, i], n_times, n_dims)
    end

    # Extract theta samples
    theta_samples = Matrix(samples_post_burnin_matrix[theta_indices, :]') # Transpose to [samples, parameters]

    # Extract log posterior values (from stats)
    lp_values = Float64[]
    # Check stats: it should be a Vector of NamedTuples after warmup is dropped
    if stats isa Vector && !isempty(stats) && length(stats) == n_samples_post_burnin
        first_stat = first(stats)
        if hasproperty(first_stat, :lp)
            lp_values = [s.lp for s in stats]
        elseif hasproperty(first_stat, :log_density)
            lp_values = [s.log_density for s in stats]
        else
             @warn "Sampler statistics vector does not contain :lp or :log_density field. Cannot extract log posterior values."
        end
    else
         # This warning handles the case where the get() call previously failed
         stats_type = typeof(stats)
         stats_len = try length(stats) catch; -1 end # Avoid error if length doesn't apply
         @warn "Log posterior values could not be extracted. Sampler statistics structure is unexpected (Type: $stats_type, Length: $stats_len, Expected Length: $n_samples_post_burnin). Returning empty vector."
    end

    @info "Finished processing results."

    # Return results as a NamedTuple
    return (
        theta = theta_samples,
        x_sampled = x_samples,
        sigma = reshape(sigma, 1, n_dims), # Return the sigma used (optimized or exogenous)
        phi = phi_all_dimensions,          # Return the phi used (optimized or exogenous)
        lp = lp_values
    )

end # end solve_magi

end # module MagiJl
