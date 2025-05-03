# src/MajiJl.jl

"""
    MagiJl

Provides the Julia implementation of the MAnifold-constrained Gaussian process Inference (MAGI)
method for Bayesian inference of Ordinary Differential Equation (ODE) parameters (θ) and
latent state trajectories (x(t)) from noisy, sparse, and potentially partially observed data.

This module orchestrates the MAGI workflow, including:
- Defining ODE systems and GP kernels.
- Calculating GP covariance matrices and their derivatives.
- Initializing parameters (θ, x(t), σ, ϕ) using data-driven heuristics and optimization.
- Setting up the log-posterior density function compatible with `LogDensityProblems.jl`.
- Running the NUTS sampler via `AdvancedHMC.jl` to obtain posterior samples.
- Providing post-processing functions for summarizing and visualizing results.
"""
module MagiJl

# =========
# External Dependencies
# =========
using LinearAlgebra
using KernelFunctions          # For defining GP kernels
using BandedMatrices           # For efficient banded matrix operations
using PositiveFactorizations   # For robust Cholesky decomposition
using LogDensityProblems       # Interface for target density/gradient
using AdvancedHMC            # NUTS sampler implementation
using Logging                  # For @info, @warn, @error messages
using Statistics               # For var, median, mean used in initialization/summaries
using Optim                    # For GP hyperparameter optimization during initialization
using Printf                   # For formatted printing in summary
using Interpolations           # For linear interpolation in initialization
using StatsPlots             # For plotting in postprocessing
using Plots                   # For plotting in postprocessing

# Optional Dependencies for Postprocessing (Load if available)
using Requires
@static if !isdefined(Base, :get_extension)
    using Requires # Ensure Requires is loaded for Julia < 1.9
end

# Define placeholder functions if packages are not loaded
function __init__()
    @static if !isdefined(Base, :get_extension)
        # Conditional loading for Julia < 1.9 using Requires.jl
        @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
            @require StatsPlots="f3b207a7-027a-5e70-b257-86293d7955fd" begin
                 @info "Plots and StatsPlots loaded for visualization."
            end
        end
        @require MCMCChains="c7f686f2-ff18-58e9-bbe5-17545b1d6079" begin
             @info "MCMCChains loaded for summary statistics and trace plots."
        end
    end
end

# Function to safely check if an optional package is loaded and usable
macro isloaded(pkg)
    return :(isdefined(Main, $(QuoteNode(pkg))) && isa(getfield(Main, $(QuoteNode(pkg))), Module))
end


# =========
# Includes: Order matters if files depend on each other
# =========
include("ode_models.jl")
include("kernels.jl")
include("gaussian_process.jl")
include("likelihoods.jl") # Needs updated gradient calc for sigma
include("logdensityproblems_interface.jl") # Needs updated dimension and unpacking
include("samplers.jl")
include("initialization.jl")
# Postprocessing functions are now integrated directly below

# =========
# Usings: Bring submodules into the main MagiJl scope for internal use
# =========
using .ODEModels
using .Kernels
using .GaussianProcess
using .Likelihoods
using .LogDensityProblemsInterface
using .Samplers
using .Initialization

# =========
# Exports: Make functions/types available to users via 'using MagiJl'
# =========

# From ode_models.jl
export OdeSystem # Struct for ODE definition
# Specific ODE system functions (examples)
export fn_ode!, hes1_ode!, hes1log_ode!, hes1log_ode_fixg!, hes1log_ode_fixf!, hiv_ode!, ptrans_ode!
# Specific ODE Jacobian functions (examples)
export fn_ode_dx!, fn_ode_dtheta, hes1_ode_dx!, hes1_ode_dtheta # Note: Using θ in export name

# From kernels.jl
export create_rbf_kernel, create_matern52_kernel, create_general_matern_kernel

# From gaussian_process.jl
export GPCov, calculate_gp_covariances!

# From likelihoods.jl
export log_likelihood_and_gradient_banded

# From LogDensityProblemsInterface
export MagiTarget # The adapter struct for LogDensityProblems

# From Samplers
export run_nuts_sampler # The function wrapping AdvancedHMC

# Main solver function
export solve_magi

# From Postprocessing (Now defined directly in this module)
export magi_summary, plot_magi, results_to_chain

# =========================================================================
# solve_magi function definition - UPDATED to sample sigma
# =========================================================================
"""
    solve_magi(
        y_obs::Matrix{Float64},
        t_obs::Vector{Float64},
        ode_system::OdeSystem,
        config::Dict{Symbol,Any}=Dict{Symbol,Any}();
        initial_params=nothing
    )

Solves the MAGI inference problem.

Estimates ODE parameters (θ), latent state trajectories (x(t)), and observation noise
standard deviations (σ) from noisy observations `y_obs` at time points `t_obs`,
given an ODE system definition `ode_system`.

# Arguments
- `y_obs::Matrix{Float64}`: Matrix of observations (n_times × n_dims). Use `NaN` for missing values.
- `t_obs::Vector{Float64}`: Vector of time points corresponding to `y_obs` rows (discretization grid).
- `ode_system::OdeSystem`: Struct containing the ODE function (`fOde`), Jacobians (`fOdeDx`, `fOdeDtheta`), parameter bounds, and size.
- `config::Dict{Symbol,Any}`: Dictionary to control solver settings (optional). See details below.
- `initial_params::Union{Vector{Float64}, Nothing}`: Optional. A single vector containing starting values for ALL parameters to be sampled: `[vec(x_init); theta_init; log(sigma_init)]`. If provided, it overrides individual initializations (`xInit`, `thetaInit`, initial `sigma` estimate). Default: `nothing`.

# Config Options (Partial List)
- `:kernel`: String, GP kernel type ("matern52", "rbf"). Default: "matern52".
- `:niterHmc`: Int, total HMC iterations. Default: 20000.
- `:burninRatio`: Float, proportion of iterations for warmup. Default: 0.5.
- `:stepSizeFactor`: Float, initial HMC step size factor. Default: 0.01.
- `:bandSize`: Int, band matrix approximation width. Default: 20.
- `:priorTemperature`: Vector{Float}, tempering factors [β_deriv, β_level, β_obs]. Default: [1.0, 1.0, 1.0].
- `:sigma`: Vector{Float}, known noise standard deviations σ. If provided (and `:phi` is also provided), σ is treated as fixed and NOT sampled. Default: `Float64[]` (σ is unknown and sampled).
- `:phi`: Matrix{Float}, known GP hyperparameters [variance; lengthscale] per dimension. Must be provided if `:sigma` is provided and intended to be fixed. Default: `Matrix{Float64}(undef, 0, 0)` (ϕ is estimated).
- `:xInit`: Matrix{Float}, initial guess for latent states x(t). Default: Linear interpolation of `y_obs`.
- `:thetaInit`: Vector{Float}, initial guess for ODE parameters θ. Default: Midpoint of bounds or small offset.
- `:targetAcceptRatio`: Float, target acceptance rate for HMC adaptation. Default: 0.8.
- `:jitter`: Float, small value added for numerical stability. Default: 1e-6.
- `:gpOptimIterations`: Int, iterations for GP hyperparameter optimization. Default: 100.
- `:verbose`: Bool, print progress messages. Default: `false`.

# Returns
- `NamedTuple`: Contains the inference results:
    - `theta::Matrix{Float64}`: Posterior samples for θ (n_samples × n_params_ode).
    - `x_sampled::Array{Float64, 3}`: Posterior samples for x(t) (n_samples × n_times × n_dims).
    - `sigma::Matrix{Float64}`: Posterior samples for σ (n_samples × n_dims) if σ was estimated, otherwise the fixed input σ repeated (n_samples x n_dims).
    - `phi::Matrix{Float64}`: Used/estimated GP hyperparameters ϕ (2 × n_dims).
    - `lp::Vector{Float64}`: Log-posterior density values for each sample.
- `nothing`: If the solver encounters a critical error.

"""
function solve_magi(
    y_obs::Matrix{Float64},       # Observations y(τ)
    t_obs::Vector{Float64},       # Time points t ∈ I (discretization grid)
    ode_system::OdeSystem,      # ODE definition (f, ∂f/∂x, ∂f/∂θ, bounds)
    config::Dict{Symbol,Any}=Dict{Symbol,Any}(); # Control dictionary
    initial_params::Union{Vector{Float64}, Nothing}=nothing  # Optional full starting vector [vec(x₀); θ₀; log(σ₀)]
)
    # --- Start of MAGI Workflow ---
    @info "Starting MAGI solver..."
    println("Input Data: $(size(y_obs, 1)) time points, $(size(y_obs, 2)) dimensions.")
    println("Time points from $(minimum(t_obs)) to $(maximum(t_obs)).")
    println("Using ODE System with $(ode_system.thetaSize) parameters.")
    # Print config carefully
    print("Config: {")
    config_items = []
    for (k, v) in config
        item_str = if k == :xInit && !isempty(v)
             ":xInit => <matrix: $(size(v))>"
        elseif k == :phi && !isempty(v)
             ":phi => <matrix: $(size(v))>"
        elseif v isa AbstractArray && length(v) > 10
             ":$k => <array: $(typeof(v)), size=$(size(v))>"
        elseif length(string(v)) > 100 # Avoid printing huge strings
            ":$k => <$(typeof(v))>"
        else
            ":$k => $v"
        end
        push!(config_items, item_str)
    end
    println(join(config_items, ", "), "}")


    # --- 1 & 2: Extract Dimensions & Configuration Settings ---
    n_times = length(t_obs)       # Number of time points n
    n_dims = size(y_obs, 2)       # Number of state dimensions D
    n_params_ode = ode_system.thetaSize # Number of ODE parameters k

    # Extract settings
    kernel_type = get(config, :kernel, "matern52")
    niter_hmc = get(config, :niterHmc, 20000)
    burnin_ratio = get(config, :burninRatio, 0.5)
    step_size_factor = get(config, :stepSizeFactor, 0.01)
    band_size = get(config, :bandSize, 20)
    prior_temperature = get(config, :priorTemperature, [1.0, 1.0, 1.0])
    sigma_exogenous = get(config, :sigma, Float64[]) # Provided known σ?
    phi_exogenous = get(config, :phi, Matrix{Float64}(undef, 0, 0)) # Provided known ϕ?
    x_init_exogenous = get(config, :xInit, Matrix{Float64}(undef, 0, 0))
    theta_init_exogenous = get(config, :thetaInit, Float64[])
    target_accept_ratio = get(config, :targetAcceptRatio, 0.8)
    jitter = get(config, :jitter, 1e-6)
    verbose_print = get(config, :verbose, false) # Control internal prints

    # Determine if sigma is fixed or needs to be sampled
    # Sigma is fixed ONLY if BOTH sigma_exogenous AND phi_exogenous are provided
    sigma_is_fixed = !isempty(sigma_exogenous) && !isempty(phi_exogenous)
    if sigma_is_fixed
        println("Sigma provided exogenously and treated as fixed.")
        if length(sigma_exogenous) != n_dims
             error("Provided :sigma vector has wrong length. Expected $n_dims, got $(length(sigma_exogenous)).")
         end
         if size(phi_exogenous) != (2, n_dims)
             error("Provided :phi matrix has wrong dimensions when sigma is fixed. Expected (2, $n_dims), got $(size(phi_exogenous)).")
         end
    else
        println("Sigma is treated as unknown and will be sampled.")
        if !isempty(sigma_exogenous) && isempty(phi_exogenous)
            @warn "Sigma provided but Phi not provided. Sigma will be treated as unknown and re-initialized."
            sigma_exogenous = Float64[] # Reset sigma_exogenous to trigger initialization
        end
         if isempty(sigma_exogenous) && !isempty(phi_exogenous)
            @warn "Phi provided but Sigma not provided. Sigma will be treated as unknown and initialized."
            # Phi will still be used if provided, sigma will be initialized.
        end
    end

    # --- 3. Initialize GP Hyperparameters (ϕ) & Observation Noise (σ) ---
    local ϕ_all_dimensions::Matrix{Float64}
    local sigma_init::Vector{Float64} # Stores the *initial* value for sigma

    # Condition 1: Phi is NOT provided OR Sigma is NOT fixed (needs initialization)
    if isempty(phi_exogenous) || !sigma_is_fixed

        # Option A: Estimate ϕ and potentially σ using GP marginal likelihood optimization
        if isempty(phi_exogenous) && isempty(sigma_exogenous)
            println("Optimizing GP hyperparameters (ϕ and σ) using marginal likelihood...")
        elseif isempty(phi_exogenous) && !isempty(sigma_exogenous) && !sigma_is_fixed # This case shouldn't happen due to logic above, but for safety
             println("Optimizing GP hyperparameters (ϕ and σ) using marginal likelihood (sigma input ignored)...")
        elseif !isempty(phi_exogenous) && isempty(sigma_exogenous)
             println("Optimizing observation noise (σ) using marginal likelihood (using provided ϕ)...")
        end

        # Prepare storage
        ϕ_est = zeros(2, n_dims)
        σ_est = zeros(n_dims) # Only this is used if phi is exogenous but sigma is not

        # Setup optimization
        optim_opts = Optim.Options(
            iterations = get(config, :gpOptimIterations, 100),
            show_trace = get(config, :gpOptimShowTrace, false),
            f_tol = get(config, :gpOptimFTol, 1e-8),
            g_tol = get(config, :gpOptimGTol, 1e-8)
        )

        # Optimize for each dimension independently
        for dim in 1:n_dims
            if verbose_print println("  Optimizing dimension $dim...") end
            y_dim = y_obs[:, dim]

            # Initial guesses
            log_var_guess, log_len_guess, log_σ_guess = 0.0, 0.0, 0.0
            valid_y = filter(!isnan, y_dim)
            if !isempty(valid_y) && length(valid_y) > 1
                var_y = var(valid_y; corrected=true)
                data_range = maximum(valid_y) - minimum(valid_y)
                time_range = maximum(t_obs) - minimum(t_obs)
                mad_val = median(abs.(valid_y .- median(valid_y))) * 1.4826
                log_var_guess = log(max(var_y, 1e-4))
                log_len_guess = log(max(time_range / 10.0, 1e-2))
                log_σ_guess = log(max(mad_val, 1e-3 * data_range, 1e-4))
            else
                log_var_guess = log(1.0)
                log_len_guess = log(max((maximum(t_obs) - minimum(t_obs)) / 10.0, 1e-2))
                log_σ_guess = log(0.1)
            end

            # Use provided phi if available, otherwise use guess
            initial_log_params = if isempty(phi_exogenous)
                 [log_var_guess, log_len_guess, log_σ_guess]
             else
                 # Use provided phi, only guess sigma
                 [log(phi_exogenous[1, dim]), log(phi_exogenous[2, dim]), log_σ_guess]
            end
             if verbose_print println("    Initial guess [log(var), log(len), log(σ)]: ", round.(initial_log_params, digits=3)) end

            # Call optimization routine
            # Need to handle potential errors during optimization
             local optimized_params
             try
                optimized_params = Initialization.optimize_gp_hyperparameters(
                    y_dim, t_obs, kernel_type, initial_log_params;
                    jitter=jitter, optim_options=optim_opts
                )
             catch opt_err
                @error "GP Hyperparameter optimization failed for dimension $dim." exception=(opt_err, catch_backtrace())
                @warn "Using initial guess as optimization result for dimension $dim."
                optimized_params = exp.(initial_log_params) # Fallback to initial guess
             end

             if verbose_print println("    Optimized [var, len, σ]: ", round.(optimized_params, digits=4)) end

            # Store optimized values
            if isempty(phi_exogenous)
                ϕ_est[1, dim] = optimized_params[1] # Variance
                ϕ_est[2, dim] = optimized_params[2] # Lengthscale
            end
            # Always store the estimated sigma if sigma is not fixed
            if !sigma_is_fixed
                # Ensure estimated sigma is positive, fallback if needed
                σ_est[dim] = max(optimized_params[3], 1e-8)
            end
        end

        # Assign estimated values
        ϕ_all_dimensions = isempty(phi_exogenous) ? ϕ_est : phi_exogenous
        # sigma_init gets the estimated value if sigma is not fixed
        sigma_init = sigma_is_fixed ? sigma_exogenous : σ_est

        if verbose_print println("Initialization optimization complete.") end

    else
        # Option B: Use exogenously provided (fixed) ϕ and σ
        ϕ_all_dimensions = phi_exogenous
        sigma_init = sigma_exogenous # sigma_init stores the fixed value
        println("Using exogenously provided fixed ϕ and σ.")
    end

    println("Using ϕ (GP hyperparameters):\n", round.(ϕ_all_dimensions, digits=4))
    println("Initial σ (observation noise SD): ", round.(sigma_init, digits=4))
    if sigma_is_fixed println("(Sigma will remain fixed during sampling)") end


    # --- 3. Initialize Latent States (x) ---
    local x_init::Matrix{Float64}
    if isempty(x_init_exogenous)
        # Option A: Initialize x via linear interpolation
        println("Initializing latent states x via linear interpolation...")
        x_init = zeros(n_times, n_dims)
        for dim in 1:n_dims
            valid_indices = findall(!isnan, y_obs[:, dim])
            if isempty(valid_indices)
                x_init[:, dim] .= 0.0
                @warn "No observations found for dimension $dim. Initializing x with zeros."
                continue
            end
            valid_times = t_obs[valid_indices]
            valid_values = y_obs[valid_indices, dim]

            if length(valid_indices) < 2
                 @warn "Dimension $dim has fewer than 2 observations. Using constant extrapolation for initialization."
                 x_init[:, dim] .= valid_values[1]
                 continue
            end
            # Create interpolation object safely
             local itp
             try
                 # Ensure unique time points for interpolation
                 unique_indices = unique(i -> valid_times[i], 1:length(valid_times))
                 if length(unique_indices) < length(valid_times)
                     @warn "Duplicate time points found for interpolation in dim $dim. Using unique points."
                 end
                 unique_times = valid_times[unique_indices]
                 unique_values = valid_values[unique_indices]

                 if length(unique_times) < 2 # Check again after unique
                      @warn "Dimension $dim has fewer than 2 unique observation times. Using constant extrapolation."
                      x_init[:, dim] .= unique_values[1]
                      continue
                 end
                 # Sort points just in case findall didn't return sorted indices
                 sort_perm = sortperm(unique_times)
                 sorted_times = unique_times[sort_perm]
                 sorted_values = unique_values[sort_perm]

                 itp = linear_interpolation(sorted_times, sorted_values, extrapolation_bc=Line())
                 x_init[:, dim] .= itp.(t_obs) # Apply interpolation
             catch itp_err
                 @error "Linear interpolation failed for dimension $dim." exception=(itp_err, catch_backtrace())
                 @warn "Falling back to zero initialization for dimension $dim."
                 x_init[:, dim] .= 0.0
             end

        end
        println("Latent state initialization complete.")
    else
         # Option B: Use exogenously provided x_init
         if size(x_init_exogenous) != (n_times, n_dims)
             error("Provided :xInit matrix has wrong dimensions. Expected ($n_times, $n_dims), got $(size(x_init_exogenous)).")
         end
        x_init = x_init_exogenous
        println("Using exogenously provided initial latent states xInit.")
    end

    # --- 3. Initialize ODE Parameters (θ) ---
    local θ_init::Vector{Float64}
    if isempty(theta_init_exogenous)
        # Option A: Initialize θ based on parameter bounds
        println("Initializing ODE parameters θ based on bounds...")
        θ_init = zeros(n_params_ode)
        for i in 1:n_params_ode
            lb = ode_system.thetaLowerBound[i]
            ub = ode_system.thetaUpperBound[i]
            # Set initial guess
            if isfinite(lb) && isfinite(ub)
                θ_init[i] = (lb + ub) / 2.0
            elseif isfinite(lb)
                θ_init[i] = lb + abs(lb)*0.1 + 0.1
            elseif isfinite(ub)
                θ_init[i] = ub - abs(ub)*0.1 - 0.1
            else
                θ_init[i] = 0.0
            end
             # Nudge within bounds
             if isfinite(lb) && θ_init[i] <= lb
                 θ_init[i] = lb + 1e-4 * (isfinite(ub) ? min(1.0, ub-lb) : 1.0)
             end
             if isfinite(ub) && θ_init[i] >= ub
                 θ_init[i] = ub - 1e-4 * (isfinite(lb) ? min(1.0, ub-lb) : 1.0)
             end
             θ_init[i] = clamp(θ_init[i], lb, ub) # Final clamp
        end
    else
         # Option B: Use exogenously provided θ_init
         if length(theta_init_exogenous) != n_params_ode
             error("Provided :thetaInit vector has wrong length. Expected $n_params_ode, got $(length(theta_init_exogenous)).")
         end
        θ_init = theta_init_exogenous
        # Check bounds
        if any(θ_init .< ode_system.thetaLowerBound) || any(θ_init .> ode_system.thetaUpperBound)
             @warn "Provided :thetaInit contains values outside the specified bounds. Clamping initial values."
             θ_init = clamp.(θ_init, ode_system.thetaLowerBound, ode_system.thetaUpperBound)
        end
         println("Using exogenously provided initial ODE parameters θInit.")
    end
    println("Initial θ: ", round.(θ_init, digits=4))


    # --- 4. Calculate GPCov Structs ---
    @info "Calculating GP Covariance structures..."
    cov_all_dimensions = Vector{GPCov}(undef, n_dims)
    actual_band_size = min(band_size, n_times - 1)
     if actual_band_size < 0; actual_band_size = 0; end
    if verbose_print println("Using Band Size: $actual_band_size (Requested: $band_size)") end

    for dim in 1:n_dims
        cov_all_dimensions[dim] = GPCov()
        local kernel
        ϕ_dim = ϕ_all_dimensions[:, dim]
        var = ϕ_dim[1]; len = ϕ_dim[2]
        # Check for invalid phi values
        if !isfinite(var) || var <= 0 || !isfinite(len) || len <= 0
            @error "Invalid GP hyperparameters for dimension $dim: variance=$var, lengthscale=$len. Check initialization or provided :phi."
            return nothing # Critical error
        end
        if kernel_type == "matern52"
            kernel = Kernels.create_matern52_kernel(var, len)
        elseif kernel_type == "rbf"
            kernel = Kernels.create_rbf_kernel(var, len)
        else
            @warn "Unsupported kernel type '$kernel_type'. Defaulting to matern52."
            kernel = Kernels.create_matern52_kernel(var, len)
        end
        try
             calculate_gp_covariances!(
                 cov_all_dimensions[dim], kernel, ϕ_dim, t_obs, actual_band_size;
                 complexity=2, jitter=jitter
             )
        catch e
             @error "Failed to calculate GP covariances for dimension $dim." phi=ϕ_dim kernel=kernel bandsize=actual_band_size jitter=jitter exception=(e, catch_backtrace())
             # Decide if this is recoverable or should stop
              return nothing # Stop execution if GP setup fails crucially
         end
    end
    @info "GP Covariance calculation complete."


    # --- 5. Define Log Posterior Target ---
    # Use the *initial* sigma value (sigma_init) for the MagiTarget struct.
    # The sampler will use the sampled sigma values internally via the logdensity functions.
    if !(typeof(prior_temperature) <: AbstractVector && length(prior_temperature) == 3)
        @warn "priorTemperature β should be a vector of 3 floats [β_deriv, β_level, β_obs]. Using default [1.0, 1.0, 1.0] or first element if scalar."
        scalar_temp = first(prior_temperature)
        prior_temps = fill(Float64(scalar_temp), 3)
    else
        prior_temps = convert(Vector{Float64}, prior_temperature)
    end
    if verbose_print println("Using Prior Temperatures β [Deriv, Level, Obs]: ", prior_temps) end

    # Pass the *initial* sigma_init to the target struct constructor
    target = MagiTarget(
        y_obs,
        cov_all_dimensions,
        ode_system.fOde,
        ode_system.fOdeDx,
        ode_system.fOdeDtheta,
        sigma_init,          # <<< Pass the initial sigma here
        prior_temps,
        n_times,
        n_dims,
        n_params_ode,
        sigma_is_fixed
    )
    @info "MagiTarget for LogDensityProblems created."


    # --- 6. Initialize Full Parameter Vector for Sampler ---
    # This vector structure depends on whether sigma is fixed or sampled
    local params_init::Vector{Float64}

    if initial_params === nothing
        # Construct initial parameters based on individual initializations
        log_sigma_init = log.(max.(sigma_init, 1e-8)) # Transform to log-scale, ensure positive argument

        if sigma_is_fixed
            # Sampler only sees x and theta
            params_init = vcat(vec(x_init), θ_init)
            println("Initializing sampler with x and theta only (sigma fixed).")
        else
            # Sampler sees x, theta, and log_sigma
            params_init = vcat(vec(x_init), θ_init, log_sigma_init)
            println("Initializing sampler with x, theta, and log_sigma.")
        end
    else
         # Use the exogenously provided full parameter vector
         expected_len = n_times * n_dims + n_params_ode + (sigma_is_fixed ? 0 : n_dims)
         if length(initial_params) != expected_len
             error("Provided initial_params vector has wrong length. Expected $expected_len, got $(length(initial_params)). Check if sigma should be included.")
         end
        params_init = initial_params

        # Check bounds for the θ part
        theta_start_idx = n_times * n_dims + 1
        theta_end_idx = n_times * n_dims + n_params_ode
        θ_part_init = @view initial_params[theta_start_idx:theta_end_idx]
         if any(θ_part_init .< ode_system.thetaLowerBound) || any(θ_part_init .> ode_system.thetaUpperBound)
             @warn "θ part of provided initial_params contains values outside bounds. Clamping initial values."
             # Create copy before modifying if initial_params should remain unchanged
             params_init = copy(initial_params)
             params_init[theta_start_idx:theta_end_idx] = clamp.(θ_part_init, ode_system.thetaLowerBound, ode_system.thetaUpperBound)
         end

         # Check positivity for the sigma part if it's included and sampled
         if !sigma_is_fixed
             log_sigma_part_init = @view initial_params[(theta_end_idx + 1):end]
             # Initial sigma values derived from this log_sigma should be positive
             if any(!isfinite, log_sigma_part_init) || any(s -> s <= 0, exp.(log_sigma_part_init))
                 @warn "log_sigma part of provided initial_params implies non-positive sigma values. Sampler might struggle."
             end
         end
         println("Using exogenously provided full initial parameter vector.")
    end

    # Calculate total dimension being sampled
    total_params_dim_sampling = length(params_init)
    println("Total number of parameters being sampled by HMC: $total_params_dim_sampling")


    # --- 7. Run Sampler ---
    @info "Setting up and running NUTS sampler..."
    n_adapts = Int(floor(niter_hmc * burnin_ratio))
    n_samples_total = niter_hmc
    n_samples_keep = n_samples_total - n_adapts
    initial_step_size = step_size_factor

    local chain = nothing
    local stats = nothing

    try
        # Run the NUTS sampler
        # The target object (via LogDensityProblems interface) now handles the correct dimensions
        # Ensure the interface implementation reflects the sampled parameters
        chain, stats = run_nuts_sampler(
            target,
            params_init;
            n_samples = n_samples_total,
            n_adapts = n_adapts,
            target_accept_ratio = target_accept_ratio,
            initial_step_size = initial_step_size
        )
    catch sampler_err
        @error "Error occurred during run_nuts_sampler call." exception=(sampler_err, catch_backtrace())
        chain = nothing
        stats = nothing
        # Depending on severity, may want to return nothing here
        # return nothing
    end

    # --- Debugging Output (Optional) ---
    if verbose_print
        println("--- solve_magi DEBUG: Sampler returned. ---")
        println("--- solve_magi DEBUG: Type of chain: $(typeof(chain))")
        if chain !== nothing
            println("--- solve_magi DEBUG: Chain Is empty: $(isempty(chain)), Length: $(length(chain)) ---")
            if !isempty(chain)
                println("--- solve_magi DEBUG: Type of first element: $(typeof(first(chain)))")
            end
        end
        println("--- solve_magi DEBUG: Type of stats: $(typeof(stats))")
         if stats !== nothing
            println("--- solve_magi DEBUG: Stats Is empty: $(isempty(stats)), Length: $(try length(stats) catch; -1 end) ---")
            if !isempty(stats)
                println("--- solve_magi DEBUG: Type of first stat: $(typeof(first(stats)))")
            end
         end
    end

    # Check if sampling was successful
    if chain === nothing || isempty(chain)
        @error "Sampling failed or returned no samples. Check sampler logs or errors above."
        return nothing
    end
    @info "Sampling completed."


    # --- 8. Process Results ---
    @info "Processing MCMC samples..."
    if verbose_print println("--- solve_magi DEBUG: Entering results processing block. ---") end

    local samples_post_burnin_matrix::Matrix{Float64}
    local x_samples::Array{Float64, 3}
    local θ_samples::Matrix{Float64}
    local sigma_samples::Matrix{Float64} # Changed type
    local lp_values::Vector{Float64}

    try
        # Ensure chain format
        if !(chain isa Vector) || (length(chain)>0 && !(first(chain) isa AbstractVector))
             error("Sampler output 'chain' is not a Vector of Vectors as expected. Type: $(typeof(chain))")
        end
        if size(first(chain), 1) != total_params_dim_sampling
            error("Sampler output chain element dimension ($(size(first(chain), 1))) does not match expected sampling dimension ($total_params_dim_sampling).")
        end

        # Convert vector of sample vectors into a matrix (dims × samples)
        samples_post_burnin_matrix = hcat(chain...)
        if verbose_print println("--- solve_magi DEBUG: samples_post_burnin_matrix size: $(size(samples_post_burnin_matrix)) ---") end

        n_samples_post_burnin = size(samples_post_burnin_matrix, 2)

        # Verify sample count
        if n_samples_post_burnin != n_samples_keep
             @warn "Number of samples after warmup ($(n_samples_post_burnin)) does not match expected ($(n_samples_keep)). Check sampler's drop_warmup setting or output."
             if n_samples_post_burnin == 0
                 error("Sampler returned no post-burn-in samples after hcat.")
             end
        end

        # Define indices based on whether sigma was sampled
        x_indices = 1:(n_times * n_dims)
        theta_indices = (n_times * n_dims + 1):(n_times * n_dims + n_params_ode)
        if !sigma_is_fixed
             log_sigma_indices = (n_times * n_dims + n_params_ode + 1):total_params_dim_sampling
             if length(log_sigma_indices) != n_dims
                error("Indexing error: Number of log_sigma indices ($(length(log_sigma_indices))) does not match n_dims ($n_dims).")
             end
        end

        # Extract and reshape x samples
        x_samples_flat = samples_post_burnin_matrix[x_indices, :]
        x_samples = Array{Float64, 3}(undef, n_samples_post_burnin, n_times, n_dims)
        # Check dimensions before reshaping loop
        @assert size(x_samples_flat) == (n_times * n_dims, n_samples_post_burnin) "Dimension mismatch for x_samples_flat"
        for i in 1:n_samples_post_burnin
            # Use view for efficiency
            x_samples[i, :, :] = reshape(view(x_samples_flat, :, i), n_times, n_dims)
        end
        if verbose_print println("--- solve_magi DEBUG: Reshaped x_samples successfully.") end

        # Extract θ samples
        θ_samples_flat = samples_post_burnin_matrix[theta_indices, :]
        @assert size(θ_samples_flat) == (n_params_ode, n_samples_post_burnin) "Dimension mismatch for θ_samples_flat"
        θ_samples = Matrix(θ_samples_flat') # Transpose to get (n_samples × n_params)
        if verbose_print println("--- solve_magi DEBUG: Created θ_samples matrix size: $(size(θ_samples)) ---") end

        # Extract sigma samples (if sampled) or use fixed value
        if sigma_is_fixed
            # Repeat the fixed initial sigma for all "samples"
            sigma_samples = repeat(reshape(sigma_init, 1, n_dims), n_samples_post_burnin, 1)
            if verbose_print println("--- solve_magi DEBUG: Using fixed sigma values.") end
        else
            # Extract log_sigma samples and transform back
            log_sigma_samples_flat = samples_post_burnin_matrix[log_sigma_indices, :]
            @assert size(log_sigma_samples_flat) == (n_dims, n_samples_post_burnin) "Dimension mismatch for log_sigma_samples_flat"
            sigma_samples = exp.(Matrix(log_sigma_samples_flat')) # Transpose first, then exp. -> (n_samples × n_dims)
            if verbose_print println("--- solve_magi DEBUG: Created sigma_samples matrix size: $(size(sigma_samples)) ---") end
        end

        # Extract log-posterior values from stats (if available)
        lp_values = Float64[] # Initialize as empty
        if verbose_print println("--- solve_magi DEBUG: Attempting to extract lp_values... ---") end
        if stats !== nothing && stats isa Vector && !isempty(stats) && length(stats) == n_samples_post_burnin
             first_stat = first(stats)
             if verbose_print println("--- solve_magi DEBUG: stats[1] type: $(typeof(first_stat)), fields: $(try fieldnames(typeof(first_stat)) catch; "N/A" end) ---") end
             # Check fields robustly
             lp_field = :lp
             if !hasproperty(first_stat, lp_field)
                 if hasproperty(first_stat, :log_density)
                     lp_field = :log_density
                 else
                     lp_field = :not_found # Indicate field not found
                 end
             end

             if lp_field != :not_found
                 try
                     lp_values = [getproperty(s, lp_field) for s in stats]
                     if verbose_print println("--- solve_magi DEBUG: Extracted lp_values using :$lp_field field. Length: $(length(lp_values)) ---") end
                 catch lp_err
                     @warn "Error extracting log posterior values using field '$lp_field'." exception=(lp_err, catch_backtrace())
                     lp_values = Float64[] # Reset if extraction failed
                 end
             else
                  @warn "Sampler statistics vector does not contain standard :lp or :log_density field."
                  if verbose_print println("--- solve_magi DEBUG: Could not find :lp or :log_density in stats objects.") end
             end
        else
             stats_type = typeof(stats); stats_len = try length(stats) catch; -1 end
             @warn "Log posterior values (lp) could not be extracted. Stats structure unexpected or length mismatch. Type: $stats_type, Length: $stats_len, Expected: Vector of length $n_samples_post_burnin."
             if verbose_print println("--- solve_magi DEBUG: Stats structure unexpected or mismatch, cannot extract lp_values.") end
        end
        # Ensure lp_values is always a Vector{Float64}, even if empty
        if !isa(lp_values, Vector{Float64}) || length(lp_values) != n_samples_post_burnin
             if isempty(lp_values) && n_samples_post_burnin > 0 # Only warn if samples exist but lp extraction failed
                 @warn "Log posterior values (lp) could not be extracted. Returning empty vector."
             end
             lp_values = Float64[] # Ensure it's an empty Float64 vector if extraction failed or resulted in wrong type/length
        end


        @info "Finished processing results."

    catch process_err
        @error "Error during results processing block!" exception=(process_err, catch_backtrace())
        return nothing
    end

    # --- Final Debug Print (Optional) ---
    if verbose_print println("--- solve_magi DEBUG: Preparing to return NamedTuple... ---") end

    # Ensure essential variables are defined
    if !@isdefined(θ_samples) || !@isdefined(x_samples) || !@isdefined(sigma_samples) || !@isdefined(lp_values)
         @error "Results processing did not complete successfully (essential variables not defined)."
         return nothing
    end

    # Return results as a NamedTuple
    return (
        theta = θ_samples,         # Posterior samples for θ (n_samples × k)
        x_sampled = x_samples,     # Posterior samples for x(I) (n_samples × n × D)
        sigma = sigma_samples,     # Samples (n_samples × D) or Fixed repeated (n_samples × D)
        phi = ϕ_all_dimensions,    # Used/estimated GP params ϕ (2 × D)
        lp = lp_values             # Log-posterior values (n_samples), empty if unavailable
    )

end # end solve_magi


# =========================================================================
# Postprocessing Functions - UPDATED for sampled sigma
# =========================================================================
"""
    results_to_chain(results::NamedTuple; par_names=nothing, include_sigma=false, include_lp=false)

Convert the MCMC samples from `solve_magi` results into an `MCMCChains.Chains` object.
Requires the `MCMCChains` package to be loaded.

# Arguments
- `results::NamedTuple`: The output from `solve_magi`.
- `par_names::Union{Vector{String}, Nothing}`: Optional names for the θ parameters. Length must match `n_params_ode`.
- `include_sigma::Bool`: If true, include the σ samples (or fixed value) as parameters in the chain. Default: `false`.
- `include_lp::Bool`: If true, include the log-posterior values as a parameter in the chain. Default: `false`.

# Returns
- `MCMCChains.Chains`: An MCMCChains object containing the selected samples.
"""
function results_to_chain(results::NamedTuple; par_names=nothing, include_sigma=false, include_lp=false)
    if !@isloaded(MCMCChains)
         error("MCMCChains package is required for results_to_chain function. Please install and load it (`using MCMCChains`).")
    end

    # Access MCMCChains functions using getfield
    Chains = getfield(Main, :MCMCChains).Chains

    θ_samples = results.theta # Samples are (n_samples × n_params_ode)
    n_samples, n_params_ode = size(θ_samples)

    # Determine number of dimensions (needed for sigma)
    n_dims = size(results.sigma, 2)

    # --- Assemble Parameter Names ---
    _par_names_vec = String[]
    # Start with θ names
    if par_names === nothing
        θ_names = ["theta[$i]" for i in 1:n_params_ode]
    else
        if length(par_names) != n_params_ode
            error("Length of provided par_names ($(length(par_names))) does not match number of θ parameters ($n_params_ode).")
        end
        θ_names = copy(par_names)
    end
    append!(_par_names_vec, θ_names)

    # Add σ names if requested
    if include_sigma
        σ_names = ["sigma[$i]" for i in 1:n_dims]
        append!(_par_names_vec, σ_names)
    end

    # Add lp name if requested
    if include_lp && haskey(results, :lp) && !isempty(results.lp)
        if length(results.lp) == n_samples
            push!(_par_names_vec, "lp")
        end
        # Warning for length mismatch is handled below
    end

    # --- Assemble Data Matrix ---
    data_matrices = Any[θ_samples] # Start with theta samples

    # Add σ samples if requested
    if include_sigma
        sigma_data = results.sigma
        if size(sigma_data, 1) != n_samples
             @warn "Sigma data in results has incorrect number of rows ($(size(sigma_data, 1)) vs $n_samples). Cannot include sigma reliably."
             # Remove sigma names if data is bad
             _par_names_vec = filter(x -> !startswith(x, "sigma["), _par_names_vec)
        else
            push!(data_matrices, sigma_data) # Append sigma samples/values
        end
    end

    # Add log-posterior (lp) samples if requested and available
    lp_added = false
    if include_lp
        if haskey(results, :lp) && !isempty(results.lp)
            lp_samples = results.lp
            if length(lp_samples) == n_samples
                # Reshape lp to be a column vector for hcat
                push!(data_matrices, reshape(lp_samples, n_samples, 1))
                lp_added = true
            else
                @warn "Length of log-posterior (:lp) samples ($(length(lp_samples))) does not match number of θ samples ($n_samples). Cannot include lp in chain."
                # Remove lp name if data is bad
                _par_names_vec = filter(x -> x != "lp", _par_names_vec)
            end
        else
             @warn "Log-posterior (:lp) key not found or vector is empty in results. Cannot include lp in chain."
             # Remove lp name if requested but not found
             _par_names_vec = filter(x -> x != "lp", _par_names_vec)
        end
    end

    # Concatenate selected data matrices horizontally
    data_matrix = hcat(data_matrices...)

    # Verify dimensions match names
    if size(data_matrix, 2) != length(_par_names_vec)
        @error "Mismatch between data matrix columns ($(size(data_matrix, 2))) and parameter names ($(length(_par_names_vec))). Check include_sigma/include_lp flags and results content."
        # Attempt to proceed with available names/data, may lead to errors in MCMCChains
        min_len = min(size(data_matrix, 2), length(_par_names_vec))
        data_matrix = data_matrix[:, 1:min_len]
        _par_names_vec = _par_names_vec[1:min_len]
    end


    # Convert final list of string names to Symbols for MCMCChains
    _par_names_symbols = Symbol.(_par_names_vec)

    # Create the Chains object
    # MCMCChains expects data in [iterations, parameters, chains]
    # Assuming a single chain from our sampler run
    chains_data = reshape(data_matrix, n_samples, size(data_matrix, 2), 1)

    # Check for NaNs or Infs in data before creating chain
    if !all(isfinite, chains_data)
        @warn "Non-finite values detected in data passed to MCMCChains. Summary/plotting might fail."
        # Optionally replace non-finite values
        # chains_data[.!isfinite.(chains_data)] .= NaN # Or some other placeholder
    end

    chn = Chains(chains_data, _par_names_symbols) # Use Symbols for names

    return chn
end


"""
    magi_summary(results::NamedTuple; par_names=nothing, include_sigma=false, digits=3, lower=0.025, upper=0.975)

Compute and print summary statistics for MCMC samples.
Requires `MCMCChains`. Falls back to basic stats if unavailable.

# Arguments
- `results::NamedTuple`: Output from `solve_magi`.
- `par_names::Union{Vector{String}, Nothing}`: Optional names for θ parameters.
- `include_sigma::Bool`: Include σ samples in the summary. Default: `false`.
- `digits::Int`: Significant digits for display. Default: 3.
- `lower::Float64`, `upper::Float64`: Quantiles for credible interval. Defaults: 0.025, 0.975.

# Returns
- `NamedTuple` containing `summarystats` and `quantiles` DataFrames (if `MCMCChains` is loaded), otherwise `nothing`.
"""
function magi_summary(results::NamedTuple; par_names=nothing, include_sigma=false, digits=3, lower=0.025, upper=0.975)
    println("--- MAGI Posterior Summary ---")

    # Basic fallback stats calculation function
    function print_basic_stats(label, samples, digits)
        if isempty(samples)
            println("$label: No samples available.")
            return
        end
        means = mean(samples, dims=1)
        medians = median(samples, dims=1)
        println("$label Mean:   ", round.(vec(means); digits=digits))
        println("$label Median: ", round.(vec(medians); digits=digits))
    end

    if !@isloaded(MCMCChains)
        # Fallback if MCMCChains is not loaded
        @warn "MCMCChains not available. Printing basic mean/median."
        print_basic_stats("Theta (θ)", results.theta, digits)
        if include_sigma
            # Check if sigma was sampled or fixed
            if size(results.sigma, 1) > 1
                print_basic_stats("Sigma (σ)", results.sigma, digits)
            else
                println("Sigma (σ) Fixed: ", round.(vec(results.sigma); digits=digits))
            end
        end
        return nothing
    end

    # Use MCMCChains for a comprehensive summary
    summarystats_func = getfield(Main, :MCMCChains).summarystats
    quantile_func = getfield(Main, :MCMCChains).quantile

    try
        # Create chain including sigma/lp if requested
        chain = results_to_chain(results; par_names=par_names, include_sigma=include_sigma, include_lp=true)

        # Calculate summary stats and quantiles
        stats = summarystats_func(chain)
        quants = quantile_func(chain; q=[lower, 0.5, upper])

        # Print nicely formatted summary tables
        println(stats)
        println("\nQuantiles ($(lower*100)% / 50% / $(upper*100)%):")
        println(quants)

        # Return the summary objects
        return (summarystats = stats, quantiles = quants)
    catch e
        @error "Error generating MCMCChains summary:" exception=(e, catch_backtrace())
        println("Falling back to basic mean/median calculation.")
        # Duplicate fallback logic
        print_basic_stats("Theta (θ)", results.theta, digits)
        if include_sigma
            if size(results.sigma, 1) > 1
                print_basic_stats("Sigma (σ)", results.sigma, digits)
            else
                println("Sigma (σ) Fixed: ", round.(vec(results.sigma); digits=digits))
            end
        end
         return nothing
    end
end

"""
    plot_magi(
        results::NamedTuple;
        type="traj", par_names=nothing, comp_names=nothing, t_obs=nothing, y_obs=nothing,
        obs=true, ci=true, ci_col=:skyblue, lower=0.025, upper=0.975,
        include_sigma=false, include_lp=true, nplotcol=3, kwargs...
    )

Generate plots from `solve_magi` results. Requires `Plots` and `StatsPlots`.

# Arguments
- `results::NamedTuple`: Output from `solve_magi`.
- `type::String`: "traj" or "trace". Default: "traj".
- `par_names`: Names for θ parameters (for trace plot).
- `comp_names`: Names for state components x_d (for trajectory plot).
- `t_obs`: Time vector for x-axis.
- `y_obs`: Observation matrix to overlay.
- `obs::Bool`: Overlay observations? Default: `true`.
- `ci::Bool`: Show credible intervals? Default: `true`.
- `ci_col`: CI color. Default: `:skyblue`.
- `lower`, `upper`: CI quantiles. Defaults: 0.025, 0.975.
- `include_sigma`: Include σ in trace plots? Default: `false`.
- `include_lp`: Include log-posterior (lp) in trace plots? Default: `true`.
- `nplotcol`: Columns for subplot layout. Default: 3.
- `kwargs...`: Additional keyword arguments passed to `Plots.plot` / `StatsPlots.plot`.

# Returns
- A `Plots.Plot` object.
"""
function plot_magi(results::NamedTuple;
                   type="traj",                   # "traj" or "trace"
                   par_names=nothing,            # Names for θ
                   comp_names=nothing,           # Names for x components
                   t_obs=nothing,                # Time vector for x-axis
                   y_obs=nothing,                # Observations to overlay
                   obs=true,                     # Show observations?
                   ci=true,                      # Show credible intervals?
                   ci_col=:skyblue,              # CI color
                   lower=0.025,                  # Lower CI quantile
                   upper=0.975,                  # Upper CI quantile
                   include_sigma=false,          # Include σ in trace?
                   include_lp=true,              # Include lp in trace?
                   nplotcol=3,                   # Subplot columns
                   kwargs...)                    # Extra args for Plots.plot

    # Check if plotting packages are loaded
    if !@isloaded(Plots) || !@isloaded(StatsPlots)
         error("Plots and StatsPlots packages are required for plotting. Please install and load them (`using Plots, StatsPlots`).")
    end

    # Access plotting functions using getfield
    Plots = getfield(Main, :Plots)
    StatsPlots = getfield(Main, :StatsPlots)

    # Get dimensions from results
    n_samples, n_times, n_dims = size(results.x_sampled)

    if type == "traj"
        # --- Plot Inferred Trajectories ---
        x_sampled = results.x_sampled

        # Set component names
        _comp_names = if comp_names === nothing
            ["Component $i" for i in 1:n_dims]
        else
            if length(comp_names) != n_dims
                @warn "Length of comp_names ($(length(comp_names))) does not match number of dimensions ($n_dims). Using defaults."
                ["Component $i" for i in 1:n_dims]
            else
                comp_names
            end
        end

        # Determine plot layout
        plot_layout = (Int(ceil(n_dims / nplotcol)), nplotcol)
        plt = Plots.plot(layout=plot_layout, legend=false, titlefont=8; kwargs...)

        # Determine time vector for x-axis
        plot_times = t_obs === nothing ? (1:n_times) : t_obs
        if length(plot_times) != n_times
             @warn "Length of provided t_obs ($(length(t_obs))) does not match trajectory length ($n_times). Using 1:$n_times for x-axis."
             plot_times = 1:n_times
        end
        xlab = t_obs === nothing ? "Index" : "Time"

        # Plot each dimension
        for d in 1:n_dims
            x_dim_samples = view(x_sampled, :, :, d) # Use view

            # Calculate posterior mean and quantiles
            x_mean = vec(mean(x_dim_samples, dims=1))
            local x_lower, x_upper
            ci_enabled_dim = ci
            try
                 # Ensure samples are finite for quantile calculation
                 finite_samples = filter(isfinite, x_dim_samples)
                 if size(finite_samples, 1) < 2 # Check if enough finite samples per time point
                      error("Not enough finite samples for CI calculation in dim $d")
                 end
                 # Calculate quantiles using mapslices for efficiency
                 x_lower = vec(mapslices(x -> quantile(filter(isfinite, x), lower), x_dim_samples, dims=1))
                 x_upper = vec(mapslices(x -> quantile(filter(isfinite, x), upper), x_dim_samples, dims=1))
                 # Check if quantiles are valid
                 if any(!isfinite, x_lower) || any(!isfinite, x_upper)
                      error("Non-finite quantiles calculated for dim $d")
                 end
            catch e_quantile
                @warn "Could not calculate quantiles for trajectory plot CI for dim $d. CI disabled for this dimension." exception=(e_quantile, catch_backtrace())
                ci_enabled_dim = false
                x_lower = x_mean # Dummy values
                x_upper = x_mean
            end

            # Add subplot
            Plots.plot!(plt[d], plot_times, x_mean, linecolor=:blue, label="Mean",
                        xlabel=xlab, ylabel="Level", title=_comp_names[d])

            # Add credible interval ribbon
            if ci_enabled_dim
                Plots.plot!(plt[d], plot_times, ribbon=(x_mean .- x_lower, x_upper .- x_mean),
                            fillalpha=0.3, fillcolor=ci_col,
                            linealpha=0, label="$((upper-lower)*100)% CI")
            end

            # Overlay observations
            if obs
                if y_obs === nothing || t_obs === nothing
                    if d==1 @warn "Cannot plot observations because y_obs or t_obs was not provided to plot_magi." end # Warn only once
                else
                     try
                         if size(y_obs, 1) != n_times || size(y_obs, 2) != n_dims
                            if d==1 @warn "Dimensions of y_obs ($(size(y_obs))) do not match results dimensions ($n_times, $n_dims). Cannot plot observations." end
                         else
                            valid_idx = findall(i -> !isnan(y_obs[i, d]), 1:n_times)
                            if !isempty(valid_idx)
                                Plots.scatter!(plt[d], t_obs[valid_idx], y_obs[valid_idx, d],
                                               markercolor=:red, markersize=3, markerstrokewidth=0, label="Obs")
                            end
                         end
                     catch e_obs
                         if d==1 @error "Error plotting observations." exception=e_obs end
                     end
                end
            end
        end # end loop over dimensions
        # Adjust layout to make space for a legend if needed
        plot!(plt, legend = :outertopright) # Example legend position
        return plt

    elseif type == "trace"
        # --- Plot MCMC Traces ---
         if !@isloaded(MCMCChains) || !@isloaded(StatsPlots)
             error("MCMCChains and StatsPlots packages are required for trace plots.")
         end

        try
            # Create chain object using the updated results_to_chain
            chain = results_to_chain(results; par_names=par_names, include_sigma=include_sigma, include_lp=include_lp)

            # Use StatsPlots.plot on the Chains object
            return StatsPlots.plot(chain; kwargs...)
        catch e
            @error "Error generating MCMCChains trace plot:" exception=(e, catch_backtrace())
            return Plots.plot() # Return empty plot on error
        end
    else
        error("Invalid plot type specified: '$type'. Use type=\"traj\" or type=\"trace\".")
    end
end


end # End module MagiJl