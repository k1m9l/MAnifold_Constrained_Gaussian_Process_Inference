# src/MajiJl.jl

"""
    MagiJl

Provides the Julia implementation of the MAnifold-constrained Gaussian process Inference (MAGI)
method for Bayesian inference of Ordinary Differential Equation (ODE) parameters (θ) and
latent state trajectories (x(t)) from noisy, sparse, and potentially partially observed data.
... (rest of docstring) ...
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
include("likelihoods.jl")
include("logdensityproblems_interface.jl")
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
export fn_ode_dx!, fn_ode_dθ, hes1_ode_dx!, hes1_ode_dθ # Note: Using θ in export name

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
# solve_magi function definition
# =========================================================================
"""
    solve_magi(
        y_obs::Matrix{Float64},
        t_obs::Vector{Float64},
        ode_system::OdeSystem,
        config::Dict{Symbol,Any}=Dict{Symbol,Any}();
        initial_params=nothing
    )
... (rest of docstring) ...
"""
function solve_magi(
    y_obs::Matrix{Float64},       # Observations y(τ)
    t_obs::Vector{Float64},       # Time points t ∈ I (discretization grid)
    ode_system::OdeSystem,      # ODE definition (f, ∂f/∂x, ∂f/∂θ, bounds)
    config::Dict{Symbol,Any}=Dict{Symbol,Any}(); # Control dictionary
    initial_params=nothing      # Optional full starting vector [vec(x₀); θ₀]
)
    # --- Start of MAGI Workflow ---
    @info "Starting MAGI solver..."
    println("Input Data: $(size(y_obs, 1)) time points, $(size(y_obs, 2)) dimensions.")
    println("Time points from $(minimum(t_obs)) to $(maximum(t_obs)).")
    println("Using ODE System with $(ode_system.thetaSize) parameters.")
    println("Config: ", config) # Be careful if config contains large xInit

    # --- 1 & 2: Extract Dimensions & Configuration Settings ---
    # (Keep existing code)
    # ...
    n_times = length(t_obs)       # Number of time points n
    n_dims = size(y_obs, 2)       # Number of state dimensions D
    n_params_ode = ode_system.thetaSize # Number of ODE parameters k

    # Extract settings from config dictionary, using defaults if keys are absent
    kernel_type = get(config, :kernel, "matern52")
    niter_hmc = get(config, :niterHmc, 20000)       # Total HMC iterations
    burnin_ratio = get(config, :burninRatio, 0.5)   # Warmup proportion
    step_size_factor = get(config, :stepSizeFactor, 0.01) # Initial step size factor ϵ₀
    band_size = get(config, :bandSize, 20)          # Band matrix width
    prior_temperature = get(config, :priorTemperature, [1.0, 1.0, 1.0]) # Tempering β
    sigma_exogenous = get(config, :sigma, Float64[]) # Provided σ?
    phi_exogenous = get(config, :phi, Matrix{Float64}(undef, 0, 0)) # Provided ϕ?
    x_init_exogenous = get(config, :xInit, Matrix{Float64}(undef, 0, 0)) # Provided x₀?
    theta_init_exogenous = get(config, :thetaInit, Float64[]) # Provided θ₀?
    target_accept_ratio = get(config, :targetAcceptRatio, 0.8) # HMC adaptation target
    jitter = get(config, :jitter, 1e-6)             # Numerical stability factor


    # --- 3. Initialize GP Hyperparameters (ϕ) & Observation Noise (σ) ---
    # (Keep existing code)
    # ...
    local ϕ_all_dimensions::Matrix{Float64} # GP params [var; len] for each dim D
    local σ::Vector{Float64}             # Noise SD σ for each dim D

    if isempty(phi_exogenous) || isempty(sigma_exogenous)
        # Option A: Estimate ϕ and σ using GP marginal likelihood optimization
        println("Optimizing GP hyperparameters (ϕ and σ) using marginal likelihood...")
        ϕ_all_dimensions = zeros(2, n_dims) # Store variance and lengthscale
        σ = zeros(n_dims)
        optim_opts = Optim.Options(
            iterations = get(config, :gpOptimIterations, 100),
            show_trace = get(config, :gpOptimShowTrace, false),
            f_tol = get(config, :gpOptimFTol, 1e-8),
            g_tol = get(config, :gpOptimGTol, 1e-8)
        )

        # Optimize for each dimension independently
        for dim in 1:n_dims
            println("  Optimizing dimension $dim...")
            y_dim = y_obs[:, dim]
            # Heuristic initial guesses based on data properties
            log_var_guess, log_len_guess, log_σ_guess = 0.0, 0.0, 0.0
            valid_y = filter(!isnan, y_dim)
            if !isempty(valid_y) && length(valid_y) > 1
                var_y = var(valid_y; corrected=true)
                data_range = maximum(valid_y) - minimum(valid_y)
                time_range = maximum(t_obs) - minimum(t_obs)
                # Median Absolute Deviation (MAD) as robust noise estimate
                mad_val = median(abs.(valid_y .- median(valid_y))) * 1.4826 # Scale factor for normality
                log_var_guess = log(max(var_y, 1e-4)) # Log signal variance guess
                log_len_guess = log(max(time_range / 10.0, 1e-2)) # Log lengthscale guess
                log_σ_guess = log(max(mad_val, 1e-3 * data_range, 1e-4)) # Log noise SD guess
            else # Fallback if no/few observations
                log_var_guess = log(1.0)
                log_len_guess = log(max((maximum(t_obs) - minimum(t_obs)) / 10.0, 1e-2))
                log_σ_guess = log(0.1)
            end
            initial_log_params = [log_var_guess, log_len_guess, log_σ_guess]
            println("    Initial guess [log(var), log(len), log(σ)]: ", round.(initial_log_params, digits=3))

            # Call optimization routine from Initialization module
            optimized_params = Initialization.optimize_gp_hyperparameters(
                y_dim, t_obs, kernel_type, initial_log_params;
                jitter=jitter, optim_options=optim_opts
            )
            println("    Optimized [var, len, σ]: ", round.(optimized_params, digits=4))

            # Store optimized values
            ϕ_all_dimensions[1, dim] = optimized_params[1] # Variance
            ϕ_all_dimensions[2, dim] = optimized_params[2] # Lengthscale
            σ[dim] = optimized_params[3]                   # Noise SD
        end
        println("Optimization complete.")

    elseif isempty(phi_exogenous) && !isempty(sigma_exogenous)
         error("If providing :sigma exogenously, must also provide :phi.")
    elseif !isempty(phi_exogenous) && isempty(sigma_exogenous)
         error("If providing :phi exogenously, must also provide :sigma.")
    else
         # Option B: Use exogenously provided ϕ and σ
         if size(phi_exogenous) != (2, n_dims)
             error("Provided :phi matrix has wrong dimensions. Expected (2, $n_dims), got $(size(phi_exogenous)).")
         end
         if length(sigma_exogenous) != n_dims
             error("Provided :sigma vector has wrong length. Expected $n_dims, got $(length(sigma_exogenous)).")
         end
         ϕ_all_dimensions = phi_exogenous
         σ = sigma_exogenous
         println("Using exogenously provided ϕ and σ.")
     end
    println("Using ϕ (GP hyperparameters):\n", round.(ϕ_all_dimensions, digits=4))
    println("Using σ (observation noise SD): ", round.(σ, digits=4))


    # --- 3. Initialize Latent States (x) ---
    local x_init::Matrix{Float64} # Initial trajectory x(I)₀
    if isempty(x_init_exogenous)
        # Option A: Initialize x via linear interpolation
        println("Initializing latent states x via linear interpolation...")
        x_init = zeros(n_times, n_dims)
        for dim in 1:n_dims
            valid_indices = findall(!isnan, y_obs[:, dim]) # Find observed points for this dim
            if isempty(valid_indices)
                # Handle completely unobserved components
                x_init[:, dim] .= 0.0 # Simple default: initialize with zeros
                @warn "No observations found for dimension $dim. Initializing x with zeros."
                continue
            end
            # Interpolate based on observed times and values
            valid_times = t_obs[valid_indices]
            valid_values = y_obs[valid_indices, dim]

            # Check if enough points for interpolation
            if length(valid_indices) < 2
                 @warn "Dimension $dim has fewer than 2 observations. Using constant extrapolation for initialization."
                 # Use the single observed value for all time points
                 x_init[:, dim] .= valid_values[1]
                 continue
            end

            # Use Interpolations.jl for robust linear interpolation and extrapolation
            # Ensure valid_times are strictly increasing if required by Interpolations.jl
            # (May need sorting and handling duplicate times if they exist)
            # Example assumes valid_times are sorted and unique after findall
            itp = LinearInterpolation(valid_times, valid_values, extrapolation_bc=Line()) # Use Line extrapolation
            x_init[:, dim] .= itp.(t_obs) # Apply interpolation to all t_obs points

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
    # (Keep existing code)
    # ...
    local θ_init::Vector{Float64} # Initial ODE parameters θ₀
    if isempty(theta_init_exogenous)
        # Option A: Initialize θ based on parameter bounds
        println("Initializing ODE parameters θ based on bounds...")
        θ_init = zeros(n_params_ode)
        for i in 1:n_params_ode
            lb = ode_system.thetaLowerBound[i]
            ub = ode_system.thetaUpperBound[i]
            # Set initial guess, preferring midpoint if bounds are finite
            if isfinite(lb) && isfinite(ub)
                θ_init[i] = (lb + ub) / 2.0
            elseif isfinite(lb)
                θ_init[i] = lb + abs(lb)*0.1 + 0.1 # Guess slightly above lower bound
            elseif isfinite(ub)
                θ_init[i] = ub - abs(ub)*0.1 - 0.1 # Guess slightly below upper bound
            else
                θ_init[i] = 0.0 # Default guess for unbounded parameters
            end
             # Ensure guess is strictly within bounds, nudging if necessary
             if isfinite(lb) && θ_init[i] <= lb
                 θ_init[i] = lb + 1e-4 * (isfinite(ub) ? min(1.0, ub-lb) : 1.0) # Small nudge
             end
             if isfinite(ub) && θ_init[i] >= ub
                 θ_init[i] = ub - 1e-4 * (isfinite(lb) ? min(1.0, ub-lb) : 1.0) # Small nudge
             end
             # Final clamp for safety, although nudging should prevent hitting bounds
             θ_init[i] = clamp(θ_init[i], lb, ub)
        end
    else
         # Option B: Use exogenously provided θ_init
         if length(theta_init_exogenous) != n_params_ode
             error("Provided :thetaInit vector has wrong length. Expected $n_params_ode, got $(length(theta_init_exogenous)).")
         end
        θ_init = theta_init_exogenous
        # Check if provided θ₀ respects bounds
        if any(θ_init .< ode_system.thetaLowerBound) || any(θ_init .> ode_system.thetaUpperBound)
             @warn "Provided :thetaInit contains values outside the specified bounds. Clamping initial values."
             θ_init = clamp.(θ_init, ode_system.thetaLowerBound, ode_system.thetaUpperBound)
        end
         println("Using exogenously provided initial ODE parameters θInit.")
    end
    println("Initial θ: ", round.(θ_init, digits=4))


    # --- 4. Calculate GPCov Structs ---
    # (Keep existing code)
    # ...
    @info "Calculating GP Covariance structures..."
    cov_all_dimensions = Vector{GPCov}(undef, n_dims)
    actual_band_size = min(band_size, n_times - 1) # Ensure band size isn't too large
     if actual_band_size < 0
        actual_band_size = 0 # Handle n_times=1 case
    end
    println("Using Band Size: $actual_band_size (Requested: $band_size)")

    # Create kernel and calculate covariances for each dimension
    for dim in 1:n_dims
        cov_all_dimensions[dim] = GPCov() # Initialize empty GPCov struct
        local kernel # Ensure kernel is defined in this scope
        ϕ_dim = ϕ_all_dimensions[:, dim] # Get [var; len] for this dimension
        var = ϕ_dim[1]
        len = ϕ_dim[2]

        # Create the specified kernel using functions from Kernels module
        if kernel_type == "matern52"
            kernel = Kernels.create_matern52_kernel(var, len)
        elseif kernel_type == "rbf"
            kernel = Kernels.create_rbf_kernel(var, len)
        # Add other kernel types here if implemented
        else
            @warn "Unsupported kernel type '$kernel_type'. Defaulting to matern52."
            kernel = Kernels.create_matern52_kernel(var, len)
        end

        # Populate the GPCov struct with calculated matrices (dense & banded)
        try
             calculate_gp_covariances!(
                 cov_all_dimensions[dim], kernel, ϕ_dim, t_obs, actual_band_size;
                 complexity=2, # Calculate necessary derivatives for MAGI
                 jitter=jitter
             )
        catch e
             @error "Failed to calculate GP covariances for dimension $dim." phi=ϕ_dim kernel=kernel bandsize=actual_band_size jitter=jitter
             rethrow(e) # Stop execution if GP setup fails
         end
    end
    @info "GP Covariance calculation complete."


    # --- 5. Define Log Posterior Target ---
    # (Keep existing code)
    # ...
    if !(typeof(prior_temperature) <: AbstractVector && length(prior_temperature) == 3)
         @warn "priorTemperature β should be a vector of 3 floats [β_deriv, β_level, β_obs]. Using default [1.0, 1.0, 1.0] or first element if scalar."
         # Attempt to gracefully handle incorrect input format
         scalar_temp = first(prior_temperature) # Take first element if it's a scalar/vector
         prior_temps = fill(Float64(scalar_temp), 3)
    else
         prior_temps = convert(Vector{Float64}, prior_temperature) # Ensure Float64
    end
    println("Using Prior Temperatures β [Deriv, Level, Obs]: ", prior_temps)

    target = MagiTarget(
        y_obs,              # Observations y(τ)
        cov_all_dimensions, # Vector of GPCov structs
        ode_system.fOde,    # ODE function f
        ode_system.fOdeDx,  # ODE Jacobian ∂f/∂x
        ode_system.fOdeDtheta,# ODE Jacobian ∂f/∂θ
        σ,                  # Noise SDs σ
        prior_temps,        # Tempering factors β
        n_times,            # n
        n_dims,             # D
        n_params_ode        # k
    )
    @info "MagiTarget for LogDensityProblems created."


    # --- 6. Initialize Full Parameter Vector for Sampler ---
    # (Keep existing code)
    # ...
    local params_init::Vector{Float64} # Combined initial state [vec(x₀); θ₀]
    if initial_params === nothing
        # Concatenate the initialized x₀ and θ₀
        params_init = vcat(vec(x_init), θ_init)
    else
         # Use the exogenously provided full parameter vector
         if length(initial_params) != (n_times * n_dims + n_params_ode)
             error("Provided initial_params vector has wrong length. Expected $(n_times * n_dims + n_params_ode), got $(length(initial_params)).")
         end
        params_init = initial_params
        # Check bounds for the θ part of the provided vector
        θ_part_init = initial_params[(n_times * n_dims + 1):end]
         if any(θ_part_init .< ode_system.thetaLowerBound) || any(θ_part_init .> ode_system.thetaUpperBound)
             @warn "θ part of provided initial_params contains values outside bounds. Sampler might struggle or reject states."
         end
         println("Using exogenously provided full initial parameter vector.")
    end
    total_params_dim = length(params_init)
    println("Total number of parameters being sampled: $total_params_dim")


    # --- 7. Run Sampler ---
    # (Keep existing code)
    # ...
    @info "Setting up and running NUTS sampler..."
    n_adapts = Int(floor(niter_hmc * burnin_ratio)) # Number of warmup steps
    n_samples_total = niter_hmc                     # Total HMC iterations
    n_samples_keep = n_samples_total - n_adapts     # Number of samples after warmup
    initial_step_size = step_size_factor            # Initial step size ϵ₀

    # Initialize chain and stats outside try block
    local chain = nothing # To store the MCMC samples
    local stats = nothing # To store sampler statistics

    try
        # Run the NUTS sampler (defined in samplers.jl)
        chain, stats = run_nuts_sampler(
            target,              # The MagiTarget object defining logπ and ∇logπ
            params_init;         # The starting parameter vector [vec(x₀); θ₀]
            n_samples = n_samples_total, # Total iterations
            n_adapts = n_adapts,         # Warmup iterations
            target_accept_ratio = target_accept_ratio, # Target for MH step
            initial_step_size = initial_step_size      # Initial step size ϵ₀
        )
    catch sampler_err
        @error "Error occurred during run_nuts_sampler call." exception=(sampler_err, catch_backtrace())
        chain = nothing
        stats = nothing
        # Option: rethrow(sampler_err) # Stop execution immediately
    end

    # --- Debugging Output ---
    println("--- solve_magi DEBUG: Sampler returned. ---")
    println("--- solve_magi DEBUG: Type of chain: $(typeof(chain))")
    if chain !== nothing
        println("--- solve_magi DEBUG: Chain Is empty: $(isempty(chain)), Length: $(length(chain)) ---")
    end
    println("--- solve_magi DEBUG: Type of stats: $(typeof(stats))")
     if stats !== nothing
        println("--- solve_magi DEBUG: Stats Is empty: $(isempty(stats)), Length: $(try length(stats) catch; -1 end) ---")
     end
    # --- End Debugging Output ---


    # Check if sampling was successful
    if chain === nothing || isempty(chain)
        @error "Sampling failed or returned no samples. Check sampler logs or errors above."
        # Return nothing to indicate critical failure
        return nothing
    end
    @info "Sampling completed."


    # --- 8. Process Results ---
    # (Keep existing code)
    # ...
    @info "Processing MCMC samples..."
    println("--- solve_magi DEBUG: Entering results processing block. ---")

    # Declare variables for processed results
    local samples_post_burnin_matrix::Matrix{Float64}
    local x_samples::Array{Float64, 3}
    local θ_samples::Matrix{Float64}
    local lp_values::Vector{Float64}

    try
        # Ensure chain is a Vector of Vectors (standard output from AdvancedHMC)
        if !(chain isa Vector) || (length(chain)>0 && !(first(chain) isa AbstractVector))
             error("Sampler output 'chain' is not a Vector of Vectors as expected. Type: $(typeof(chain))")
        end

        # Convert vector of sample vectors into a matrix (dims × samples)
        samples_post_burnin_matrix = hcat(chain...)
        println("--- solve_magi DEBUG: samples_post_burnin_matrix size: $(size(samples_post_burnin_matrix)) ---")

        n_samples_post_burnin = size(samples_post_burnin_matrix, 2)

        # Verify number of samples returned matches expectation
        if n_samples_post_burnin != n_samples_keep
             @warn "Number of samples after warmup ($(n_samples_post_burnin)) does not match expected ($(n_samples_keep)). Check sampler's drop_warmup setting or output."
             if n_samples_post_burnin == 0
                 error("Sampler returned no post-burn-in samples after hcat.")
             end
        end

        # Define indices for x and θ within the flattened parameter vector
        x_indices = 1:(n_times * n_dims)
        θ_indices = (n_times * n_dims + 1):(total_params_dim)

        # Extract and reshape x samples
        x_samples_flat = samples_post_burnin_matrix[x_indices, :] # (n*D) × n_samples
        # Reshape into (n_samples × n_times × n_dims)
        x_samples = Array{Float64, 3}(undef, n_samples_post_burnin, n_times, n_dims)
        println("--- solve_magi DEBUG: Initialized x_samples array size: $(size(x_samples)) ---")
        for i in 1:n_samples_post_burnin
            # Reshape each sample vector and store it
            x_samples[i, :, :] = reshape(view(x_samples_flat, :, i), n_times, n_dims)
        end
        println("--- solve_magi DEBUG: Reshaped x_samples successfully. ---")

        # Extract and transpose θ samples to get (n_samples × n_params)
        θ_samples = Matrix(samples_post_burnin_matrix[θ_indices, :]')
        println("--- solve_magi DEBUG: Created θ_samples matrix size: $(size(θ_samples)) ---")

        # Extract log-posterior values from stats (if available)
        lp_values = Float64[] # Initialize as empty
        println("--- solve_magi DEBUG: Attempting to extract lp_values... ---")
        if stats !== nothing && stats isa Vector && !isempty(stats) && length(stats) == n_samples_post_burnin
             first_stat = first(stats) # Check the structure of the first stats object
             println("--- solve_magi DEBUG: stats[1] type: $(typeof(first_stat)), fields: $(try fieldnames(typeof(first_stat)) catch; "N/A" end) ---")
             # Check for common fields storing log-posterior
             if hasproperty(first_stat, :lp) # AdvancedHMC often uses :lp
                 lp_values = [s.lp for s in stats]
                 println("--- solve_magi DEBUG: Extracted lp_values using :lp field. Length: $(length(lp_values)) ---")
             elseif hasproperty(first_stat, :log_density) # Another possibility
                 lp_values = [s.log_density for s in stats]
                 println("--- solve_magi DEBUG: Extracted lp_values using :log_density field. Length: $(length(lp_values)) ---")
             else
                 @warn "Sampler statistics vector does not contain standard :lp or :log_density field."
                 println("--- solve_magi DEBUG: Could not find :lp or :log_density in stats objects.")
             end
        else
             stats_type = typeof(stats)
             stats_len = try length(stats) catch; -1 end
             @warn "Log posterior values (lp) could not be extracted. Stats structure unexpected or length mismatch. Type: $stats_type, Length: $stats_len, Expected: Vector of length $n_samples_post_burnin."
             println("--- solve_magi DEBUG: Stats structure unexpected or mismatch, cannot extract lp_values.")
        end
        @info "Finished processing results."

    catch process_err
        @error "Error during results processing block!" exception=(process_err, catch_backtrace())
        # Return nothing if processing fails
        return nothing
    end

    # --- Final Debug Print ---
    println("--- solve_magi DEBUG: Preparing to return NamedTuple... ---")
    # --- End Final Debug Print ---

    # Ensure all essential result variables were successfully assigned
    if !@isdefined(θ_samples) || !@isdefined(x_samples) || !@isdefined(lp_values)
         @error "Results processing did not complete successfully (θ_samples, x_samples, or lp_values not defined). Cannot return standard output."
         return nothing # Indicate failure
    end

    # Return results as a NamedTuple for easy access
    return (
        theta = θ_samples,         # Posterior samples for θ (n_samples × k)
        x_sampled = x_samples,     # Posterior samples for x(I) (n_samples × n × D)
        sigma = reshape(σ, 1, n_dims), # Used/estimated noise SD σ (1 × D)
        phi = ϕ_all_dimensions,    # Used/estimated GP params ϕ (2 × D)
        lp = lp_values             # Log-posterior values (n_samples)
    )

end # end solve_magi


# =========================================================================
# Postprocessing Functions
# =========================================================================
"""
    results_to_chain(results::NamedTuple; par_names=nothing, include_sigma=false, include_lp=false)

Convert the θ samples (and optionally σ, lp) from `solve_magi` results
into an `MCMCChains.Chains` object for use with the `MCMCChains.jl` ecosystem
(e.g., for diagnostics, plotting, summaries).

Requires the `MCMCChains` package to be loaded.

# Arguments
- `results::NamedTuple`: The output from `solve_magi`.
- `par_names::Union{Vector{String}, Nothing}`: Optional names for the θ parameters. Length must match `n_params_ode`.
- `include_sigma::Bool`: If true, include the fixed/estimated σ values as parameters in the chain. Default: `false`.
- `include_lp::Bool`: If true, include the log-posterior values as a parameter in the chain. Default: `false`.

# Returns
- `MCMCChains.Chains`: An MCMCChains object containing the selected samples.
"""
function results_to_chain(results::NamedTuple; par_names=nothing, include_sigma=false, include_lp=false)
    if !@isloaded(MCMCChains)
         error("MCMCChains package is required for results_to_chain function. Please install and load it (`using MCMCChains`).")
    end

    # Access MCMCChains functions using getfield (safer when conditionally loaded)
    Chains = getfield(Main, :MCMCChains).Chains

    θ_samples = results.theta # Samples are (n_samples × n_params_ode)
    n_samples, n_params_ode = size(θ_samples)

    # --- Assemble Parameter Names ---
    # Start with θ names
    if par_names === nothing
        # Default names: theta[1], theta[2], ... <<<--- FIX: Use ASCII
        _par_names_vec = ["theta[$i]" for i in 1:n_params_ode]
    else
        # Use provided names, checking length only for θ part
        if length(par_names) != n_params_ode
            error("Length of provided par_names ($(length(par_names))) does not match number of θ parameters ($n_params_ode).")
        end
        _par_names_vec = copy(par_names) # Use a copy
    end

    # --- Assemble Data Matrix ---
    # Start with θ samples
    data_matrix = θ_samples # (n_samples × n_params_ode)

    # Add σ samples if requested
    if include_sigma
        σ_values = vec(results.sigma) # σ is usually fixed/estimated once (1 × n_dims)
        n_dims = length(σ_values)
        # Repeat the single σ vector for all samples
        σ_matrix = repeat(reshape(σ_values, 1, n_dims), n_samples, 1) # (n_samples × n_dims)
        data_matrix = hcat(data_matrix, σ_matrix)
        # Add σ names <<<--- FIX: Use ASCII
        σ_names = ["sigma[$i]" for i in 1:n_dims]
        append!(_par_names_vec, σ_names)
    end

    # Add log-posterior (lp) samples if requested and available
    if include_lp && haskey(results, :lp) && !isempty(results.lp)
        lp_samples = results.lp
        if length(lp_samples) == n_samples
            # Add lp as the last column
            data_matrix = hcat(data_matrix, lp_samples)
            # Add lp name
            push!(_par_names_vec, "lp")
        else
            @warn "Length of log-posterior (:lp) samples ($(length(lp_samples))) does not match number of θ samples ($n_samples). Cannot include lp in chain."
        end
    elseif include_lp
         @warn "Log-posterior (:lp) key not found or vector is empty in results. Cannot include lp in chain."
    end

    # Convert final list of string names to Symbols for MCMCChains
    _par_names_symbols = Symbol.(_par_names_vec)

    # Create the Chains object
    # MCMCChains expects data in [iterations, parameters, chains]
    # Assuming a single chain from our sampler run
    chains_data = reshape(data_matrix, n_samples, size(data_matrix, 2), 1)
    chn = Chains(chains_data, _par_names_symbols) # Use Symbols for names

    return chn
end

# ... (magi_summary function) ...
"""
    magi_summary(results::NamedTuple; par_names=nothing, include_sigma=false, digits=3, lower=0.025, upper=0.975)

Compute and print summary statistics (mean, median, quantiles, standard deviation, etc.)
for the MCMC samples of θ (and optionally σ).

Requires the `MCMCChains` package to be loaded for a full summary table.
Falls back to basic mean/median if `MCMCChains` is not available.

# Arguments
- `results::NamedTuple`: The output from `solve_magi`.
- `par_names::Union{Vector{String}, Nothing}`: Optional names for the θ parameters.
- `include_sigma::Bool`: If true, include σ in the summary. Default: `false`.
- `digits::Int`: Number of significant digits to display. Default: 3.
- `lower::Float64`: Lower quantile for the credible interval. Default: 0.025.
- `upper::Float64`: Upper quantile for the credible interval. Default: 0.975.

# Returns
- `NamedTuple` containing `summarystats` and `quantiles` DataFrames (if `MCMCChains` is loaded).
- `nothing` otherwise.
"""
function magi_summary(results::NamedTuple; par_names=nothing, include_sigma=false, digits=3, lower=0.025, upper=0.975)
    println("--- MAGI Posterior Summary ---")

    if !@isloaded(MCMCChains)
        # Fallback if MCMCChains is not loaded
        @warn "MCMCChains not available. Printing basic mean/median."
        θ_mean = mean(results.theta, dims=1)
        θ_median = median(results.theta, dims=1)
        println("Theta (θ) Mean:   ", round.(vec(θ_mean); digits=digits))
        println("Theta (θ) Median: ", round.(vec(θ_median); digits=digits))
        if include_sigma
             println("Sigma (σ) Used:   ", round.(vec(results.sigma); digits=digits))
        end
        return nothing
    end

    # Use MCMCChains for a comprehensive summary
    # Access MCMCChains functions using getfield
    summarystats_func = getfield(Main, :MCMCChains).summarystats
    quantile_func = getfield(Main, :MCMCChains).quantile

    try
        # Create chain including sigma if requested (lp is included for potential diagnostics)
        chain = results_to_chain(results; par_names=par_names, include_sigma=include_sigma, include_lp=true)

        # Calculate summary stats and quantiles using MCMCChains functions
        stats = summarystats_func(chain)
        quants = quantile_func(chain; q=[lower, 0.5, upper]) # Get lower, median, upper quantiles

        # Print nicely formatted summary tables
        println(stats)
        println("\nQuantiles ($(lower*100)% / 50% / $(upper*100)%):")
        println(quants)

        # Return the summary objects
        return (summarystats = stats, quantiles = quants)
    catch e
        @error "Error generating MCMCChains summary:" exception=(e, catch_backtrace())
        println("Falling back to basic mean/median calculation.")
        # Duplicate fallback logic in case of error during MCMCChains processing
        θ_mean = mean(results.theta, dims=1)
        θ_median = median(results.theta, dims=1)
        println("Theta (θ) Mean:   ", round.(vec(θ_mean); digits=digits))
        println("Theta (θ) Median: ", round.(vec(θ_median); digits=digits))
        if include_sigma
             println("Sigma (σ) Used:   ", round.(vec(results.sigma); digits=digits))
        end
         return nothing
    end
end

# ... (plot_magi function) ...
"""
    plot_magi(
        results::NamedTuple;
        type="traj", par_names=nothing, comp_names=nothing, t_obs=nothing, y_obs=nothing,
        obs=true, ci=true, ci_col=:skyblue, lower=0.025, upper=0.975,
        include_sigma=false, include_lp=true, nplotcol=3, kwargs...
    )

Generate plots from `solve_magi` results. Requires `Plots` and `StatsPlots` packages.

Two plot types are supported:
- `type="traj"` (default): Plots the inferred mean trajectories x(t) with credible intervals,
  optionally overlaying the original observations y(τ).
- `type="trace"`: Generates MCMC trace plots for θ (and optionally σ, lp) using `MCMCChains.jl`.

# Arguments
- `results::NamedTuple`: The output from `solve_magi`.
- `type::String`: Plot type, either "traj" or "trace". Default: "traj".
- `par_names::Union{Vector{String}, Nothing}`: Names for θ parameters (for trace plot legends).
- `comp_names::Union{Vector{String}, Nothing}`: Names for state components x_d (for trajectory plot titles).
- `t_obs::Union{Vector{Float64}, Nothing}`: Time vector corresponding to `results.x_sampled` and `y_obs`. Usually the `t_obs` passed to `solve_magi`.
- `y_obs::Union{Matrix{Float64}, Nothing}`: Observation matrix (n_times × n_dims) to overlay on trajectory plots. Usually the `y_obs` passed to `solve_magi`.
- `obs::Bool`: If true, overlay observations on trajectory plots. Default: `true`.
- `ci::Bool`: If true, show credible intervals/bands on plots. Default: `true`.
- `ci_col`: Color for credible intervals. Default: `:skyblue`.
- `lower::Float64`: Lower quantile for credible interval. Default: 0.025.
- `upper::Float64`: Upper quantile for credible interval. Default: 0.975.
- `include_sigma::Bool`: Include σ in trace plots. Default: `false`.
- `include_lp::Bool`: Include log-posterior (lp) in trace plots. Default: `true`.
- `nplotcol::Int`: Number of columns for subplot layout. Default: 3.
- `kwargs...`: Additional keyword arguments passed to the underlying `Plots.plot` or `StatsPlots.plot` functions.

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

    if type == "traj"
        # --- Plot Inferred Trajectories ---
        x_sampled = results.x_sampled # Samples are (n_samples × n_times × n_dims)
        n_samples, n_times, n_dims = size(x_sampled)

        # Set component names
        if comp_names === nothing
            _comp_names = ["Component $i" for i in 1:n_dims]
        else
            if length(comp_names) != n_dims
                error("Length of comp_names ($(length(comp_names))) does not match number of dimensions ($n_dims).")
            end
            _comp_names = comp_names
        end

        # Determine plot layout (rows, columns)
        plot_layout = (Int(ceil(n_dims / nplotcol)), nplotcol)
        # Create overall plot object
        plt = Plots.plot(layout=plot_layout, legend=false, titlefont=8; kwargs...)

        # Determine time vector for x-axis
        plot_times = t_obs === nothing ? (1:n_times) : t_obs
        if length(plot_times) != n_times
             @warn "Length of provided t_obs ($(length(t_obs))) does not match trajectory length ($n_times). Using 1:$n_times for x-axis."
             plot_times = 1:n_times
        end
        xlab = t_obs === nothing ? "Index" : "Time"

        # Plot each dimension in a subplot
        for d in 1:n_dims
            x_dim_samples = x_sampled[:, :, d] # Get samples for dimension d (n_samples × n_times)

            # Calculate posterior mean and quantiles for credible interval
            x_mean = mean(x_dim_samples, dims=1) |> vec # Posterior mean trajectory
            local x_lower, x_upper # Ensure scope
            ci_enabled_dim = ci # Use local flag in case quantile calculation fails
            try
                # Calculate quantiles across samples for each time point
                x_lower = mapslices(x -> quantile(x, lower), x_dim_samples, dims=1) |> vec
                x_upper = mapslices(x -> quantile(x, upper), x_dim_samples, dims=1) |> vec
            catch e_quantile
                @warn "Could not calculate quantiles for trajectory plot CI for dim $d. CI disabled for this dimension." exception=e_quantile
                ci_enabled_dim = false # Disable CI for this dimension
                x_lower = x_mean # Assign dummy values if CI fails
                x_upper = x_mean
            end

            # Add subplot for this dimension
            Plots.plot!(plt[d], plot_times, x_mean, linecolor=:blue, label="Mean",
                        xlabel=xlab, ylabel="Level", title=_comp_names[d])

            # Add credible interval ribbon if enabled
            if ci_enabled_dim
                Plots.plot!(plt[d], plot_times, x_upper, fillrange=x_lower, fillalpha=0.3, fillcolor=ci_col,
                            linealpha=0, label="$((upper-lower)*100)% CI")
                # Optionally plot CI boundaries faintly
                # Plots.plot!(plt[d], plot_times, x_lower, linecolor=ci_col, linestyle=:dash, linealpha=0.5, label="")
                # Plots.plot!(plt[d], plot_times, x_upper, linecolor=ci_col, linestyle=:dash, linealpha=0.5, label="")
            end

            # Overlay observations if requested and available
            if obs
                if y_obs === nothing || t_obs === nothing
                    @warn "Cannot plot observations for dimension $d because y_obs or t_obs was not provided to plot_magi."
                else
                     # Check dimensions before plotting observations
                     try
                         if size(y_obs) != (n_times, n_dims)
                             @warn "Dimensions of y_obs ($(size(y_obs))) do not match trajectory dimensions ($n_times, $n_dims). Cannot plot observations for dim $d."
                         else
                            # Find finite observations for this dimension
                            valid_idx = findall(!isnan, y_obs[:, d])
                            if !isempty(valid_idx)
                                Plots.scatter!(plt[d], t_obs[valid_idx], y_obs[valid_idx, d],
                                               markercolor=:red, markersize=3, markerstrokewidth=0, label="Obs")
                            end
                         end
                     catch e_obs
                          @error "Error plotting observations for dimension $d." exception=e_obs
                     end
                end
            end
        end # end loop over dimensions
        return plt

    elseif type == "trace"
        # --- Plot MCMC Traces ---
         if !@isloaded(MCMCChains)
             error("MCMCChains package is required for trace plots. Please install and load it (`using MCMCChains`).")
         end

        try
            # Validate parameter names length if provided
            n_params_ode = size(results.theta, 2)
            if par_names !== nothing && length(par_names) != n_params_ode
                @warn "Length of provided par_names ($(length(par_names))) does not match number of θ parameters ($n_params_ode). Using default names θ[i]."
                par_names = nothing # Use default names generated by results_to_chain
            end

            # Convert results to MCMCChains object
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
