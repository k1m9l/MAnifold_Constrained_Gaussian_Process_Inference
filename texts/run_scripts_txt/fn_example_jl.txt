# fn_example_fixed.jl

using MagiJl
using DifferentialEquations
using Interpolations         # For linear interpolation
using DataFrames             # For handling data frames (optional, can use matrices)
using CSV                    # For writing CSV files
using Printf                 # For formatted printing
using Random                 # For random seed and noise generation
using Statistics             # For mean calculation
using Plots                  # For plotting results
using MCMCChains             # For summary statistics

# Make packages available to the plot_magi function
Main.Plots = Plots
Main.StatsPlots = StatsPlots

println("--- DEBUG: Starting Script ---")
println("--- DEBUG: Loading Packages ---")
println("Packages loaded.")

# --- Configuration ---
println("\n--- DEBUG: Setting up Configuration ---")
config = Dict{Symbol, Any}(
    :nobs => 41,                       # Number of observation points
    :noise => [0.2, 0.2],              # Noise SD for V and R
    :seed => rand(UInt32),             # Generate a random seed
    :niterHmc => 5000, # 20000,       # HMC iterations (reduced for faster testing)
    :burninRatio => 0.5,               # Burn-in ratio
    :filllevel => 2,                   # Discretization level
    :t_start => 0.0,                   # Start time
    :t_end => 20.0,                    # End time
    :modelName => "FN",                # Model name
    :stepSizeFactor => 0.06,           # NUTS step size factor from R script control
    :nstepsHmc => 100,                 # Default or adjust as needed
    :targetAcceptRatio => 0.8,         # Default or adjust
    :jitter => 1e-6,                   # Default jitter
    :verbose => true                   # Already prints internal MagiJl steps
)

# Set random seed
Random.seed!(config[:seed])
println("--- DEBUG: Configuration set ---")
for (k, v) in config
    # Avoid printing potentially large xInit in config summary
    if k != :xInit
      println("  $k = $v")
    else
      println("  $k = <matrix>")
    end
end
@printf("--- DEBUG: Random Seed set to: %d ---\n", config[:seed])

# --- Output Directory ---
outDir = "../results/fn/"
mkpath(outDir) # Creates the directory if it doesn't exist
println("--- DEBUG: Output directory set to: $outDir ---")

# --- True Parameters and Simulation ---
println("\n--- DEBUG: Defining True Parameters ---")
pram_true = Dict{Symbol, Any}(
    :theta => [0.2, 0.2, 3.0], # (a, b, c)
    :x0 => [-1.0, 1.0],       # Initial state (V0, R0)
    # phi and sigma are typically estimated, but kept here for reference
    :phi => [0.9486433 1.9840824; 3.2682434 1.1185157], # Format: [var; len] per dim
    :sigma => config[:noise]
)
@printf("--- DEBUG: True Theta = [%.2f, %.2f, %.2f] ---\n", pram_true[:theta]...)
@printf("--- DEBUG: True x0 = [%.2f, %.2f] ---\n", pram_true[:x0]...)

# Define the ODE problem for DifferentialEquations.jl
ode_prob = ODEProblem(MagiJl.fn_ode!, pram_true[:x0], (config[:t_start], config[:t_end]), pram_true[:theta])

# Simulate the true trajectory with high resolution
println("--- DEBUG: Simulating true trajectory using DifferentialEquations.jl ---")
sol_true = solve(ode_prob, Tsit5(), abstol=1e-8, reltol=1e-8, saveat=0.05)
t_true = sol_true.t
xtrue_tmp = hcat(sol_true.u...)' # Transpose to get [time x dim]
xtrue_df = DataFrame(time = t_true, V = xtrue_tmp[:, 1], R = xtrue_tmp[:, 2])
println("--- DEBUG: True trajectory simulated. Size: $(size(xtrue_df)) ---")
println("--- DEBUG: Head of xtrue_df:")
show(stdout, "text/plain", first(xtrue_df, 5))
println("\n")

# --- Generate Noisy Observations ---
println("--- DEBUG: Generating Noisy Observations ---")
# Create observation time points
t_obs_eq = collect(range(config[:t_start], config[:t_end], length=config[:nobs]))
@printf("--- DEBUG: Observation times 't_obs_eq' (length %d): [%.2f, ..., %.2f] ---\n", length(t_obs_eq), first(t_obs_eq), last(t_obs_eq))

# Interpolate true trajectory at observation times
println("--- DEBUG: Interpolating true trajectory at observation times ---")
itp_V = LinearInterpolation(xtrue_df.time, xtrue_df.V)
itp_R = LinearInterpolation(xtrue_df.time, xtrue_df.R)
x_at_obs = hcat(itp_V(t_obs_eq), itp_R(t_obs_eq))
println("--- DEBUG: Interpolated true values at obs times. Size: $(size(x_at_obs)) ---")

# Add noise
println("--- DEBUG: Adding noise (SDs = $(config[:noise])) ---")
y_obs_eq = x_at_obs .+ randn(config[:nobs], 2) .* pram_true[:sigma]' # Noise based on true sigma
xsim_obs_df = DataFrame(time = t_obs_eq, V = y_obs_eq[:, 1], R = y_obs_eq[:, 2])
println("--- DEBUG: Generated 'xsim_obs_df'. Size: $(size(xsim_obs_df)) ---")
println("--- DEBUG: Head of xsim_obs_df (observations):")
show(stdout, "text/plain", first(xsim_obs_df, 5))
println("\n")

# --- Discretization ---
println("--- DEBUG: Applying Discretization (filllevel = $(config[:filllevel])) ---")
t_discretized = Float64[]
points_to_insert = 2^config[:filllevel] - 1
@printf("--- DEBUG: Inserting %d points between each observation ---\n", points_to_insert)
for i in 1:(nrow(xsim_obs_df) - 1)
    t_start_interval = xsim_obs_df.time[i]
    t_end_interval = xsim_obs_df.time[i+1]
    # Ensure start and end are included correctly, handle potential floating point issues if intervals are tiny
    interval_points = range(t_start_interval, stop=t_end_interval, length = points_to_insert + 2)
    append!(t_discretized, interval_points[1:(end-1)])
end
push!(t_discretized, xsim_obs_df.time[end]) # Add the last observation time point
n_times_discretized = length(t_discretized)
@printf("--- DEBUG: Discretized time vector 't_discretized' created. Length: %d. Range: [%.3f, %.3f] ---\n", n_times_discretized, first(t_discretized), last(t_discretized))

# Create the y_obs matrix corresponding to t_discretized, filling with NaNs
y_discretized = Matrix{Union{Float64, Missing}}(missing, n_times_discretized, 2)
# Find indices in t_discretized that match t_obs_eq and fill in observed values
# Use a tolerance for matching floating point times
obs_indices = Int[]
for t_obs_val in xsim_obs_df.time
    idx = findfirst(t -> isapprox(t, t_obs_val, atol=1e-9), t_discretized)
    if idx !== nothing
        push!(obs_indices, idx)
    else
        @warn "Observation time $t_obs_val not found exactly in discretized times."
    end
end

if length(obs_indices) == nrow(xsim_obs_df)
    y_discretized[obs_indices, 1] = xsim_obs_df.V
    y_discretized[obs_indices, 2] = xsim_obs_df.R
    println("--- DEBUG: Filled observed values into discretized matrix ---")
else
    @error "--- DEBUG: Mismatch between observation times and discretized times. Check discretization logic. ---"
end
# Convert to Float64, replacing missing with NaN
y_discretized_nan = convert(Matrix{Float64}, coalesce.(y_discretized, NaN))
println("--- DEBUG: Created 'y_discretized_nan' matrix. Size: $(size(y_discretized_nan)). Contains $(sum(isnan, y_discretized_nan)) NaNs. ---")
println("--- DEBUG: Head of y_discretized_nan:")
max_rows_show = min(5, size(y_discretized_nan, 1))
show(stdout, "text/plain", round.(y_discretized_nan[1:max_rows_show, :]; digits=3))
println("\n")


# --- Define OdeSystem for MagiJl ---
println("--- DEBUG: Defining OdeSystem for MagiJl ---")
ode_system_fn = MagiJl.OdeSystem(
    MagiJl.fn_ode!,
    MagiJl.fn_ode_dx!,
    MagiJl.fn_ode_dtheta,
    [0.0, 0.0, 0.0], # Lower bounds from R script
    [Inf, Inf, Inf], # Upper bounds from R script
    3                # thetaSize
)
println("--- DEBUG: OdeSystem defined. thetaSize = $(ode_system_fn.thetaSize) ---")

# --- Initial Trajectory Guess (xInitExogenous) ---
println("--- DEBUG: Creating Initial Trajectory Guess 'xInit' via Linear Interpolation ---")
xInit = similar(y_discretized_nan)
# Use linear interpolation between the *observed* points
itp_obs_V = LinearInterpolation(xsim_obs_df.time, xsim_obs_df.V, extrapolation_bc=Line())
itp_obs_R = LinearInterpolation(xsim_obs_df.time, xsim_obs_df.R, extrapolation_bc=Line())
xInit[:, 1] = itp_obs_V(t_discretized)
xInit[:, 2] = itp_obs_R(t_discretized)
println("--- DEBUG: 'xInit' matrix created. Size: $(size(xInit)). Contains finite values: $(all(isfinite, xInit)) ---")
println("--- DEBUG: Head of xInit:")
show(stdout, "text/plain", round.(xInit[1:max_rows_show, :]; digits=3))
println("\n")

# Add xInit to the config for the solver
config[:xInit] = xInit
println("--- DEBUG: Added xInit to config dictionary ---")

# --- Run MagiSolver ---
println("\n--- DEBUG: === Starting MagiSolver === ---")
OursStartTime = time()

# *** FIX APPLIED HERE ***
local results = nothing # Explicitly declare as local BEFORE the try block

try
    # Call the main solver function from MagiJl.jl
    # Make sure config[:verbose]=true is passed if you want internal MagiJl messages
    global results = MagiJl.solve_magi(y_discretized_nan, t_discretized, ode_system_fn, config) # This now assigns to the 'results' declared above
    println("\n--- DEBUG: === MagiSolver Finished === ---")
    # +++ ADDED PRINT TYPE HERE +++
    println("--- DEBUG: Type of 'results' after call: $(typeof(results)) ---")
    # +++++++++++++++++++++++++++++

catch e
    println("\n--- DEBUG: === ERROR during MagiSolver execution === ---")
    showerror(stdout, e, catch_backtrace())
    println("\n")
    println("--- DEBUG: Error caught in main script's try-catch for solve_magi. ---") # +++ ADDED +++
    # Optional: rethrow(e) to stop execution on error
end

OursTimeUsed = time() - OursStartTime
# Moved timing print here for clarity after try-catch finishes
@printf("--- DEBUG: Time used by MagiSolver: %.2f seconds ---\n", OursTimeUsed)

# --- Post-processing ---
# This check should now work correctly if the try block succeeded
if results !== nothing && results isa NamedTuple
    println("\n--- DEBUG: Starting Post-processing ---")

    # Rename results fields slightly for clarity if desired
    gpode = results # Use the same name as R script for consistency
    println("--- DEBUG: Results object 'gpode' structure: ---")
    println("  Keys: $(keys(gpode))")
    @printf("  theta size: %s\n", string(size(gpode.theta)))
    @printf("  x_sampled size: %s\n", string(size(gpode.x_sampled)))
    @printf("  sigma size: %s\n", string(size(gpode.sigma)))
    @printf("  phi size: %s\n", string(size(gpode.phi)))
    @printf("  lp length: %d\n", length(gpode.lp))


    # Calculate inferred trajectory mean
    println("--- DEBUG: Calculating mean of inferred trajectory (x_sampled) ---")
    xsampled_mean = mean(gpode.x_sampled, dims=1)[1,:,:] #[time x dim]
    println("--- DEBUG: Calculated 'xsampled_mean'. Size: $(size(xsampled_mean)) ---")

    # Calculate inferred theta mean
    println("--- DEBUG: Calculating mean of inferred parameters (theta) ---")
    theta_mean = mean(gpode.theta, dims=1)[1,:] # Vector
    @printf("--- DEBUG: Calculated 'theta_mean': [%.4f, %.4f, %.4f] ---\n", theta_mean...)


    # Save inferred means to CSV
    println("--- DEBUG: Saving inferred means to CSV ---")
    csv_traj_filename = joinpath(outDir, config[:modelName] * "-$(config[:seed])-" * "inferred_trajectory.csv")
    try
        CSV.write(csv_traj_filename, DataFrame(time=t_discretized, V=xsampled_mean[:,1], R=xsampled_mean[:,2]))
        println("--- DEBUG: Saved inferred trajectory mean to: $csv_traj_filename ---")
    catch err_csv
        @error "--- DEBUG: Failed to save trajectory CSV." error=err_csv
    end


    csv_theta_filename = joinpath(outDir, config[:modelName] * "-$(config[:seed])-" * "inferred_theta.csv")
    try
        theta_df = DataFrame(parameter=["a", "b", "c"], mean_value=theta_mean)
        CSV.write(csv_theta_filename, theta_df)
        println("--- DEBUG: Saved inferred theta mean to: $csv_theta_filename ---")
    catch err_csv
         @error "--- DEBUG: Failed to save theta CSV." error=err_csv
    end

    # Print summary statistics
    println("--- DEBUG: Generating MCMC summary statistics using MCMCChains ---")
    param_names_fn = ["a", "b", "c"]
    try
      summary_info = magi_summary(gpode; par_names=param_names_fn, include_sigma=true)
      # println(summary_info) # magi_summary already prints
      println("--- DEBUG: Summary statistics generated and printed. ---")
    catch err_summary
      @error "--- DEBUG: Failed to generate summary statistics." error=err_summary
    end


    # Generate Plots (similar to R's plotPostSamplesFlex)
    # 1. Trajectory Plot
    println("--- DEBUG: Generating Trajectory Plot ---")
    comp_names_fn = ["V (Voltage)", "R (Recovery)"]
    traj_plot = plot(layout=(2, 1), legend=false, size=(800, 600))
    try
        # Determine credible interval bounds
        x_lower_V = mapslices(x -> quantile(x, 0.025), gpode.x_sampled[:,:,1], dims=1)'
        x_upper_V = mapslices(x -> quantile(x, 0.975), gpode.x_sampled[:,:,1], dims=1)'
        x_lower_R = mapslices(x -> quantile(x, 0.025), gpode.x_sampled[:,:,2], dims=1)'
        x_upper_R = mapslices(x -> quantile(x, 0.975), gpode.x_sampled[:,:,2], dims=1)'

        # Plot V
        plot!(traj_plot[1], t_discretized, xsampled_mean[:, 1], ribbon=(xsampled_mean[:, 1] .- x_lower_V, x_upper_V .- xsampled_mean[:, 1]), fillalpha=0.2, color=:blue, label="Inferred Mean V")
        plot!(traj_plot[1], xtrue_df.time, xtrue_df.V, color=:red, linestyle=:dash, label="True V")
        scatter!(traj_plot[1], xsim_obs_df.time, xsim_obs_df.V, color=:black, markersize=3, label="Observations V")
        title!(traj_plot[1], comp_names_fn[1])
        ylabel!(traj_plot[1], "Level")

        # Plot R
        plot!(traj_plot[2], t_discretized, xsampled_mean[:, 2], ribbon=(xsampled_mean[:, 2] .- x_lower_R, x_upper_R .- xsampled_mean[:, 2]), fillalpha=0.2, color=:green, label="Inferred Mean R")
        plot!(traj_plot[2], xtrue_df.time, xtrue_df.R, color=:orange, linestyle=:dash, label="True R")
        scatter!(traj_plot[2], xsim_obs_df.time, xsim_obs_df.R, color=:black, markersize=3, label="Observations R")
        title!(traj_plot[2], comp_names_fn[2])
        xlabel!(traj_plot[2], "Time")
        ylabel!(traj_plot[2], "Level")

        plot!(traj_plot, legend=true) # Add legend to the whole plot
        display(traj_plot) # Display the plot in the plot pane if running interactively
        traj_plot_filename = joinpath(outDir, config[:modelName] * "-$(config[:seed])-" * "trajectory_plot.png")
        savefig(traj_plot, traj_plot_filename)
        println("--- DEBUG: Saved trajectory plot to: $traj_plot_filename ---")
    catch err_plot
        @error "--- DEBUG: Failed to generate or save trajectory plot." error=err_plot
    end

    # 2. Trace Plot
    println("--- DEBUG: Generating Trace Plot ---")
    try
        trace_plot = plot_magi(gpode; type="trace", par_names=param_names_fn, include_sigma=true, include_lp=true)
        display(trace_plot)
        trace_plot_filename = joinpath(outDir, config[:modelName] * "-$(config[:seed])-" * "trace_plot.png")
        savefig(trace_plot, trace_plot_filename)
        println("--- DEBUG: Saved trace plot to: $trace_plot_filename ---")
    catch err_plot
         @error "--- DEBUG: Failed to generate or save trace plot." error=err_plot
    end

    println("\n--- DEBUG: Post-processing Complete ---")
else
    # This message will now only print if the catch block was executed OR solve_magi returned something other than a NamedTuple
    println("\n--- DEBUG: Skipping post-processing because 'results' is nothing or not a NamedTuple. Type was: $(typeof(results)) ---")
end

println("\n--- DEBUG: Translated FN Example Script Finished. ---")