# fn_example_api_reworked.jl

using MagiJl
using DifferentialEquations
using Interpolations         # For data generation/interpolation
using DataFrames             # For handling data frames
using CSV                    # For writing CSV files
using Printf                 # For formatted printing
using Random                 # For random seed and noise generation
using Statistics             # For mean calculation
using Plots                  # For plotting results
using MCMCChains             # For summary statistics
using StatsPlots             # For plotting MCMC chains

println("--- DEBUG: Starting API Reworked Script ---")
println("--- DEBUG: Loading Packages ---")
println("Packages loaded.")

# --- Configuration for Data Generation ---
# Kept separate from solver config for clarity
data_gen_config = Dict{Symbol, Any}(
    :nobs => 100,                       # Number of observation points
    :noise_sd => [0.2, 0.2],           # True noise SD for V and R
    :seed => rand(UInt32),             # Generate a random seed
    :t_start => 0.0,                   # Start time
    :t_end => 20.0,                    # End time
    :modelName => "FN",                # Model name (used for output)
    :filllevel => 2                    # Discretization level for preparing input data
)

# --- Configuration for Magi Solver ---
# Define only settings relevant to the solver itself
# OMIT :sigma, :phi, :xInit, :thetaInit to let MagiJl handle estimation/initialization
config_solver = Dict{Symbol, Any}(
    :niterHmc => 50000,                # HMC iterations
    :burninRatio => 0.5,               # Burn-in ratio
    :stepSizeFactor => 0.06,           # NUTS step size factor
    :targetAcceptRatio => 0.8,         # NUTS target acceptance
    :jitter => 1e-6,                   # Numerical stability jitter
    :verbose => true,                   # Print internal MagiJl steps
    # :bandSize => 20,                 # Optional: Set if default isn't desired
    :priorTemperature => [1.0, 1.0, 5.0] # Optional: Set if default isn't desired
)

# Set random seed for reproducibility
Random.seed!(data_gen_config[:seed])
println("--- DEBUG: Data Generation Config ---")
for (k, v) in data_gen_config; println("  $k = $v"); end
println("--- DEBUG: Solver Config ---")
for (k, v) in config_solver; println("  $k = $v"); end
@printf("--- DEBUG: Random Seed set to: %d ---\n", data_gen_config[:seed])

# --- Output Directory ---
outDir = joinpath(homedir(), "Desktop", "MAGI_OUTPUT") # Example absolute path
mkpath(outDir)
println("--- DEBUG: Output directory set to: $outDir ---")

# ==================================================
# Section 1: Generate Synthetic Data (Keep similar logic)
# ==================================================
println("\n--- DEBUG: Generating Synthetic Data ---")
pram_true = Dict{Symbol, Any}(
    :theta => [0.2, 0.2, 3.0], # (a, b, c)
    :x0 => [-1.0, 1.0],       # Initial state (V0, R0)
    :sigma => data_gen_config[:noise_sd] # Use true noise from data config
)
@printf("--- DEBUG: True Theta = [%.2f, %.2f, %.2f] ---\n", pram_true[:theta]...)
@printf("--- DEBUG: True x0 = [%.2f, %.2f] ---\n", pram_true[:x0]...)
@printf("--- DEBUG: True Sigma = [%.2f, %.2f] ---\n", pram_true[:sigma]...)

ode_prob = ODEProblem(MagiJl.fn_ode!, pram_true[:x0], (data_gen_config[:t_start], data_gen_config[:t_end]), pram_true[:theta])
sol_true = solve(ode_prob, Tsit5(), abstol=1e-8, reltol=1e-8, saveat=0.05)
xtrue_df = DataFrame(time = sol_true.t, V = hcat(sol_true.u...)[1, :], R = hcat(sol_true.u...)[2, :])

t_obs_eq = collect(range(data_gen_config[:t_start], data_gen_config[:t_end], length=data_gen_config[:nobs]))
itp_V = LinearInterpolation(xtrue_df.time, xtrue_df.V)
itp_R = LinearInterpolation(xtrue_df.time, xtrue_df.R)
x_at_obs = hcat(itp_V(t_obs_eq), itp_R(t_obs_eq))
y_obs_eq = x_at_obs .+ randn(data_gen_config[:nobs], 2) .* pram_true[:sigma]' # Add true noise
xsim_obs_df = DataFrame(time = t_obs_eq, V = y_obs_eq[:, 1], R = y_obs_eq[:, 2])

println("--- DEBUG: Simulated Observations (Head):")
show(stdout, "text/plain", first(xsim_obs_df, 5))
println("\n")

# ==================================================
# Section 2: Prepare Data for MagiJl (Discretization)
# ==================================================
# This section remains necessary if solve_magi expects the discretized format.
# If solve_magi could take raw data + filllevel, this could be removed.
println("--- DEBUG: Preparing Discretized Data for Solver ---")
t_discretized = Float64[]
points_to_insert = 2^data_gen_config[:filllevel] - 1
for i in 1:(nrow(xsim_obs_df) - 1)
    t_start_interval, t_end_interval = xsim_obs_df.time[i], xsim_obs_df.time[i+1]
    interval_points = range(t_start_interval, stop=t_end_interval, length = points_to_insert + 2)
    append!(t_discretized, interval_points[1:(end-1)])
end
push!(t_discretized, xsim_obs_df.time[end])
n_times_discretized = length(t_discretized)
@printf("--- DEBUG: Discretized time vector created. Length: %d. Range: [%.3f, %.3f] ---\n", n_times_discretized, first(t_discretized), last(t_discretized))

y_discretized = Matrix{Union{Float64, Missing}}(missing, n_times_discretized, 2)
obs_indices = Int[]
for t_obs_val in xsim_obs_df.time
    idx = findfirst(t -> isapprox(t, t_obs_val, atol=1e-9), t_discretized)
    if idx !== nothing; push!(obs_indices, idx); else @warn "Obs time $t_obs_val not found in discretized times."; end
end

if length(obs_indices) == nrow(xsim_obs_df)
    y_discretized[obs_indices, 1] = xsim_obs_df.V
    y_discretized[obs_indices, 2] = xsim_obs_df.R
else
    @error "Mismatch between observation times and discretized times."
end
y_discretized_nan = convert(Matrix{Float64}, coalesce.(y_discretized, NaN)) # Input for MagiJl
println("--- DEBUG: Created 'y_discretized_nan' matrix for solver. Size: $(size(y_discretized_nan)). Contains $(sum(isnan, y_discretized_nan)) NaNs. ---")

# ==================================================
# Section 3: Define ODE System for MagiJl
# ==================================================
println("--- DEBUG: Defining OdeSystem for MagiJl ---")
ode_system_fn = MagiJl.OdeSystem(
    MagiJl.fn_ode!,
    MagiJl.fn_ode_dx!,
    MagiJl.fn_ode_dtheta,
    [0.0, 0.0, 0.0], # Lower bounds for theta (a, b, c)
    [Inf, Inf, Inf], # Upper bounds for theta
    3                # thetaSize
)
println("--- DEBUG: OdeSystem defined. thetaSize = $(ode_system_fn.thetaSize) ---")

# ==================================================
# Section 4: Run MagiSolver (Estimating Theta and Sigma)
# ==================================================
println("\n--- DEBUG: === Starting MagiSolver (Estimating Theta & Sigma) === ---")
# NOTE: We do NOT provide :sigma, :phi, :xInit, or :thetaInit in config_solver.
# MagiJl will initialize x and theta internally and estimate sigma and phi.
OursStartTime = time()
local results = nothing

try
    # Pass the discretized data and the *solver-specific* config
    global results = MagiJl.solve_magi(
        y_discretized_nan,
        t_discretized,
        ode_system_fn,
        config_solver # Pass the config containing only solver settings
    )
    println("\n--- DEBUG: === MagiSolver Finished === ---")
    println("--- DEBUG: Type of 'results' after call: $(typeof(results)) ---")
catch e
    println("\n--- DEBUG: === ERROR during MagiSolver execution === ---")
    showerror(stdout, e, catch_backtrace())
    println("\n--- DEBUG: Error caught in main script's try-catch for solve_magi. ---")
end

OursTimeUsed = time() - OursStartTime
@printf("--- DEBUG: Time used by MagiSolver: %.2f seconds ---\n", OursTimeUsed)

# ==================================================
# Section 5: Post-processing (Keep similar logic)
# ==================================================
if results !== nothing && results isa NamedTuple
    println("\n--- DEBUG: Starting Post-processing ---")

    gpode = results # Use consistent naming
    println("--- DEBUG: Results object 'gpode' structure: ---")
    println("  Keys: $(keys(gpode))")
    # Print sizes, check that sigma is now a matrix of samples
    @printf("  theta size: %s\n", string(size(gpode.theta)))
    @printf("  x_sampled size: %s\n", string(size(gpode.x_sampled)))
    @printf("  sigma size: %s (Expected: n_samples x n_dims)\n", string(size(gpode.sigma)))
    @printf("  phi size: %s\n", string(size(gpode.phi))) # Shows estimated phi
    @printf("  lp length: %d\n", length(gpode.lp))

    # Calculate inferred trajectory mean
    xsampled_mean = mean(gpode.x_sampled, dims=1)[1,:,:]
    println("--- DEBUG: Calculated inferred trajectory mean. Size: $(size(xsampled_mean)) ---")

    # Calculate inferred theta mean
    theta_mean = mean(gpode.theta, dims=1)[1,:]
    @printf("--- DEBUG: Calculated mean estimated theta: [%.4f, %.4f, %.4f] ---\n", theta_mean...)

    # Calculate inferred sigma mean <<< NEW since sigma is estimated
    sigma_mean = mean(gpode.sigma, dims=1)[1,:]
    @printf("--- DEBUG: Calculated mean estimated sigma: [%.4f, %.4f] ---\n", sigma_mean...)

    # --- Save Results ---
    println("--- DEBUG: Saving inferred means and results to CSV/Plots ---")
    # Trajectory CSV
    csv_traj_filename = joinpath(outDir, data_gen_config[:modelName] * "-$(data_gen_config[:seed])-" * "inferred_trajectory.csv")
    try CSV.write(csv_traj_filename, DataFrame(time=t_discretized, V=xsampled_mean[:,1], R=xsampled_mean[:,2])) catch e @error "CSV Traj Save Error" error=e end
    println("--- DEBUG: Saved inferred trajectory mean to: $csv_traj_filename ---")

    # Parameters CSV (Theta and Sigma)
    csv_params_filename = joinpath(outDir, data_gen_config[:modelName] * "-$(data_gen_config[:seed])-" * "inferred_parameters.csv")
    try
        param_df = DataFrame(
            parameter = ["theta_a", "theta_b", "theta_c", "sigma_V", "sigma_R"],
            true_value = [pram_true[:theta]..., pram_true[:sigma]...],
            mean_estimate = [theta_mean..., sigma_mean...]
        )
        CSV.write(csv_params_filename, param_df)
    catch e @error "CSV Params Save Error" error=e end
    println("--- DEBUG: Saved inferred parameter means to: $csv_params_filename ---")

    # --- MCMC Summary Statistics ---
    println("--- DEBUG: Generating MCMC summary statistics (including sigma) ---")
    param_names_fn = ["a", "b", "c"]
    try
      # Use MagiJl's summary function, ensuring sigma is included
      summary_info = magi_summary(gpode; par_names=param_names_fn, include_sigma=true)
      println("--- DEBUG: Summary statistics generated and printed. ---")
    catch err_summary @error "--- DEBUG: Failed to generate summary statistics." error=err_summary end

    # --- Generate Plots ---
    comp_names_fn = ["V (Voltage)", "R (Recovery)"]

    # 1. Trajectory Plot (Keep similar logic, ensure CI calculation is robust)
    println("--- DEBUG: Generating Trajectory Plot ---")
    traj_plot = plot(layout=(2, 1), legend=false, size=(800, 600))
    try
        x_lower_V = mapslices(x -> quantile(filter(isfinite, x), 0.025), gpode.x_sampled[:,:,1], dims=1)'
        x_upper_V = mapslices(x -> quantile(filter(isfinite, x), 0.975), gpode.x_sampled[:,:,1], dims=1)'
        x_lower_R = mapslices(x -> quantile(filter(isfinite, x), 0.025), gpode.x_sampled[:,:,2], dims=1)'
        x_upper_R = mapslices(x -> quantile(filter(isfinite, x), 0.975), gpode.x_sampled[:,:,2], dims=1)'

        # Plot V
        plot!(traj_plot[1], t_discretized, xsampled_mean[:, 1], ribbon=(xsampled_mean[:, 1] .- x_lower_V, x_upper_V .- xsampled_mean[:, 1]), fillalpha=0.2, color=:blue, label="Inferred Mean V")
        plot!(traj_plot[1], xtrue_df.time, xtrue_df.V, color=:red, linestyle=:dash, label="True V")
        scatter!(traj_plot[1], xsim_obs_df.time, xsim_obs_df.V, color=:black, markersize=3, label="Observations V")
        title!(traj_plot[1], comp_names_fn[1]); ylabel!(traj_plot[1], "Level")

        # Plot R
        plot!(traj_plot[2], t_discretized, xsampled_mean[:, 2], ribbon=(xsampled_mean[:, 2] .- x_lower_R, x_upper_R .- xsampled_mean[:, 2]), fillalpha=0.2, color=:green, label="Inferred Mean R")
        plot!(traj_plot[2], xtrue_df.time, xtrue_df.R, color=:orange, linestyle=:dash, label="True R")
        scatter!(traj_plot[2], xsim_obs_df.time, xsim_obs_df.R, color=:black, markersize=3, label="Observations R")
        title!(traj_plot[2], comp_names_fn[2]); xlabel!(traj_plot[2], "Time"); ylabel!(traj_plot[2], "Level")

        plot!(traj_plot, legend=true) # Add legend
        display(traj_plot)
        traj_plot_filename = joinpath(outDir, data_gen_config[:modelName] * "-$(data_gen_config[:seed])-" * "trajectory_plot.png")
        savefig(traj_plot, traj_plot_filename)
        println("--- DEBUG: Saved trajectory plot to: $traj_plot_filename ---")
    catch err_plot @error "--- DEBUG: Failed to generate or save trajectory plot." error=err_plot end

    # 2. Trace Plot (Ensure sigma is included)
    println("--- DEBUG: Generating Trace Plot (including sigma) ---")
    try
        # Use MagiJl's plot function for trace plots
        trace_plot = plot_magi(gpode; type="trace", par_names=param_names_fn, include_sigma=true, include_lp=true)
        display(trace_plot)
        trace_plot_filename = joinpath(outDir, data_gen_config[:modelName] * "-$(data_gen_config[:seed])-" * "trace_plot.png")
        savefig(trace_plot, trace_plot_filename)
        println("--- DEBUG: Saved trace plot to: $trace_plot_filename ---")
    catch err_plot @error "--- DEBUG: Failed to generate or save trace plot." error=err_plot end

    println("\n--- DEBUG: Post-processing Complete ---")
else
    println("\n--- DEBUG: Skipping post-processing because 'results' is nothing or not a NamedTuple. Type was: $(typeof(results)) ---")
end

println("\n--- DEBUG: API Reworked FN Example Script Finished. ---")