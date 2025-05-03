# test/runtests.jl (Version with ALL descriptions removed from @test)

using MagiJl
using Test
using DifferentialEquations # For generating test data
using Statistics          # For checking means
using Random              # For seeding
using Optim               # Needed for fixed sigma test setup

println("--- Running MagiJl Tests ---")

# --- Helper Function to Generate Test Data ---
function generate_fn_test_data(;
    true_theta = [0.2, 0.2, 3.0],
    true_x0 = [-1.0, 1.0],
    true_sigma = [0.25, 0.35], # Known noise levels for testing
    t_start = 0.0,
    t_end = 5.0, # Shorter time for faster tests
    dt_obs = 0.5,
    dt_save = 0.05
    )

    n_dims = length(true_x0)
    # Define ODE problem
    ode_prob = ODEProblem(MagiJl.fn_ode!, true_x0, (t_start, t_end), true_theta)
    # Simulate true trajectory
    sol_true = solve(ode_prob, Tsit5(), abstol=1e-7, reltol=1e-7, saveat=dt_save)
    # Observation times
    t_obs_vec = collect(t_start:dt_obs:t_end)
    # Get true values at observation times
    xtrue_at_obs = sol_true(t_obs_vec)
    x_matrix = hcat(xtrue_at_obs.u...)' # n_obs x n_dims
    # Add noise
    Random.seed!(123) # Seed for noise generation consistency
    noise = randn(size(x_matrix)) .* reshape(true_sigma, 1, n_dims)
    y_obs_matrix = x_matrix .+ noise

    # Prepare full time grid (matches observation times for simplicity here)
    # In real tests, you might use discretization like in the examples
    t_full = t_obs_vec
    y_full_nan = y_obs_matrix # Assuming observations at all discretization points for this test

    return t_full, y_full_nan, true_theta, true_sigma
end

# --- Define the ODE System for Tests ---
ode_system_fn_test = MagiJl.OdeSystem(
    MagiJl.fn_ode!,
    MagiJl.fn_ode_dx!,
    MagiJl.fn_ode_dtheta,
    [0.0, 0.0, 0.0], # Lower bounds
    [Inf, Inf, Inf], # Upper bounds
    3                # thetaSize
)

# === Test Suite ===
@testset "MagiJl Sigma Sampling Tests" begin

    # Generate common test data
    println("Generating test data...")
    t_test, y_test, true_theta_test, true_sigma_test = generate_fn_test_data()
    n_times_test, n_dims_test = size(y_test)
    println("Test data generated: $(n_times_test) time points, $(n_dims_test) dimensions.")

    test_config_base = Dict{Symbol,Any}(
        :niterHmc => 10000, # INCREASED SIGNIFICANTLY (e.g., 2000 samples after burn-in)
        :burninRatio => 0.5,
        :verbose => false,
        :bandSize => 20, # INCREASED to default or higher if needed
        :stepSizeFactor => 0.005 # DECREASED - try smaller step size
    )

    # --- Test Case 1: Sigma is Unknown (Default Behavior) ---
    @testset "Unknown Sigma Estimation" begin
        println("Running test: Unknown Sigma...")
        config_unknown = copy(test_config_base)
        # NO :sigma or :phi provided in config

        results_unknown = nothing
        try
            results_unknown = MagiJl.solve_magi(y_test, t_test, ode_system_fn_test, config_unknown)
        catch e
            println("Error during solve_magi (Unknown Sigma): $e")
            # Optionally rethrow or handle specific errors
        end

        @test results_unknown !== nothing # Description removed

        if results_unknown !== nothing
            @test results_unknown isa NamedTuple

            # Check sigma results specifically
            @test haskey(results_unknown, :sigma)
            sigma_samples = results_unknown.sigma
            @test sigma_samples isa Matrix{Float64}

            n_samples_out = size(sigma_samples, 1)
            n_dims_out = size(sigma_samples, 2)
            expected_samples = config_unknown[:niterHmc] * (1.0 - config_unknown[:burninRatio])

            @test round(Int, n_samples_out) == round(Int, expected_samples) # Description removed
            @test n_dims_out == n_dims_test # Description removed

            # Check if mean of sampled sigma is reasonable
            mean_sigma_est = vec(mean(sigma_samples, dims=1))
            println("  True Sigma: ", round.(true_sigma_test; digits=3))
            println("  Estimated Mean Sigma: ", round.(mean_sigma_est; digits=3))
            @test isapprox(mean_sigma_est, true_sigma_test, atol=0.3) # Keep atol here

            # Check theta results
            @test haskey(results_unknown, :theta)
            mean_theta_est = vec(mean(results_unknown.theta, dims=1))
            println("  True Theta: ", round.(true_theta_test; digits=3))
            println("  Estimated Mean Theta: ", round.(mean_theta_est; digits=3))
            @test isapprox(mean_theta_est, true_theta_test, atol=0.5) # Keep atol here
        end
    end

    # --- Test Case 2: Sigma is Fixed ---
    @testset "Fixed Sigma Execution" begin
        println("Running test: Fixed Sigma...")
        config_fixed = copy(test_config_base)

        # Provide known sigma AND estimated phi
        phi_init_fixed = zeros(2, n_dims_test)
        # sigma_placeholder = zeros(n_dims_test) # Not needed
        println("  Initializing Phi for fixed sigma test...")
        for dim in 1:n_dims_test
            log_params_guess = [log(1.0), log(1.0), log(0.1)]
            try
                opt_params = MagiJl.Initialization.optimize_gp_hyperparameters(
                    y_test[:, dim], t_test, "matern52", log_params_guess;
                    jitter=1e-6, optim_options=Optim.Options(iterations=50)
                )
                 phi_init_fixed[:, dim] = opt_params[1:2]
            catch init_err
                @warn "GP Init failed for dim $dim in fixed sigma test setup. Using defaults."
                phi_init_fixed[:, dim] = [1.0, 1.0]
            end
        end
        println("  Using Initialized Phi: ", round.(phi_init_fixed; digits=3))
        println("  Using Fixed Sigma: ", round.(true_sigma_test; digits=3))

        config_fixed[:sigma] = true_sigma_test
        config_fixed[:phi] = phi_init_fixed

        results_fixed = nothing
        try
             results_fixed = MagiJl.solve_magi(y_test, t_test, ode_system_fn_test, config_fixed)
        catch e
            println("Error during solve_magi (Fixed Sigma): $e")
        end

        @test results_fixed !== nothing # Description removed

        if results_fixed !== nothing
            @test results_fixed isa NamedTuple

            # Check sigma results
            @test haskey(results_fixed, :sigma)
            sigma_values_out = results_fixed.sigma
            @test sigma_values_out isa Matrix{Float64}

            n_samples_out_fix = size(sigma_values_out, 1)
            n_dims_out_fix = size(sigma_values_out, 2)
            expected_samples_fix = config_fixed[:niterHmc] * (1.0 - config_fixed[:burninRatio])

            @test round(Int, n_samples_out_fix) == round(Int, expected_samples_fix) # Description removed
            @test n_dims_out_fix == n_dims_test # Description removed

            # Check that ALL rows are equal to the fixed input sigma
            expected_sigma_matrix = repeat(reshape(true_sigma_test, 1, n_dims_test), round(Int, n_samples_out_fix), 1)
            @test sigma_values_out ≈ expected_sigma_matrix # Description removed

             # Check theta results
            @test haskey(results_fixed, :theta)
            mean_theta_est_fix = vec(mean(results_fixed.theta, dims=1))
            println("  True Theta: ", round.(true_theta_test; digits=3))
            println("  Estimated Mean Theta (Fixed Sigma): ", round.(mean_theta_est_fix; digits=3))
            @test isapprox(mean_theta_est_fix, true_theta_test, atol=0.5) # Keep atol
        end
    end

    # --- Test Case 3: Providing Full Initial Parameters (including log sigma) ---
     @testset "Provided Full Initial Parameters" begin
        println("Running test: Full Initial Parameters Provided...")
        config_full_init = copy(test_config_base)

        # Create a plausible full initial parameter vector
        x_init_flat = vec(y_test) # Simple init: use noisy data
        theta_init_full = true_theta_test .* 0.9 # Start slightly off
        log_sigma_init_full = log.(true_sigma_test .* 1.1) # Start slightly off
        full_params = vcat(x_init_flat, theta_init_full, log_sigma_init_full)

        results_full_init = nothing
        try
            # Call solve_magi providing the full vector
             results_full_init = MagiJl.solve_magi(y_test, t_test, ode_system_fn_test, config_full_init; initial_params=full_params)
        catch e
            println("Error during solve_magi (Full Init Params): $e")
        end

         @test results_full_init !== nothing # Description removed

         if results_full_init !== nothing
            @test results_full_init isa NamedTuple
            # Check that sigma was sampled and is reasonable
            @test haskey(results_full_init, :sigma)
            sigma_samples_fi = results_full_init.sigma
            @test sigma_samples_fi isa Matrix{Float64}
            @test size(sigma_samples_fi, 2) == n_dims_test
            mean_sigma_est_fi = vec(mean(sigma_samples_fi, dims=1))
             println("  True Sigma: ", round.(true_sigma_test; digits=3))
             println("  Estimated Mean Sigma (Full Init): ", round.(mean_sigma_est_fi; digits=3))
            @test isapprox(mean_sigma_est_fi, true_sigma_test, atol=0.3) # Keep atol
         end
     end


end # End Test Suite

println("--- MagiJl Tests Finished ---")