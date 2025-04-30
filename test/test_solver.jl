# test/test_solver.jl

using MagiJl       # Import the main module
using Test
using LinearAlgebra
using Statistics   # <<<--- ADD THIS LINE to import mean()

# using DelimitedFiles # Uncomment if you load data from FN.csv

@testset "solve_magi End-to-End Tests" begin

    @testset "FitzHugh-Nagumo Example (Short Run)" begin
        println("\n--- Solver Test: FN Example (Short Run) ---")

        # --- 1. Load/Define Data ---
        # Using a simple, small dataset for this initial test
        t_obs = [0.0, 1.0, 2.0, 3.0, 4.0]
        n_times = length(t_obs)
        n_dims = 2 # V, R for FN

        # Create some plausible observation data
        # You can replace this with data loaded from FN.csv if preferred
        # Example: simulate noisy data around a simple trajectory
        true_V = 1.0 .+ 0.1 .* sin.(t_obs)
        true_R = 0.5 .+ 0.05 .* cos.(t_obs)
        y_obs = hcat(true_V, true_R) .+ randn(n_times, n_dims) * 0.15

        # Introduce a missing value to test handling
        y_obs[3, 1] = NaN
        println("Using data: $(n_times) time points, $(n_dims) dimensions.")

        # --- 2. Define OdeSystem ---
        # Explicitly qualify OdeSystem with the module name
        ode_system_fn = MagiJl.OdeSystem(
            MagiJl.fn_ode!,
            MagiJl.fn_ode_dx!,
            MagiJl.fn_ode_dtheta,
            [-Inf, -Inf, 0.01], # Example Lower bounds (e.g., c > 0)
            [Inf, Inf, Inf],    # Example Upper bounds
            3                   # thetaSize
        )

        # --- 3. Create Config ---
        # Use fixed initial values and minimal MCMC iterations for this test
        phi_known = [1.0 1.0;  # Variance Dim 1, Dim 2
                     1.0 1.0]  # Lengthscale Dim 1, Dim 2
        sigma_known = [0.15, 0.15] # Noise SD Dim 1, Dim 2
        theta_init_known = [0.5, 0.6, 0.7] # Optional: Fix theta init

        config = Dict{Symbol, Any}(
            :kernel => "matern52",
            :bandSize => min(n_times - 1, 5), # Keep bandSize small & valid
            :niterHmc => 20,          # VERY FEW iterations
            :burninRatio => 0.5,      # 50% burn-in -> 10 samples post-burnin
            :stepSizeFactor => 0.05,  # Initial step size factor for NUTS
            :phi => phi_known,        # USE FIXED PHI
            :sigma => sigma_known,    # USE FIXED SIGMA
            :thetaInit => theta_init_known, # Use fixed theta init
            :jitter => 1e-5           # Specify jitter
            # :xInit => #= Optionally provide known x matrix here =#
        )
        n_samples_expected = Int(config[:niterHmc] * (1 - config[:burninRatio]))
        println("Test Config: niterHmc=$(config[:niterHmc]), bandSize=$(config[:bandSize])")

        # --- 4. Call solve_magi ---
        results = nothing # Ensure scope outside try block
        println("Calling solve_magi...")
        # Use @testset to capture potential errors during execution
        @testset "Function Execution" begin
             try
                 # Call solve_magi directly as it's exported by MagiJl
                 results = solve_magi(y_obs, t_obs, ode_system_fn, config)
                 println("solve_magi completed without throwing an error.")
                 @test results !== nothing # Basic check that it returned something
             catch e
                 println("\nERROR during solve_magi execution:")
                 showerror(stdout, e, catch_backtrace())
                 println()
                 @test false # Force test failure if an error occurs
             end
        end

        # --- 5. Check Results (only if execution succeeded) ---
        if results !== nothing && typeof(results) <: NamedTuple
            println("Checking results structure...")
            n_params_ode = ode_system_fn.thetaSize

            @testset "Output Dimensions and Types" begin
                 @test haskey(results, :theta) && results.theta isa Matrix{Float64}
                 @test size(results.theta) == (n_samples_expected, n_params_ode)

                 @test haskey(results, :x_sampled) && results.x_sampled isa Array{Float64, 3}
                 @test size(results.x_sampled) == (n_samples_expected, n_times, n_dims)

                 @test haskey(results, :sigma) && results.sigma isa Matrix{Float64}
                 @test size(results.sigma) == (1, n_dims) # Should be the fixed sigma

                 @test haskey(results, :phi) && results.phi isa Matrix{Float64}
                 @test size(results.phi) == (2, n_dims) # Should be the fixed phi

                 # Check lp length AFTER fixing the extraction logic
                 @test haskey(results, :lp) && results.lp isa Vector{Float64}
                 @test length(results.lp) == n_samples_expected
            end

            @testset "Output Values Sanity Check" begin
                @test all(isfinite, results.theta)
                @test all(isfinite, results.x_sampled)
                @test all(results.sigma .> 0)
                @test all(results.phi .> 0)
                # Only check lp if it's not empty
                if !isempty(results.lp)
                    @test all(isfinite, results.lp)
                    # Use mean now that Statistics is imported
                    println("Log Posterior mean: ", round(mean(results.lp), digits=3))
                else
                    @warn "Log posterior vector is empty, skipping check."
                end

                # Check if sampled theta respects bounds (approximately, allow for float issues)
                for p_idx in 1:n_params_ode
                    lb = ode_system_fn.thetaLowerBound[p_idx]
                    ub = ode_system_fn.thetaUpperBound[p_idx]
                    @test all(results.theta[:, p_idx] .>= lb - 1e-9)
                    @test all(results.theta[:, p_idx] .<= ub + 1e-9)
                end
                 # Use mean now that Statistics is imported
                println("Sampled theta mean: ", round.(mean(results.theta, dims=1), digits=3))
            end
        else
             @warn "Skipping result checks because solve_magi failed or returned unexpected type ($(typeof(results)))."
             # Force failure if results is not the expected type
             @test results isa NamedTuple
        end
        println("--- End Solver Test: FN Example ---")

    end # End FN Testset

    # Add other testsets here later (e.g., Hes1, different configs)

end # End solve_magi Testset
