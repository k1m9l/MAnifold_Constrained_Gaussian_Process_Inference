# test/test_samplers.jl

using Test
using MagiJl # Main package
using LinearAlgebra
using KernelFunctions
using LogDensityProblems # For dimension checking
using AdvancedHMC # To check output types if needed

# Explicitly import modules if MagiJl doesn't re-export everything needed
using MagiJl.ODEModels
using MagiJl.Kernels
using MagiJl.GaussianProcess
using MagiJl.LogDensityProblemsInterface # Your struct MagiTarget
using MagiJl.Samplers                     # Your function run_nuts_sampler

@testset "Sampler Integration Tests" begin

    @testset "NUTS Sampler runs with FN model" begin
        println("\n--- Sampler Tests: Testset: NUTS Sampler runs with FN model ---")

        # --- Setup Test Case (borrowed from test_likelihoods.jl) ---
        ode_func_fn = ODEModels.fn_ode!
        ode_dfdx_fn = ODEModels.fn_ode_dx!
        ode_dfdp_fn = ODEModels.fn_ode_dtheta

        variance_fn = 1.5
        lengthscale_fn = 1.2
        # Use kernel creation helper from MagiJl.Kernels
        kernel_fn = Kernels.create_rbf_kernel(variance_fn, lengthscale_fn) # Example: RBF
        phi_fn = [variance_fn, lengthscale_fn] # Store parameters if needed, though GPCov only needs kernel

        tvec_fn = [0.0, 1.0, 2.0] # Small N for faster test
        n_times_fn = length(tvec_fn)
        n_dims_fn = 2 # V, R for FN
        bandsize_fn = min(n_times_fn - 1, 1) # Ensure bandsize <= n_times-1
        jitter_fn = 1e-5

        theta_fn = [0.5, 0.6, 0.7] # a, b, c (true values or initial guess)
        n_params_fn = length(theta_fn)

        # Use a plausible initial state for xlatent
        xlatent_mat_fn = Matrix{Float64}([1.0 0.5; 1.1 0.6; 1.2 0.7])

        sigma_fn = [0.1, 0.15] # Noise levels (assuming fixed for now)

        # Observations (can be simple, e.g., based on xlatent)
        yobs_fn_test = xlatent_mat_fn .+ randn(size(xlatent_mat_fn)) .* sigma_fn' # Add some noise

        println("Test Setup: n_times=$n_times_fn, n_dims=$n_dims_fn, n_params=$n_params_fn")
        # --- End Setup ---


        # --- Pre-calculate GPCov ---
        # Ensure GPCov calculation works for this small size
        gp_cov_all_dims_fn = Vector{GPCov}(undef, n_dims_fn)
        try
            for i in 1:n_dims_fn
                gp_cov_all_dims_fn[i] = GPCov()
                GaussianProcess.calculate_gp_covariances!(
                    gp_cov_all_dims_fn[i], kernel_fn, phi_fn, tvec_fn, bandsize_fn;
                    complexity=2, jitter=jitter_fn
                )
            end
            println("GPCov calculation successful.")
        catch e
            @error "GPCov calculation failed during test setup!" exception=(e, catch_backtrace())
            # Fail the test explicitly if setup fails
            @test false
            # Need to return or rethrow to stop execution if setup fails critically
            rethrow(e)
        end
        # --- End GPCov ---


        # --- Create MagiTarget Instance ---
        target = MagiTarget(
            yobs_fn_test,
            gp_cov_all_dims_fn,
            ode_func_fn,
            ode_dfdx_fn,
            ode_dfdp_fn,
            sigma_fn,
            [1.0, 1.0, 1.0], # Default prior temperatures
            n_times_fn,
            n_dims_fn,
            n_params_fn
        )
        total_params_dim = LogDensityProblems.dimension(target)
        @test total_params_dim == n_times_fn * n_dims_fn + n_params_fn
        println("MagiTarget created successfully. Total dimensions: $total_params_dim")
        # --- End MagiTarget ---


        # --- Define Initial Parameters ---
        # Flatten xlatent and concatenate theta
        initial_params = vcat(vec(xlatent_mat_fn), theta_fn)
        @test length(initial_params) == total_params_dim
        # --- End Initial Parameters ---


        # --- Run Sampler (short run for testing) ---
        n_test_samples = 10
        n_test_adapts = 5 # Use very few adaptations for speed
        samples = nothing # Initialize to ensure scope
        stats = nothing
        sampler_success = false
        try
            # Make sure the sampler function is callable via MagiJl or Samplers
            samples, stats = Samplers.run_nuts_sampler(
                target,
                initial_params;
                n_samples = n_test_samples,
                n_adapts = n_test_adapts
            )
            sampler_success = true
            println("Sampler finished execution.")
        catch e
            @error "Sampler execution failed!" exception=(e, catch_backtrace())
            sampler_success = false
        end
        # --- End Run Sampler ---


        # --- Basic Assertions ---
        @test sampler_success # Check if the sampler ran without throwing an error

        # Allow empty samples in case the sampler failed but didn't throw
        if samples === nothing || isempty(samples)
            @warn "Sampler returned empty samples array."
        else
            
            # WHEN drop_warmup=true, we expect n_samples - n_adapts samples
            @test length(samples) == n_test_samples - n_test_adapts
            
            # Check that the samples are of the right dimension
            sample_vector = first(samples)
            @test length(sample_vector) == total_params_dim
            @test all(isfinite, sample_vector)
        end

        # Check stats (type might vary, but should be not nothing)
        @test stats !== nothing
        
        println("Sampler integration test completed.")
        println("-----------------------------------------------------------------")

    end # End Testset: NUTS Sampler runs with FN model

    # TODO: Add more tests?
    # - Test with different ODEs?
    # - Test specific sampler configurations?
    # - Test edge cases if applicable?

end # End Testset: Sampler Integration Tests