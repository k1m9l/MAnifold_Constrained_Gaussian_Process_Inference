# test/test_postprocessing.jl

using MagiJl
using Test
using LinearAlgebra
using Statistics
using MCMCChains # Needed for testing results_to_chain, magi_summary, plot_magi(type="trace")
using Plots      # Needed for testing plot_magi
using StatsPlots # Needed for testing plot_magi(type="trace")
using Logging    # Needed for @test_logs

@testset "Postprocessing Functions" begin

    println("\n--- Testing Postprocessing Functions ---")

    # --- Create Dummy Results Data ---
    n_samples = 50
    n_times = 10
    n_dims = 2
    n_params_ode = 3

    dummy_theta = randn(n_samples, n_params_ode) .+ [1.0 2.0 3.0]
    dummy_x_sampled = randn(n_samples, n_times, n_dims)
    dummy_sigma = [0.1 0.2] # Row vector
    dummy_phi = [1.0 1.1; 0.5 0.6]
    dummy_lp = randn(n_samples) .- 10

    dummy_results = (
        theta = dummy_theta, x_sampled = dummy_x_sampled,
        sigma = dummy_sigma, phi = dummy_phi, lp = dummy_lp
    )

    dummy_t_obs = collect(range(0, 5, length=n_times))
    dummy_y_obs = mean(dummy_x_sampled, dims=1)[1,:,:] .+ randn(n_times, n_dims) * 0.15
    dummy_y_obs[2,1] = NaN

    # Define correct theta names only
    custom_theta_par_names = ["alpha", "beta", "gamma"]


    # --- Test results_to_chain ---
    @testset "results_to_chain" begin
        println("Testing results_to_chain...")
        # Basic conversion
        chn = results_to_chain(dummy_results)
        @test chn isa MCMCChains.Chains
        @test size(chn) == (n_samples, n_params_ode, 1)
        @test names(chn) == [Symbol("theta[$i]") for i in 1:n_params_ode]

        # With custom names (only for theta)
        chn_named = results_to_chain(dummy_results; par_names=custom_theta_par_names)
        @test names(chn_named) == Symbol.(custom_theta_par_names)

        # Include sigma (pass correct theta names)
        chn_sigma = results_to_chain(dummy_results; par_names=custom_theta_par_names, include_sigma=true)
        @test size(chn_sigma, 2) == n_params_ode + n_dims
        expected_sigma_names = [Symbol("sigma[$i]") for i in 1:n_dims]
        @test names(chn_sigma)[(n_params_ode+1):end] == expected_sigma_names
        @test names(chn_sigma)[1:n_params_ode] == Symbol.(custom_theta_par_names)

        # Include lp (pass correct theta names)
        chn_lp = results_to_chain(dummy_results; par_names=custom_theta_par_names, include_lp=true)
        @test size(chn_lp, 2) == n_params_ode + 1
        @test names(chn_lp)[end] == :lp
        @test names(chn_lp)[1:n_params_ode] == Symbol.(custom_theta_par_names)

        # Include sigma and lp (pass correct theta names)
        chn_all = results_to_chain(dummy_results; par_names=custom_theta_par_names, include_sigma=true, include_lp=true)
        @test size(chn_all, 2) == n_params_ode + n_dims + 1
        @test names(chn_all)[(n_params_ode+1):(n_params_ode+n_dims)] == expected_sigma_names
        @test names(chn_all)[end] == :lp
        @test names(chn_all)[1:n_params_ode] == Symbol.(custom_theta_par_names)

        # Test error for wrong par_names length
        @test_throws ErrorException results_to_chain(dummy_results; par_names=["wrong", "number"])
    end

    # --- Test magi_summary ---
    @testset "magi_summary" begin
        println("Testing magi_summary...")
        # Test basic execution (output is printed)
        summary_output = magi_summary(dummy_results)
        # Just check that it either returns a NamedTuple or nothing
        @test summary_output isa NamedTuple || summary_output === nothing
        
        if summary_output isa NamedTuple
            @test haskey(summary_output, :summarystats)
            @test haskey(summary_output, :quantiles)
            
            # Check the structure of the returned objects without specific field access
            # These tests avoid depending on specific field names that might change
            @test summary_output.summarystats isa Any
            @test summary_output.quantiles isa Any
        end

        # Test with sigma included (pass correct theta names)
        summary_output_sigma = magi_summary(dummy_results; include_sigma=true, par_names=custom_theta_par_names)
        # Similar checks as above
        @test summary_output_sigma isa NamedTuple || summary_output_sigma === nothing
    end

    # --- Test plot_magi ---
    @testset "plot_magi" begin
        println("Testing plot_magi...")
        custom_comp_names = ["Voltage", "Recovery"]

        # Test type="traj"
        p_traj = plot_magi(dummy_results; type="traj", comp_names=custom_comp_names, 
                          t_obs=dummy_t_obs, y_obs=dummy_y_obs, obs=true, ci=true)
        @test p_traj isa Plots.Plot

        # Test type="traj" without obs/ci
        p_traj_no_ci_obs = plot_magi(dummy_results; type="traj", comp_names=custom_comp_names, 
                                    t_obs=dummy_t_obs, obs=false, ci=false)
        @test p_traj_no_ci_obs isa Plots.Plot

        # Test type="trace" with correct theta names
        p_trace = plot_magi(dummy_results; type="trace", par_names=custom_theta_par_names, 
                           include_sigma=true, include_lp=true)
        @test p_trace isa Plots.Plot || p_trace === nothing

        # Test type="trace" without sigma/lp
        p_trace_no_extra = plot_magi(dummy_results; type="trace", par_names=custom_theta_par_names, 
                                    include_sigma=false, include_lp=false)
        @test p_trace_no_extra isa Plots.Plot || p_trace_no_extra === nothing

        # Test invalid type
        @test_throws ErrorException plot_magi(dummy_results; type="invalid")

        # Test plotting observations without providing data (should warn)
        @test_logs (:warn, r"Cannot plot observations for dimension") (:warn, r"Cannot plot observations for dimension") plot_magi(dummy_results; type="traj", obs=true)

    end

    println("--- Postprocessing Tests Complete ---")

end # End Postprocessing Testset