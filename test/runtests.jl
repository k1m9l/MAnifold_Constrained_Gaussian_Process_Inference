# test/runtests.jl

# Load the main module being tested
using MagiJl

# Load testing utilities
using Test
using TestSetExtensions

# Define the main test suite for the entire package
# Using ExtendedTestSet for enhanced output (dots, diffs on failure)
@testset ExtendedTestSet "MagiJl.jl Tests" begin

    println("Running End-to-End Tests...")
    @testset "End-to-End Tests" begin
        # Include the end-to-end test for the solver
        include("test_solver.jl")
    end

    # Run tests for the ODE model definitions
    println("\nRunning ODE Model Tests...")
    @testset "ODE Models Sub-Tests" begin # Optional inner testset for grouping
        include("test_ode_models.jl")
    end

    # Run tests for the kernel function implementations/usage
    println("\nRunning Kernel Tests...")
    @testset "Kernel Functions Sub-Tests" begin # Optional inner testset
       include("test_kernels.jl") # Assumes this file exists and contains kernel tests
    end

    # Run tests for Gaussian Process utility functions
    println("\nRunning GP Utils Tests...")
    @testset "GP Utilities Sub-Tests" begin # Optional inner testset
        include("test_gp_utils.jl")
    end

    # Run tests for the GPCov struct and its calculation
    println("\nRunning GP Covariance Tests...")
    @testset "GP Covariance Sub-Tests" begin # Optional inner testset
        include("test_gp.jl")
    end

    # Run tests for the log-likelihood calculation
    println("\nRunning Likelihood Tests...")
    @testset "Likelihood Sub-Tests" begin # Optional inner testset
        include("test_likelihoods.jl")
    end

    # Run tests for the sampler integration
    println("\nRunning Sampler Tests...")
    @testset "Sampler Sub-Tests" begin # Optional inner testset
       include("test_samplers.jl") # Include the new sampler tests
    end

    # Placeholder for solver tests
    # println("\nRunning Solver Tests...")
    # @testset "Solver Sub-Tests" begin
    #    include("test_solver.jl")
    # end

    # Add the performance tests, but make them optional
    if get(ENV, "MAGI_RUN_PERFORMANCE_TESTS", "false") == "true"
        println("\nRunning Performance Tests...")
        @testset "Performance Tests" begin
            include("test_performance.jl")
        end
    else
        @info "Skipping performance tests. Set MAGI_RUN_PERFORMANCE_TESTS=true to run them."
    end

end # End main MagiJl.jl testset