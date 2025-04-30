@testset "Performance Benchmarks" begin
    using BenchmarkTools
    using MagiJl.Kernels: create_rbf_kernel, create_matern52_kernel
    using MagiJl.GaussianProcess: calculate_gp_covariances!, GPCov

    # Suppress detailed benchmark output for cleaner test results
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 3
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
    
    println("\n=== Performance Benchmark Tests ===")

    @testset "Kernel Function Performance" begin
        println("\nBenchmarking kernel function evaluation:")
        
        # Test points
        x_vals = collect(-2.0:0.05:2.0)  # 81 points
        n_points = length(x_vals)
        
        # Create different kernels
        rbf_kernel = create_rbf_kernel(1.0, 1.0)
        matern52_kernel = create_matern52_kernel(1.0, 1.0)
        
        # Benchmark individual kernel evaluations
        b_rbf_single = @benchmark $rbf_kernel($(x_vals[1]), $(x_vals[end]))
        b_matern_single = @benchmark $matern52_kernel($(x_vals[1]), $(x_vals[end]))
        
        println("  RBF single evaluation: $(median(b_rbf_single).time / 1e9) seconds")
        println("  Matern52 single evaluation: $(median(b_matern_single).time / 1e9) seconds")
        
        # Benchmark kernel matrix construction
        b_rbf_matrix = @benchmark [$(rbf_kernel)(x, y) for x in $(x_vals), y in $(x_vals)]
        b_matern_matrix = @benchmark [$(matern52_kernel)(x, y) for x in $(x_vals), y in $(x_vals)]
        
        println("  RBF matrix construction ($(n_points)×$(n_points)): $(median(b_rbf_matrix).time / 1e9) seconds")
        println("  Matern52 matrix construction ($(n_points)×$(n_points)): $(median(b_matern_matrix).time / 1e9) seconds")
        
        # Relative performance comparison
        rel_single = median(b_matern_single).time / median(b_rbf_single).time
        rel_matrix = median(b_matern_matrix).time / median(b_rbf_matrix).time
        
        println("  Matern52/RBF time ratio (single eval): $(rel_single)")
        println("  Matern52/RBF time ratio (matrix): $(rel_matrix)")
    end

    @testset "GP Covariance Calculation Performance" begin
        println("\nBenchmarking GP covariance calculation:")
        
        # Test different problem sizes
        sizes = [10, 50, 100]
        
        for n in sizes
            println("\n  Problem size n = $n:")
            # Setup
            tvec = collect(0.0:0.1:(n-1)*0.1)
            
            # Test different kernels
            kernel_rbf = create_rbf_kernel(1.0, 1.0)
            kernel_matern = create_matern52_kernel(1.0, 1.0)
            phi = [1.0, 1.0]
            
            # Test different bandwidths
            for bandsize in [0, 2, n-1]
                println("    Bandsize = $bandsize:")
                
                # RBF kernel with complexity=0
                gp_cov_rbf_c0 = GPCov()
                b_rbf_c0 = @benchmark calculate_gp_covariances!(
                    $(gp_cov_rbf_c0), $(kernel_rbf), $(phi), $(tvec), $(bandsize);
                    complexity=0, jitter=1e-6
                )
                println("      RBF (complexity=0): $(median(b_rbf_c0).time / 1e9) seconds")
                
                # RBF kernel with complexity=2
                gp_cov_rbf_c2 = GPCov()
                b_rbf_c2 = @benchmark calculate_gp_covariances!(
                    $(gp_cov_rbf_c2), $(kernel_rbf), $(phi), $(tvec), $(bandsize);
                    complexity=2, jitter=1e-6
                )
                println("      RBF (complexity=2): $(median(b_rbf_c2).time / 1e9) seconds")
                
                # Matern kernel with complexity=0
                gp_cov_matern_c0 = GPCov()
                b_matern_c0 = @benchmark calculate_gp_covariances!(
                    $(gp_cov_matern_c0), $(kernel_matern), $(phi), $(tvec), $(bandsize);
                    complexity=0, jitter=1e-6
                )
                println("      Matern52 (complexity=0): $(median(b_matern_c0).time / 1e9) seconds")
                
                # Matern kernel with complexity=2
                gp_cov_matern_c2 = GPCov()
                b_matern_c2 = @benchmark calculate_gp_covariances!(
                    $(gp_cov_matern_c2), $(kernel_matern), $(phi), $(tvec), $(bandsize);
                    complexity=2, jitter=1e-6
                )
                println("      Matern52 (complexity=2): $(median(b_matern_c2).time / 1e9) seconds")
                
                # Calculate speedup of using smaller bandsize compared to full matrix
                if bandsize < n-1 && n > 10
                    speedup_rbf_c2 = median(b_rbf_c2).time / median(b_rbf_c2).time
                    println("      Bandsize $bandsize speedup vs. full matrix (RBF, complexity=2): $(speedup_rbf_c2)x")
                end
            end
        end
    end

    @testset "ODE Function Performance" begin
        println("\nBenchmarking ODE functions:")
        
        # Test ODE function evaluation
        u_test_fn = [1.0, 0.5]
        p_test_fn = [0.5, 0.6, 0.7]
        t_test = 0.0
        du_fn = similar(u_test_fn)
        
        b_fn_ode = @benchmark ODEModels.fn_ode!($(du_fn), $(u_test_fn), $(p_test_fn), $(t_test))
        println("  FN ODE evaluation: $(median(b_fn_ode).time / 1e9) seconds")
        
        # Test ODE Jacobian calculations
        J_fn = zeros(length(u_test_fn), length(u_test_fn))
        b_fn_dx = @benchmark ODEModels.fn_ode_dx!($(J_fn), $(u_test_fn), $(p_test_fn), $(t_test))
        println("  FN ODE dx Jacobian: $(median(b_fn_dx).time / 1e9) seconds")
        
        b_fn_dtheta = @benchmark ODEModels.fn_ode_dtheta($(u_test_fn), $(p_test_fn), $(t_test))
        println("  FN ODE dtheta Jacobian: $(median(b_fn_dtheta).time / 1e9) seconds")
        
        # Compare with other ODE models
        u_test_hes = [1.0, 2.0, 3.0]
        p_test_hes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        du_hes = similar(u_test_hes)
        
        b_hes_ode = @benchmark ODEModels.hes1_ode!($(du_hes), $(u_test_hes), $(p_test_hes), $(t_test))
        println("  Hes1 ODE evaluation: $(median(b_hes_ode).time / 1e9) seconds")
        
        # Compare with log-transformed models
        u_test_log = log.(u_test_hes)
        du_log = similar(u_test_log)
        
        b_log_ode = @benchmark ODEModels.hes1log_ode!($(du_log), $(u_test_log), $(p_test_hes), $(t_test))
        println("  Hes1 log-transformed ODE evaluation: $(median(b_log_ode).time / 1e9) seconds")
        
        # Compare regular vs. log-transformed ratio
        log_ratio = median(b_log_ode).time / median(b_hes_ode).time
        println("  Log-transformed/regular time ratio: $(log_ratio)")
    end

    @testset "Likelihood Function Scaling" begin
        println("\nBenchmarking likelihood function scaling:")
        
        # Setup function to create test cases of different sizes
        function create_likelihood_test(n_times, n_dims)
            tvec = collect(0.0:0.1:(n_times-1)*0.1)
            xlatent = rand(n_times, n_dims) .* 2.0 .- 1.0
            yobs = copy(xlatent) .+ randn(size(xlatent)) * 0.1
            
            # For FN ODE, we need exactly 3 parameters
            # For dimensions other than 2, we'll need a different ODE model
            if n_dims == 2
                # Use FN ODE for 2D case
                theta = [0.5, 0.6, 0.7]  # FN ODE parameters
                ode_f = ODEModels.fn_ode!
                ode_dfdx = ODEModels.fn_ode_dx!
                ode_dfdp = ODEModels.fn_ode_dtheta
            else
                # Use Hes1 ODE for other dimensions
                # Hes1 requires 3D state, so only use for n_dims == 3
                if n_dims == 3
                    theta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Hes1 parameters
                    ode_f = ODEModels.hes1_ode!
                    ode_dfdx = ODEModels.hes1_ode_dx!
                    ode_dfdp = ODEModels.hes1_ode_dtheta
                else
                    # For other dimensions, just return test data without running benchmark
                    return nothing
                end
            end
            
            sigma = fill(0.1, n_dims)
            
            kernel = create_matern52_kernel(1.0, 1.0)
            phi = [1.0, 1.0]
            bandsize = min(5, n_times-1)  # Use fixed bandsize or full matrix for small cases
            
            gp_covs = [GPCov() for _ in 1:n_dims]
            for i in 1:n_dims
                calculate_gp_covariances!(
                    gp_covs[i], kernel, phi, tvec, bandsize,
                    complexity=2, jitter=1e-6
                )
            end
            
            return xlatent, theta, sigma, yobs, gp_covs, tvec, n_times, n_dims, bandsize, ode_f, ode_dfdx, ode_dfdp
        end
        
        # Tests with varying size in different dimensions
        println("\n  Scaling with time points (fixed dims=2):")
        time_sizes = [10, 20, 30]
        time_results = zeros(length(time_sizes))
        
        for (i, n) in enumerate(time_sizes)
            test_data = create_likelihood_test(n, 2)
            if test_data === nothing
                println("    Skipping n_times=$n, n_dims=2 due to ODE model constraints")
                continue
            end
            
            x, t, s, y, g, tv, nt, nd, bs, ode_f, ode_dfdx, ode_dfdp = test_data
            
            b = @benchmark log_likelihood_and_gradient_banded(
                $x, $t, $s, $y, $g,
                $ode_f, $ode_dfdx, $ode_dfdp
            ) samples=2 evals=1
            
            time_results[i] = median(b).time / 1e9
            println("    n_times=$n, n_dims=2, bandsize=$bs: $(time_results[i]) seconds")
        end
        
        # Calculate scaling factor for time dimension
        if length(time_sizes) >= 2
            time_scaling = log(time_results[end]/time_results[1]) / 
                           log(time_sizes[end]/time_sizes[1])
            println("    Time dimension scaling factor: $(time_scaling)")
        end
        
        # Test with varying dimensions (fixed time points)
        println("\n  Scaling with dimensions (limited models):")
        # Only test dimensions that work with our ODE models
        dim_sizes = [2, 3]  # FN (2D) and Hes1 (3D)
        dim_results = zeros(length(dim_sizes))
        
        for (i, d) in enumerate(dim_sizes)
            test_data = create_likelihood_test(20, d)
            if test_data === nothing
                println("    Skipping n_times=20, n_dims=$d due to ODE model constraints")
                continue
            end
            
            x, t, s, y, g, tv, nt, nd, bs, ode_f, ode_dfdx, ode_dfdp = test_data
            
            b = @benchmark log_likelihood_and_gradient_banded(
                $x, $t, $s, $y, $g,
                $ode_f, $ode_dfdx, $ode_dfdp
            ) samples=2 evals=1
            
            dim_results[i] = median(b).time / 1e9
            println("    n_times=20, n_dims=$d, bandsize=$bs: $(dim_results[i]) seconds (using $(d == 2 ? "FN" : "Hes1") ODE)")
        end
        
        # Calculate scaling factor for dimension if we have at least 2 valid results
        valid_results = findall(x -> x > 0, dim_results)
        if length(valid_results) >= 2
            i1, i2 = valid_results[1], valid_results[end]
            dim_scaling = log(dim_results[i2]/dim_results[i1]) / 
                          log(dim_sizes[i2]/dim_sizes[i1])
            println("    Dimension scaling factor: $(dim_scaling)")
        else
            println("    Not enough valid dimension tests to calculate scaling factor")
        end
    end

    @testset "Memory Allocation Benchmarks" begin
        println("\nBenchmarking memory allocations:")
        
        # Setup medium-sized test case
        n_times = 30
        n_dims = 2
        
        tvec = collect(0.0:0.1:(n_times-1)*0.1)
        xlatent = rand(n_times, n_dims) .* 2.0 .- 1.0
        yobs = copy(xlatent) .+ randn(size(xlatent)) * 0.1
        theta = [0.5, 0.6, 0.7]
        sigma = fill(0.1, n_dims)
        
        kernel = create_matern52_kernel(1.0, 1.0)
        phi = [1.0, 1.0]
        bandsize = 5
        
        # Measure memory allocations for GP covariance calculation
        gp_cov = GPCov()
        b_gp_alloc = @benchmark calculate_gp_covariances!(
            $(gp_cov), $(kernel), $(phi), $(tvec), $(bandsize);
            complexity=2, jitter=1e-6
        )
        
        gp_alloc_bytes = median(b_gp_alloc).memory
        gp_alloc_count = median(b_gp_alloc).allocs
        println("  GP covariance calculation (n=$n_times):")
        println("    Memory allocated: $(gp_alloc_bytes/1024) KB")
        println("    Allocation count: $(gp_alloc_count)")
        
        # Create GP covs for likelihood test
        gp_covs = [GPCov() for _ in 1:n_dims]
        for i in 1:n_dims
            calculate_gp_covariances!(
                gp_covs[i], kernel, phi, tvec, bandsize,
                complexity=2, jitter=1e-6
            )
        end
        
        # Measure memory allocations for likelihood calculation
        b_ll_alloc = @benchmark log_likelihood_and_gradient_banded(
            $(xlatent), $(theta), $(sigma), $(yobs), $(gp_covs),
            ODEModels.fn_ode!, ODEModels.fn_ode_dx!, ODEModels.fn_ode_dtheta
        )
        
        ll_alloc_bytes = median(b_ll_alloc).memory
        ll_alloc_count = median(b_ll_alloc).allocs
        println("  Likelihood calculation (n=$n_times, dims=$n_dims):")
        println("    Memory allocated: $(ll_alloc_bytes/1024) KB")
        println("    Allocation count: $(ll_alloc_count)")
        
        # Memory efficiency per time point
        bytes_per_point = ll_alloc_bytes / (n_times * n_dims)
        println("    Bytes per data point: $(bytes_per_point)")
    end
    
    println("=== End Performance Benchmark Tests ===\n")
end