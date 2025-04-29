# test/test_ode_models.jl

using Test
using MagiJl # Still needed for the ODE value functions if exported directly
using LinearAlgebra

@testset "ODE Models" begin
    t = 0.0 # Define time once

    # --- ODE Value Tests (Keep using MagiJl. prefix as it worked before) ---

    @testset "fn_ode! Value" begin
        u = [1.0, 2.0]; p = (0.5, 0.6, 0.7); du_actual = similar(u)
        du_expected = [5.6/3.0, -17.0/7.0]
        MagiJl.fn_ode!(du_actual, u, p, t) # Assuming this worked before
        @test du_actual ≈ du_expected
    end

    @testset "hes1_ode! Value" begin
         u = [1.0, 2.0, 3.0]; p = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
         du_actual = similar(u); du_expected = [-0.2, -0.55, -2.1]
         MagiJl.hes1_ode!(du_actual, u, p, t) # Assuming this worked before
         @test du_actual ≈ du_expected
    end

    # ... include all other ODE value testsets, using MagiJl. prefix ...

    @testset "hes1log_ode! Value" begin
        u = [log(1.0), log(2.0), log(3.0)]
        p = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        du_actual = similar(u)
        du_expected = [-0.2, -0.275, -0.7]
        MagiJl.hes1log_ode!(du_actual, u, p, t)
        @test du_actual ≈ du_expected
    end

    @testset "hes1log_ode_fixg! Value" begin
        u = [log(1.0), log(2.0), log(3.0)]
        p = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        du_actual = similar(u)
        du_expected = [-0.2, -0.275, -0.3]
        MagiJl.hes1log_ode_fixg!(du_actual, u, p, t)
        @test du_actual ≈ du_expected
    end

     @testset "hes1log_ode_fixf! Value" begin
        u = [log(1.0), log(2.0), log(3.0)]
        p = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
        du_actual = similar(u)
        du_expected = [-0.2, -0.275, 7.6/3.0]
        MagiJl.hes1log_ode_fixf!(du_actual, u, p, t)
        @test du_actual ≈ du_expected
    end

    @testset "hiv_ode! Value" begin
        p = (10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        u = log.([1000.0, 100.0, 50.0, 20.0])
        du_actual = similar(u)
        du_expected = [9.99983, 1.001, 1.001, 1.001]
        MagiJl.hiv_ode!(du_actual, u, p, t)
        @test du_actual ≈ du_expected
    end

    @testset "ptrans_ode! Value" begin
        p = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0)
        u = [10.0, 1.0, 5.0, 1.0, 2.0]
        du_actual = similar(u)
        du_expected = [-10.7, 1.0, -28.1/3.0, 9.3, 0.2/3.0]
        MagiJl.ptrans_ode!(du_actual, u, p, t)
        @test du_actual ≈ du_expected
    end


    # --- Jacobian Tests (Using Full Qualification) ---

    @testset "fn_ode Jacobians" begin
        u = [1.0, 2.0]          # V, R
        p = (0.5, 0.6, 0.7)     # a, b, c
        V, R = u
        a, b, c = p

        # Test dF/dX (in-place)
        J_actual = zeros(2, 2)
        # --- Use Full Qualification ---
        MagiJl.ODEModels.fn_ode_dx!(J_actual, u, p, t)
        # -----------------------------
        J_expected = [c*(1-V^2)  c;
                     -1/c       -b/c]
        @test J_actual ≈ J_expected

        # Test dF/dTheta (returns matrix)
        # --- Use Full Qualification ---
        Jp_actual = MagiJl.ODEModels.fn_ode_dtheta(u, p, t)
        # -----------------------------
        Jp_expected = [0.0  0.0  (V - V^3/3.0 + R);
                       1/c  -R/c  (V - a + b*R)/c^2]
        @test Jp_actual ≈ Jp_expected
    end

    @testset "hes1_ode Jacobians" begin
        u = [1.0, 2.0, 3.0]      # P, M, H
        p = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7) # p1..p7
        P, M, H = u
        p1, p2, p3, p4, p5, p6, p7 = p
        one_plus_Psq = 1 + P^2

        # Test dF/dX (in-place)
        J_actual = zeros(3, 3)
        # --- Use Full Qualification ---
        MagiJl.ODEModels.hes1_ode_dx!(J_actual, u, p, t)
        # -----------------------------
        J_expected = [
            -p1*H-p3       p2           -p1*P;
            -p5*2*P/one_plus_Psq^2  -p4  0.0;
            -p1*H-p6*2*P/one_plus_Psq^2  0.0  -p1*P-p7
        ]
        @test J_actual ≈ J_expected

        # Test dF/dTheta (returns matrix)
        # --- Use Full Qualification ---
        Jp_actual = MagiJl.ODEModels.hes1_ode_dtheta(u, p, t)
        # -----------------------------
        Jp_expected = [
            -P*H  M  -P  0.0  0.0  0.0  0.0;
             0.0  0.0 0.0 -M   1/one_plus_Psq 0.0 0.0;
            -P*H  0.0 0.0 0.0  0.0  1/one_plus_Psq -H
        ]
        @test Jp_actual ≈ Jp_expected
    end

    # TODO: Add testsets for the Jacobians of the other ODEs
    # using the MagiJl.ODEModels. prefix for the function calls

end # End ODE Models Testset

@testset "ODE Model Integration" begin
    @testset "FitzHugh-Nagumo System" begin
        # Import required packages
        using DifferentialEquations
        using MagiJl.ODEModels: fn_ode!
        
        # Basic test - verify ODE function produces expected outputs
        du_test = zeros(2)
        u_test = [1.0, 0.5]
        p_test = [0.5, 0.6, 0.7]  # a, b, c
        t_test = 0.0
        
        # Call the ODE function manually
        fn_ode!(du_test, u_test, p_test, t_test)
        
        # Calculate expected result by hand
        expected_du1 = p_test[3] * (u_test[1] - (u_test[1]^3) / 3.0 + u_test[2])
        expected_du2 = -1.0/p_test[3] * (u_test[1] - p_test[1] + p_test[2] * u_test[2])
        
        # Test that ODE function produces correct values
        @test isapprox(du_test[1], expected_du1, rtol=1e-10)
        @test isapprox(du_test[2], expected_du2, rtol=1e-10)
        
        # Test simple integration - just verify it runs without errors
        u0 = [1.0, 0.0]
        tspan = (0.0, 1.0)
        prob = ODEProblem(fn_ode!, u0, tspan, p_test)
        sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6)
        
        # Basic property test - solution should have values at start and end
        @test length(sol.t) > 2
        @test sol(0.0) ≈ u0 rtol=1e-6
        
        # Test consistency between ODE function and integration
        # at a few sample points
        t_mid = 0.5
        u_mid = sol(t_mid)
        
        # Direct calculation
        du_mid = zeros(2)
        fn_ode!(du_mid, u_mid, p_test, t_mid)
        
        # Numerical derivative from solution
        t_delta = 0.01
        u_forward = sol(t_mid + t_delta)
        u_backward = sol(t_mid - t_delta)
        du_numerical = (u_forward - u_backward) / (2 * t_delta)
        
        # Verify consistency with more generous tolerance (numerical differentiation isn't exact)
        @test isapprox(du_mid, du_numerical, rtol=0.1)
    end
end