# MagiJl Project - Progress Summary

**Date:** April 29, 2025

**Project Goal:** To reimplement the C++ MAGI (MAnifold Gaussian process Inference for systems of ordinary differential equations) algorithm in the Julia programming language, aiming to maintain the core logic and performance characteristics, particularly the banded matrix optimizations.

**Initial Language Choice Discussion:**

* Considered PyTorch, Julia, and JAX.
* Key challenge identified: Efficiently handling banded matrix operations (matrix-vector products, solves) as done in the C++ code using BLAS/LAPACK (`dgbmv_`) for Gaussian Process calculations, which is crucial for performance and scalability ($O(Nb)$ or $O(Nb^2)$ complexity vs. $O(N^2)$ or $O(N^3)$ for dense).
* PyTorch/JAX: Would likely require custom C++/FFI extensions to call optimized BLAS routines or careful manual implementation for banded matrix-vector products, adding complexity. `jax.scipy.linalg.solve_banded` exists, but no direct equivalent for `dgbmv_`.
* Julia: Chosen due to its high performance for scientific computing, excellent `DifferentialEquations.jl` package, and strong native support for specialized linear algebra, including the `BandedMatrices.jl` package, making it the most direct path to replicating the C++ banded matrix optimizations.

**Project Setup:**

* A standard Julia package named `MagiJl` was generated using `Pkg.generate`.
* The project uses Julia's package manager (`Pkg.jl`) with `Project.toml` and `Manifest.toml` for dependency management within a project-specific environment.
* Standard directory structure is used: `src/`, `test/`.

**Implemented Components (`src/` directory):**

1.  **`ode_models.jl`**:
    * Defines the `ODEModels` module.
    * Contains Julia translations of all ODE systems from `dynamicalSystemModels.cpp` (`fn_ode!`, `hes1_ode!`, `hes1log_ode!`, `hes1log_ode_fixg!`, `hes1log_ode_fixf!`, `hiv_ode!`, `ptrans_ode!`).
    * Functions use the in-place signature `f!(du, u, p, t)` suitable for `DifferentialEquations.jl`.
    * Exports the defined ODE functions.

2.  **`kernels.jl`**:
    * Defines the `Kernels` module.
    * Uses `KernelFunctions.jl` to define standard kernels (e.g., `SqExponentialKernel`, `Matern52Kernel`).
    * Includes helper functions (e.g., `create_rbf_kernel`) that apply variance and lengthscale using the composition syntax (`KernelType() ∘ ScaleTransform(1/lengthscale)`), which was found to work reliably during testing.
    * Exports helper function names.

3.  **`gaussian_process.jl`**:
    * Defines the `GaussianProcess` module.
    * Defines the `GPCov` mutable struct to hold GP covariance information (dense and banded matrices, parameters, etc.), analogous to the C++ `gpcov` struct.
    * Implements `calculate_gp_covariances!`:
        * Calculates the dense covariance matrix `C` using `KernelFunctions.kernelmatrix`.
        * Adds jitter and calculates the inverse `Cinv` using `cholesky(PositiveFactorizations.Positive, ...)`.
        * **Limitation:** Currently initializes time derivatives (`Cprime`, `Cdoubleprime`) and derived matrices (`mphi`, `Kphi`, `Kinv`) with placeholders (zeros or jittered identity). The actual calculation based on kernel derivatives is **not yet implemented**.
        * Implements `mat2band` helper and correctly converts `Cinv`, `mphi`, `Kinv` to `BandedMatrix` types using `BandedMatrices.jl`.
    * Exports `GPCov` and `calculate_gp_covariances!`.

4.  **`likelihoods.jl`**:
    * Defines the `Likelihoods` module.
    * Implements `log_likelihood_banded`:
        * Calculates the ODE trajectory (`fderiv`) based on input states (`xlatent`) and parameters (`theta`).
        * Calculates the log-likelihood *value* using efficient banded matrix operations (relying on `BandedMatrices.jl` overloading `*` for `BandedMatrix * Vector`).
        * Handles missing observations (`NaN` in `yobs`).
        * Includes prior temperature scaling.
        * Includes normalization constants for the Gaussian likelihood terms.
        * **Limitation:** Gradient calculation (`d(loglik)/d(xtheta)`) is explicitly marked as **TODO**.
    * Exports `log_likelihood_banded`.

5.  **`MagiJl.jl`**:
    * Main module file.
    * Correctly uses `include` for all implemented source files (`ode_models.jl`, `kernels.jl`, `gaussian_process.jl`, `likelihoods.jl`).
    * Uses `using .SubModuleName` to bring submodules into scope.
    * Exports relevant functions and types (`fn_ode!`, ..., `GPCov`, `calculate_gp_covariances!`, `log_likelihood_banded`, etc.).

**Testing Setup (`test/` directory):**

* Uses `Test.jl` (standard library) and `TestSetExtensions.jl` (for enhanced output with progress dots).
* `runtests.jl`: Main test runner, uses `@testset ExtendedTestSet`, includes specific test files.
* `test_ode_models.jl`: Contains passing tests for all implemented ODE functions. Uses explicit qualification (`MagiJl.ode_func!`) to call functions from the main module.
* Kernel tests are currently integrated directly within `runtests.jl` (after troubleshooting `include` issues) and verify `KernelFunctions.jl` usage.
* `test_gp.jl`: Contains passing tests for `GPCov` struct creation, field population (including checks on `C`, `Cinv`, placeholder checks for `Kinv`/`mphi`), positive definiteness, inverse correctness, and banded matrix consistency.
* `test_likelihoods.jl`: Contains basic passing tests ensuring `log_likelihood_banded` runs and returns a finite `Float64`.
* **Overall Status:** All implemented tests (currently 104) are passing.

**Dependencies Added:**

* `DifferentialEquations` (Implicitly needed for ODE function compatibility)
* `KernelFunctions`
* `BandedMatrices`
* `PositiveFactorizations`
* `TestSetExtensions`
* `LinearAlgebra` (Standard Library)
* `Test` (Standard Library)

**Troubleshooting Notes:**

* Initial `UndefVarError` issues during testing were resolved by:
    * Ensuring functions were correctly exported from submodules (`ODEModels`, `Kernels`).
    * Using explicit qualification (e.g., `MagiJl.fn_ode!`) in test files.
    * Ensuring necessary `using` statements were present in the correct scope (either test file or `runtests.jl`).
    * Using the correct function call for `PositiveFactorizations` (`cholesky(PositiveFactorizations.Positive, ...)` instead of non-existent `pchol`).
    * Moving kernel tests temporarily into `runtests.jl` helped diagnose scoping issues with `include`. (Tests have since been modularized again).
    * Restarting Julia / Re-instantiating the environment helped clear stale states.

**Next Steps:**

1.  **Implement Time Derivatives:** Calculate `Cprime` and `Cdoubleprime` in `calculate_gp_covariances!` (`src/gaussian_process.jl`) based on the specific kernel formulas. This is needed for `complexity=2`.
2.  **Implement Log-Likelihood Gradient:** Calculate the gradient `d(loglik)/d(xtheta)` in `log_likelihood_banded` (`src/likelihoods.jl`). This requires using ODE Jacobians and transposed banded matrix operations.
3.  **Implement Sampler:** Implement the HMC/NUTS logic (e.g., in `src/samplers.jl`), potentially leveraging packages like `AdvancedHMC.jl` or `DynamicHMC.jl`, using the calculated log-likelihood and gradient.
4.  **Implement Main Solver:** Create the main `solve_magi` function (`src/solver.jl` or `MagiJl.jl`) to orchestrate the setup, sampling, and result processing.
5.  **Add Tests:** Create tests for the gradient calculation, sampler, and the main solver.
# MAGI
