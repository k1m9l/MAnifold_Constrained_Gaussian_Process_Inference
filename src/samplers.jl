# src/samplers.jl
module Samplers

using AdvancedHMC
using LogDensityProblems
using ..LogDensityProblemsInterface: MagiTarget # Assumes MagiTarget implements the LogDensityProblems interface
using LinearAlgebra
using Statistics
using Random

export run_nuts_sampler

"""
    logdensity_func_wrapper(target, θ)

Helper function to wrap the target log-density calculation.

This function serves as a simple interface adapter, calling the required
`LogDensityProblems.logdensity` method on the `target` object (e.g., `MagiTarget`).
It ensures compatibility with the function signature expected by `AdvancedHMC.jl`.

# Arguments
- `target`: An object implementing the `LogDensityProblems` interface (e.g., `MagiTarget`).
- `θ`: The parameter vector at which to evaluate the log-density.

# Returns
- The log-density value `log π(θ)`.
"""
function logdensity_func_wrapper(target, θ)
    # Calls the logdensity method defined for the specific target type
    return LogDensityProblems.logdensity(target, θ)
end

"""
    logdensity_and_gradient_func_wrapper(target, θ)

Helper function to wrap the target log-density AND gradient calculation.

This function calls the required `LogDensityProblems.logdensity_and_gradient`
method on the `target` object. It also includes basic validation checks
to ensure the returned log-density (`val`) and gradient (`grad`) are finite
and have the expected types, which is crucial for the stability of HMC.

# Arguments
- `target`: An object implementing the `LogDensityProblems` interface (order 1).
- `θ`: The parameter vector at which to evaluate the log-density and gradient.

# Returns
- A tuple `(value, gradient)`:
    - `value`: The log-density value `log π(θ)`.
    - `gradient`: The gradient vector `∇log π(θ)`.
"""
function logdensity_and_gradient_func_wrapper(target, θ)
    # Calls the logdensity_and_gradient method defined for the specific target type
    val, grad = LogDensityProblems.logdensity_and_gradient(target, θ)

    # Basic validation to catch potential issues during HMC steps
    @assert val isa Real && isfinite(val) "Log density is not a finite Real! Value: $val"
    @assert grad isa AbstractVector{<:Real} "Gradient is not a Vector! Type: $(typeof(grad))"
    @assert all(isfinite, grad) "Gradient contains non-finite values! Grad: $grad"

    return val, grad
end

"""
    run_nuts_sampler(
        target::MagiTarget,
        initial_params::Vector{Float64};
        n_samples::Int = 2000,
        n_adapts::Int = 1000,
        target_accept_ratio = 0.8,
        initial_step_size = 0.1
    )

Runs the No-U-Turn Sampler (NUTS), a variant of Hamiltonian Monte Carlo (HMC),
using the `AdvancedHMC.jl` library to sample from the posterior distribution
defined by the `target`.

HMC simulates Hamiltonian dynamics to propose moves in the parameter space.
It uses the gradient of the log-posterior (provided by `target`) to guide proposals,
allowing for more efficient exploration than random-walk methods, especially in
high dimensions. NUTS automatically tunes the simulation length.

# Arguments
- `target::MagiTarget`: The target distribution object, implementing the
    `LogDensityProblems` interface. Provides `log π(θ)` and `∇log π(θ)`,
    which define the potential energy landscape for HMC.
- `initial_params::Vector{Float64}`: The starting point (initial parameter vector `θ₀`)
    for the MCMC chain in the parameter space.
- `n_samples::Int`: Total number of MCMC samples to generate (including adaptation/warmup).
- `n_adapts::Int`: Number of initial "adaptation" or "warmup" steps used by the
    `StanHMCAdaptor` to tune the step size (`ϵ`) and potentially the mass matrix.
    These steps are typically discarded.
- `target_accept_ratio`: The desired average acceptance probability for the
    Metropolis-Hastings step within HMC. The `StepSizeAdaptor` adjusts `ϵ`
    during warmup to try and achieve this ratio (common values are 0.65 to 0.9).
- `initial_step_size`: The initial leapfrog step size `ϵ` for the numerical integrator.
    This value is adapted during warmup.

# Returns
- `Tuple{Vector, Any}`: A tuple `(chain, stats)` containing:
    - `chain`: A vector of MCMC samples (parameter vectors `θ`) drawn from the
      posterior distribution *after* the warmup phase.
    - `stats`: A collection of statistics gathered during sampling (e.g., acceptance rate,
      step size, tree depth for NUTS). The exact structure depends on `AdvancedHMC.jl`.

# Notes
- Assumes `target` provides the log-posterior density (up to a constant).
- The chain returned contains only post-warmup samples (`drop_warmup=true`).
- A diagonal Euclidean metric (identity mass matrix scaled during adaptation)
  is used for the kinetic energy term in the Hamiltonian dynamics.
- Step size `ϵ` is adapted using `StanHMCAdaptor` during the warmup phase.
"""
function run_nuts_sampler(
        target::MagiTarget,
        initial_params::Vector{Float64}; # θ₀
        n_samples::Int = 20000,           # Total iterations
        n_adapts::Int = 10000,            # Warmup iterations
        target_accept_ratio = 0.8,       # Target for MH acceptance
        initial_step_size = 0.1          # Initial integrator step size ϵ₀
    )

    # Get the total dimensionality of the parameter space (e.g., dim(θ))
    n_dims_total = LogDensityProblems.dimension(target)
    @assert length(initial_params) == n_dims_total "Initial parameters dimension mismatch"

    # Declare variables to hold results outside the try block
    local chain, stats

    try
        # --- HMC Setup ---

        # 1. Define Log-density and Gradient Functions for HMC
        #    These closures capture the specific `target` instance.
        #    logπ(θ) -> Potential Energy U(θ) = -log π(θ) (up to constant)
        #    ∂logπ∂θ(θ) -> Gradient of log density ∇log π(θ) = -∇U(θ)
        logπ = (θ) -> logdensity_func_wrapper(target, θ)
        ∂logπ∂θ = (θ) -> logdensity_and_gradient_func_wrapper(target, θ)

        # 2. Define the Metric (Mass Matrix) for Kinetic Energy
        #    Specifies the geometry of the parameter space for HMC.
        #    DiagEuclideanMetric corresponds to a diagonal mass matrix M (initially I).
        #    Kinetic Energy K(p) = 0.5 * pᵀ * M⁻¹ * p, where p is momentum.
        metric = DiagEuclideanMetric(n_dims_total)

        # 3. Create the Hamiltonian
        #    Combines the potential energy (from logπ) and kinetic energy (from metric).
        #    H(θ, p) = U(θ) + K(p)
        hamiltonian = Hamiltonian(metric, logπ, ∂logπ∂θ)

        # 4. Create the Integrator and Trajectory (NUTS specific)
        #    - Leapfrog: Numerical integrator to simulate Hamilton's equations.
        #      Uses step size ϵ.
        #    - GeneralisedNoUTurn: The NUTS criterion for dynamically building
        #      the simulation trajectory and determining when to stop.
        #    - Trajectory: Combines integrator and termination criterion.
        init_ϵ = initial_step_size # Initial step size ϵ₀
        integrator = Leapfrog(init_ϵ)
        trajectory = Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()) # NUTS trajectory
        kernel = HMCKernel(trajectory) # The HMC transition kernel using the NUTS trajectory

        # 5. Setup Adaptation
        #    Uses standard Stan algorithms to adapt:
        #    - MassMatrixAdaptor: Tunes the diagonal elements of the metric (mass matrix M).
        #    - StepSizeAdaptor: Tunes the leapfrog step size ϵ to meet the target_accept_ratio.
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), # Adapts M
                              StepSizeAdaptor(target_accept_ratio, init_ϵ)) # Adapts ϵ

        # --- Run Sampler ---
        # Uses the defined Hamiltonian, NUTS kernel, initial parameters,
        # total sample count, adaptor configuration, and warmup count.
        rng = Random.default_rng() # Default random number generator
        chain, stats = sample(
            rng,
            hamiltonian,
            kernel,
            initial_params, # θ₀
            n_samples,      # Total iterations
            adaptor,        # Configuration for adapting M and ϵ
            n_adapts;       # Number of warmup steps
            drop_warmup=true, # Discard adaptation samples from the final chain
            progress=true,    # Show progress bar
            verbose=false     # Reduce console output from AdvancedHMC
        )

    catch e
        # Catch potential errors during HMC setup or sampling
        @error "ERROR in NUTS sampler!" exception=(e, catch_backtrace())
        return nothing, nothing # Return nothing to indicate failure
    end

    # Return the posterior samples and associated statistics
    return chain, stats
end

end # module Samplers
