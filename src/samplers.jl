# src/samplers.jl
module Samplers

using AdvancedHMC
using LogDensityProblems
using ..LogDensityProblemsInterface: MagiTarget
using LinearAlgebra
using Statistics
using Random

export run_nuts_sampler

"""
Helper function to wrap the target log-density method.
"""
function logdensity_func_wrapper(target, θ)
    return LogDensityProblems.logdensity(target, θ)
end

"""
Helper function to wrap the target log-density AND gradient method.
Includes basic validation to catch potential issues.
"""
function logdensity_and_gradient_func_wrapper(target, θ)
    val, grad = LogDensityProblems.logdensity_and_gradient(target, θ)
    # Basic validation to catch issues
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

Runs the No-U-Turn Sampler (NUTS) using AdvancedHMC.jl.

# Arguments
- `target`: MagiTarget object implementing the LogDensityProblems interface
- `initial_params`: Starting point for the MCMC chain
- `n_samples`: Total number of MCMC samples to generate
- `n_adapts`: Number of adaptation steps for tuning the sampler
- `target_accept_ratio`: Target acceptance rate for step size adaptation
- `initial_step_size`: Initial leapfrog step size

# Returns
A tuple `(chain, stats)` with the MCMC samples and sampling statistics

# Notes
- The chain includes only post-warmup samples (warmup samples are discarded)
- Diagonal mass matrix is used for the Hamiltonian dynamics
- Step size is adapted during the warmup phase
"""
function run_nuts_sampler(
        target::MagiTarget,
        initial_params::Vector{Float64};
        n_samples::Int = 2000,
        n_adapts::Int = 1000,
        target_accept_ratio = 0.8,
        initial_step_size = 0.1
    )

    n_dims_total = LogDensityProblems.dimension(target)
    @assert length(initial_params) == n_dims_total "Initial parameters dimension mismatch"

    local chain, stats

    try
        # Create closures bound to the specific target instance
        logπ = (θ) -> logdensity_func_wrapper(target, θ)
        ∂logπ∂θ = (θ) -> logdensity_and_gradient_func_wrapper(target, θ)

        # Create metric
        metric = DiagEuclideanMetric(n_dims_total)

        # Create Hamiltonian
        hamiltonian = Hamiltonian(metric, logπ, ∂logπ∂θ)

        # Create Integrator and Kernel
        init_ϵ = initial_step_size
        integrator = Leapfrog(init_ϵ)
        trajectory = Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn())
        kernel = HMCKernel(trajectory)

        # Setup adaptation
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric),
                              StepSizeAdaptor(target_accept_ratio, init_ϵ))

        # Run sampling
        rng = Random.default_rng()
        chain, stats = sample(rng, hamiltonian, kernel, initial_params, n_samples, adaptor, n_adapts;
                            drop_warmup=true, progress=true, verbose=false)

    catch e
        @error "ERROR in NUTS sampler!" exception=(e, catch_backtrace())
        return nothing, nothing # Return nothing on error
    end

    return chain, stats
end

end # module Samplers