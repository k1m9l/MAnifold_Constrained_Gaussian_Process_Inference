# In src/samplers.jl
module Samplers

using AdvancedHMC
using LogDensityProblems
using ..LogDensityProblemsInterface: MagiTarget # Your wrapper struct
using LinearAlgebra # For Diagonal metric
using Statistics # For std deviation calculation if needed for init
using Random # To ensure RNG access if needed

export run_nuts_sampler

# Helper function to wrap the target log-density method
function logdensity_func_wrapper(target, θ)
     return LogDensityProblems.logdensity(target, θ)
end

# Helper function to wrap the target log-density AND gradient method
function logdensity_and_gradient_func_wrapper(target, θ)
    # This function MUST return the tuple (log_density, gradient)
    val, grad = LogDensityProblems.logdensity_and_gradient(target, θ)
    # Perform checks if desired
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
        initial_step_size = 0.1 # Step size to use
    )

Runs the NUTS sampler using AdvancedHMC.jl (Adjusted for Hamiltonian constructor).
"""
function run_nuts_sampler(
        target::MagiTarget,
        initial_params::Vector{Float64};
        n_samples::Int = 2000,
        n_adapts::Int = 1000,
        target_accept_ratio = 0.8,
        initial_step_size = 0.1 # Step size to use directly
    )

    n_dims_total = LogDensityProblems.dimension(target)
    @assert length(initial_params) == n_dims_total "Initial parameters dimension mismatch"

    println("Starting NUTS sampler setup (Explicit Kernel, Corrected Hamiltonian)...")

    local chain, stats # Ensure scope outside try block

    try
        # Create closures bound to the specific target instance
        # Function for log-density ONLY
        logπ = (θ) -> logdensity_func_wrapper(target, θ)
        # Function for log-density AND gradient
        ∂logπ∂θ = (θ) -> logdensity_and_gradient_func_wrapper(target, θ)

        # 1. Create metric
        metric = DiagEuclideanMetric(n_dims_total)
        println("- Created metric: $(typeof(metric))")

        # 2. Create Hamiltonian - **Pass correct functions**
        # The 3rd argument is for log-density only, the 4th for log-density AND gradient
        hamiltonian = Hamiltonian(metric, logπ, ∂logπ∂θ)
        println("- Created Hamiltonian: $(typeof(hamiltonian))")

        # 3. Create Integrator and Kernel
        init_ϵ = initial_step_size
        println("- Using fixed initial step size: $init_ϵ")
        integrator = Leapfrog(init_ϵ)
        println("- Created integrator: $(typeof(integrator))")
        trajectory = Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn())
        println("- Created trajectory: $(typeof(trajectory))")
        kernel = HMCKernel(trajectory)
        println("- Created kernel: $(typeof(kernel))")

        # 4. Setup adaptation
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric),
                               StepSizeAdaptor(target_accept_ratio, init_ϵ))
        println("- Created adaptor: $(typeof(adaptor))")

        # 5. Run sampling
        println("Starting sampling...")
        rng = Random.default_rng()
        chain, stats = sample(rng, hamiltonian, kernel, initial_params, n_samples, adaptor, n_adapts;
                             drop_warmup=true, progress=true, verbose=false)

        println("Sampling completed successfully.")

    catch e
        @error "ERROR in NUTS sampler!" exception=(e, catch_backtrace())
        return nothing, nothing # Return nothing on error
    end

    return chain, stats # Return samples and statistics
end

end # module Samplers