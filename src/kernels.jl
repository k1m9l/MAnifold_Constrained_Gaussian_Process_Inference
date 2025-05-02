# src/kernels.jl

"""
    Kernels

Provides helper functions to create commonly used covariance kernel objects
from the `KernelFunctions.jl` package, specifically tailored for use within MAGI.

These functions simplify the creation of scaled kernels with specified variance (σ²)
and lengthscale (ℓ) parameters, which are the typical hyperparameters (ϕ) estimated
or used in the MAGI framework.
"""
module Kernels

using KernelFunctions # Dependency for kernel types (SqExponentialKernel, Matern52Kernel, etc.)

# EXPORTS
export create_rbf_kernel, create_matern52_kernel, create_general_matern_kernel

"""
    create_rbf_kernel(variance::Real, lengthscale::Real)

Create a Squared Exponential (also known as Radial Basis Function or RBF) kernel
with specified variance (σ²) and lengthscale (ℓ).

The formula for the RBF kernel is:
k(x, x') = σ² * exp( -‖x - x'‖² / (2 * ℓ²) )

This function uses `KernelFunctions.jl` composition:
- `SqExponentialKernel()`: Base kernel exp(-‖x - x'‖²/2).
- `ScaleTransform(1.0 / lengthscale)`: Transforms the input distance `d = ‖x - x'‖`
  to `d / ℓ`, effectively scaling the squared distance in the exponent to `‖x - x'‖² / ℓ²`.
- `variance * ...`: Scales the entire kernel by the signal variance σ².

# Arguments
- `variance::Real`: The signal variance parameter (σ²). Must be positive.
- `lengthscale::Real`: The lengthscale parameter (ℓ). Must be positive.

# Returns
- `KernelFunctions.Kernel`: A scaled RBF kernel object.
"""
function create_rbf_kernel(variance::Real, lengthscale::Real)
    # Input validation (optional but recommended)
    @assert variance > 0 "Variance (σ²) must be positive"
    @assert lengthscale > 0 "Lengthscale (ℓ) must be positive"

    # Construct the kernel: σ² * k_base(x/ℓ, x'/ℓ)
    # ScaleTransform(1/ℓ) applied to the input achieves the desired scaling inside the kernel.
    return variance * SqExponentialKernel() ∘ ScaleTransform(1.0 / lengthscale)
end

"""
    create_matern52_kernel(variance::Real, lengthscale::Real)

Create a Matérn 5/2 kernel with specified variance (σ²) and lengthscale (ℓ).

The formula for the Matérn 5/2 kernel is:
k(r) = σ² * (1 + √5*r/ℓ + 5*r²/(3*ℓ²)) * exp(-√5*r/ℓ)
where r = ‖x - x'‖.

This kernel produces sample functions that are twice mean-square differentiable,
a property often desired for modeling physical systems and required by MAGI
for calculating necessary derivatives.

This function uses `KernelFunctions.jl` composition similar to `create_rbf_kernel`.

# Arguments
- `variance::Real`: The signal variance parameter (σ²). Must be positive.
- `lengthscale::Real`: The lengthscale parameter (ℓ). Must be positive.

# Returns
- `KernelFunctions.Kernel`: A scaled Matérn 5/2 kernel object.
"""
function create_matern52_kernel(variance::Real, lengthscale::Real)
    # Input validation
    @assert variance > 0 "Variance (σ²) must be positive"
    @assert lengthscale > 0 "Lengthscale (ℓ) must be positive"

    # Construct the kernel: σ² * k_base(x/ℓ, x'/ℓ)
    return variance * Matern52Kernel() ∘ ScaleTransform(1.0 / lengthscale)
end

"""
    create_general_matern_kernel(variance::Real, lengthscale::Real, ν::Real)

Create a general Matérn kernel with specified variance (σ²), lengthscale (ℓ),
and smoothness parameter (ν).

The formula involves Bessel functions:
k(r) = σ² * (2^(1-ν) / Γ(ν)) * (√2ν * r/ℓ)^ν * K_ν(√2ν * r/ℓ)
where r = ‖x - x'‖, Γ is the Gamma function, and K_ν is the modified Bessel
function of the second kind.

The smoothness parameter ν controls the differentiability of the sample functions.
Common special cases:
- ν = 1/2: Exponential kernel (Ornstein-Uhlenbeck process)
- ν = 3/2: Matérn 3/2 kernel
- ν = 5/2: Matérn 5/2 kernel (equivalent to `create_matern52_kernel`)
- ν → ∞: Squared Exponential (RBF) kernel

# Arguments
- `variance::Real`: The signal variance parameter (σ²). Must be positive.
- `lengthscale::Real`: The lengthscale parameter (ℓ). Must be positive.
- `ν::Real`: The smoothness parameter (ν, often written as nu). Must be positive.

# Returns
- `KernelFunctions.Kernel`: A scaled general Matérn kernel object.
"""
function create_general_matern_kernel(variance::Real, lengthscale::Real, ν::Real)
    # Input validation
    @assert variance > 0 "Variance (σ²) must be positive"
    @assert lengthscale > 0 "Lengthscale (ℓ) must be positive"
    @assert ν > 0 "Smoothness parameter (ν) must be positive"

    # Construct the kernel: σ² * k_base(x/ℓ, x'/ℓ)
    # Pass the smoothness parameter ν to the MaternKernel constructor.
    return variance * MaternKernel(ν=ν) ∘ ScaleTransform(1.0 / lengthscale)
end

end # module Kernels
