# src/kernels.jl

module Kernels

using KernelFunctions

# EXPORTS
export create_rbf_kernel, create_matern52_kernel, create_general_matern_kernel

"""
    create_rbf_kernel(variance, lengthscale)

Create a squared exponential (RBF) kernel with specified variance and lengthscale parameters.
"""
function create_rbf_kernel(variance, lengthscale)
    # sigma_c_sq -> variance = sigma^2
    # alpha -> lengthscale = l
    # This syntax worked in tests: k = var * BaseKernel() ∘ ScaleTransform(1/l)
    return variance * SqExponentialKernel() ∘ ScaleTransform(1.0 / lengthscale)
end

"""
    create_matern52_kernel(variance, lengthscale)

Create a Matérn 5/2 kernel with specified variance and lengthscale parameters.
"""
function create_matern52_kernel(variance, lengthscale)
    # This syntax worked in tests: k = var * BaseKernel() ∘ ScaleTransform(1/l)
    return variance * Matern52Kernel() ∘ ScaleTransform(1.0 / lengthscale)
end

"""
    create_general_matern_kernel(variance, lengthscale, nu)

Create a general Matérn kernel with specified variance, lengthscale, and smoothness parameter.
"""
function create_general_matern_kernel(variance, lengthscale, nu)
    # This syntax worked in tests: k = var * BaseKernel() ∘ ScaleTransform(1/l)
    return variance * MaternKernel(ν=nu) ∘ ScaleTransform(1.0 / lengthscale)
end

end # module Kernels