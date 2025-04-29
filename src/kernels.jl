# src/kernels.jl

module Kernels

using KernelFunctions

# --- Decide what to export ---
# Option A: Export helper functions
export create_rbf_kernel, create_matern52_kernel, create_general_matern_kernel
# -----------------------------

# Helper functions using the working syntax from tests
function create_rbf_kernel(variance, lengthscale)
    # sigma_c_sq -> variance = sigma^2
    # alpha -> lengthscale = l
    # This syntax worked in tests: k = var * BaseKernel() ∘ ScaleTransform(1/l)
    return variance * SqExponentialKernel() ∘ ScaleTransform(1.0 / lengthscale)
end

function create_matern52_kernel(variance, lengthscale)
    # This syntax worked in tests: k = var * BaseKernel() ∘ ScaleTransform(1/l)
    return variance * Matern52Kernel() ∘ ScaleTransform(1.0 / lengthscale)
end

function create_general_matern_kernel(variance, lengthscale, nu)
    # This syntax worked in tests: k = var * BaseKernel() ∘ ScaleTransform(1/l)
    return variance * MaternKernel(ν=nu) ∘ ScaleTransform(1.0 / lengthscale)
end

# Add similar functions for other kernels if needed...

end # module Kernels