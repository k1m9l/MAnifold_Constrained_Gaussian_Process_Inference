# src/gaussian_process.jl

module GaussianProcess

using KernelFunctions
using LinearAlgebra
using BandedMatrices
using PositiveFactorizations # Use for robust Cholesky
using Logging # Added for @warn

export GPCov, calculate_gp_covariances!

# ... (GPCov struct, mat2band remain the same) ...
mutable struct GPCov
    # Inputs (Store for reference)
    phi::Vector{Float64}
    tvec::Vector{Float64}
    kernel::Kernel # Store the kernel object

    # Dense Matrices
    C::Matrix{Float64}
    Cinv::Matrix{Float64}
    Cprime::Matrix{Float64}
    Cdoubleprime::Matrix{Float64}
    mphi::Matrix{Float64}
    Kphi::Matrix{Float64} # This will store the *jittered* Kphi
    Kinv::Matrix{Float64}

    # Banded Matrices
    bandsize::Int
    CinvBand::BandedMatrix{Float64}
    mphiBand::BandedMatrix{Float64}
    KinvBand::BandedMatrix{Float64}

    # Optional
    mu::Vector{Float64}
    dotmu::Vector{Float64}

    # Constructor
    function GPCov()
        new(
            Float64[], Float64[], WhiteKernel(),
            Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
            Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
            Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
            Matrix{Float64}(undef, 0, 0),
            0,
            BandedMatrix{Float64}(undef, (0,0), (0,0)),
            BandedMatrix{Float64}(undef, (0,0), (0,0)),
            BandedMatrix{Float64}(undef, (0,0), (0,0)),
            Float64[], Float64[]
        )
    end
end

function mat2band(mat_input::AbstractMatrix{T}, l::Int, u::Int) where {T}
    # Ensure input is Float64 before creating BandedMatrix
    mat_float = convert(Matrix{Float64}, mat_input)
    return BandedMatrix(mat_float, (l, u))
end


# --- Function for Matern 5/2 Derivatives ---
function _calculate_matern52_derivatives!(
    Cprime::AbstractMatrix{T},
    Cdoubleprime::AbstractMatrix{T},
    tvec::AbstractVector{T},
    variance,
    lengthscale
    ) where {T<:Real}

    variance_scalar = Float64(variance isa AbstractArray ? variance[1] : variance)
    lengthscale_scalar = Float64(lengthscale isa AbstractArray ? lengthscale[1] : lengthscale)

    n = length(tvec)
    if size(Cprime) != (n,n) || size(Cdoubleprime) != (n,n)
        error("Output matrices Cprime/Cdoubleprime have incorrect dimensions")
    end

    sqrt5 = sqrt(5.0)
    l = lengthscale_scalar
    l_sq = l^2
    l_cub = l^3
    term_div_3l_sq = 1.0 / (3.0 * l_sq)
    term_div_3l_cub = 1.0 / (3.0 * l_cub)

    for j in 1:n, i in 1:n
        if i == j # Handle diagonal separately for Cdoubleprime
            Cprime[i, j] = 0.0
            # FIX: Corrected sign based on C++ implicit calculation
            Cdoubleprime[i, j] = 5.0 * variance_scalar / (3.0 * l_sq)
        else
            t_diff = tvec[i] - tvec[j]
            dist = abs(t_diff)
            dist_sq = dist^2
            sgn = sign(t_diff)
            exp_term = exp(-sqrt5 * dist / l)

            common_term_Cprime_base = exp_term * (5*dist*term_div_3l_sq + 5*sqrt5*dist_sq*term_div_3l_cub)
            Cprime[i, j] = -sgn * variance_scalar * common_term_Cprime_base

            term1_Cdoubleprime_base = (-sqrt5/l * exp_term) * (5*dist*term_div_3l_sq + 5*sqrt5*dist_sq*term_div_3l_cub)
            term2_Cdoubleprime_base = (exp_term) * (5*term_div_3l_sq + 10*sqrt5*dist*term_div_3l_cub)
            Cdoubleprime[i, j] = variance_scalar * (term1_Cdoubleprime_base + term2_Cdoubleprime_base)
        end
    end
    # No need for separate diagonal loop anymore
    return nothing
end


# --- Function for RBF (SqExponential) Derivatives ---
# (Keep as before)
function _calculate_rbf_derivatives!(
    Cprime::AbstractMatrix{T},
    Cdoubleprime::AbstractMatrix{T},
    C::AbstractMatrix{T}, # Pass the calculated C matrix
    tvec::AbstractVector{T},
    lengthscale
    ) where {T<:Real}

    lengthscale_scalar = Float64(lengthscale isa AbstractArray ? lengthscale[1] : lengthscale)
    n = length(tvec)
    if size(Cprime) != (n,n) || size(Cdoubleprime) != (n,n) || size(C) != (n,n)
        error("Input/Output matrices C/Cprime/Cdoubleprime have incorrect dimensions")
    end

    l_sq = lengthscale_scalar^2
    l_quad = lengthscale_scalar^4

    for j in 1:n
        for i in 1:n
            t_diff = tvec[i] - tvec[j]
            t_diff_sq = t_diff^2
            Cprime[i, j] = -C[i, j] * t_diff / l_sq
            Cdoubleprime[i, j] = C[i, j] * (1.0/l_sq - t_diff_sq / l_quad)
        end
    end
    return nothing
end


# --- Function to extract parameters ---
# (Keep as before)
function extract_kernel_params(kernel)
    variance = 1.0
    lengthscale = nothing
    base_kernel = kernel
    while true
        if base_kernel isa ScaledKernel
            if hasfield(typeof(base_kernel), :σ²)
                variance *= base_kernel.σ²
            elseif hasfield(typeof(base_kernel), :variance)
                variance *= base_kernel.variance
            else
                 @warn "Could not directly find variance field (:σ² or :variance) in ScaledKernel $(typeof(base_kernel)). Check KernelFunctions.jl version/struct."
            end
            base_kernel = base_kernel.kernel
        elseif base_kernel isa TransformedKernel
            if base_kernel.transform isa ScaleTransform
                 if lengthscale !== nothing; @warn "Multiple lengthscale transforms found?"; end
                 ls_val = base_kernel.transform.s
                 lengthscale = 1.0 / (ls_val isa AbstractVector ? ls_val[1] : ls_val)
            elseif base_kernel.transform isa ARDTransform
                 if lengthscale !== nothing; @warn "Multiple lengthscale transforms found?"; end
                 if length(base_kernel.transform.v) == 1
                     lengthscale = 1.0 / base_kernel.transform.v[1]
                 else
                     @warn "ARDTransform found for multi-dimensional input - derivative functions assume isotropic lengthscale."
                     lengthscale = 1.0 / base_kernel.transform.v[1] # Use first element
                 end
            end
            base_kernel = base_kernel.kernel
        else
            break
        end
    end

    return (variance, base_kernel, lengthscale)
end


"""
    calculate_gp_covariances!(...)

Populates the GPCov struct. Includes derivative calculations for supported kernels.
Applies jitter for numerical stability.
"""
function calculate_gp_covariances!(
        gp_cov::GPCov,
        kernel::KernelFunctions.Kernel,
        phi::Vector{Float64},
        tvec::Vector{Float64},
        bandsize::Int;
        complexity::Int = 0,
        jitter::Float64 = 1e-7 # Default jitter value matching C++
    )

    n = length(tvec)
    gp_cov.phi = phi
    gp_cov.tvec = tvec
    gp_cov.kernel = kernel
    gp_cov.bandsize = bandsize
    l, u = bandsize, bandsize

    # Initialize matrices
    gp_cov.C = Matrix{Float64}(undef, n, n)
    gp_cov.Cinv = Matrix{Float64}(undef, n, n)
    gp_cov.Cprime = Matrix{Float64}(undef, n, n)
    gp_cov.Cdoubleprime = Matrix{Float64}(undef, n, n)
    gp_cov.mphi = Matrix{Float64}(undef, n, n)
    gp_cov.Kphi = Matrix{Float64}(undef, n, n)
    gp_cov.Kinv = Matrix{Float64}(undef, n, n)
    gp_cov.mu = zeros(Float64, n)
    gp_cov.dotmu = zeros(Float64, n)

    # 1. Calculate Dense Covariance Matrix C
    try
        gp_cov.C .= kernelmatrix(kernel, tvec)
    catch e
        @error "Error calculating kernel matrix C:" exception=(e, catch_backtrace()) kernel=kernel tvec=tvec
        rethrow(e)
    end
    if eltype(gp_cov.C) != Float64
        gp_cov.C = Matrix{Float64}(gp_cov.C)
    end
    C_sym_jittered = Symmetric(gp_cov.C + jitter * I)

    # 2. Handle Derivatives
    gp_cov.Cprime .= 0.0
    gp_cov.Cdoubleprime .= 0.0
    derivatives_calculated = false

    if complexity >= 2
        variance, base_kernel_for_deriv, lengthscale = extract_kernel_params(kernel)

        if lengthscale === nothing
             @warn "Lengthscale could not be determined for derivative calculation via extract_kernel_params. Derivatives will be zero."
        else
            try
                if base_kernel_for_deriv isa Matern52Kernel
                    # Call the corrected function
                    _calculate_matern52_derivatives!(gp_cov.Cprime, gp_cov.Cdoubleprime, tvec, variance, lengthscale)
                    derivatives_calculated = true
                elseif base_kernel_for_deriv isa SqExponentialKernel
                    _calculate_rbf_derivatives!(gp_cov.Cprime, gp_cov.Cdoubleprime, gp_cov.C, tvec, lengthscale)
                    derivatives_calculated = true
                else
                    @warn "Time derivative calculation not implemented for base kernel type $(typeof(base_kernel_for_deriv)). Derivatives will be zero."
                end
            catch e
                @error "Error calculating analytical derivatives:" exception=(e, catch_backtrace()) kernel=kernel variance=variance lengthscale=lengthscale
                gp_cov.Cprime .= 0.0
                gp_cov.Cdoubleprime .= 0.0
                derivatives_calculated = false
            end
        end
    end

    # 3. Calculate Inverses and related terms (using jittered matrices)
    local Kphi_sym_jittered

    try
        # Calculate C inverse using jittered matrix
        cholesky_C = cholesky(PositiveFactorizations.Positive, C_sym_jittered)
        gp_cov.Cinv .= inv(cholesky_C)

        # Calculate mphi, Kphi, Kinv based on whether derivatives were calculated
        if derivatives_calculated && !all(iszero, gp_cov.Cprime) && !all(iszero, gp_cov.Cdoubleprime)
            Cprime_T = gp_cov.Cprime
            Cinv_T = gp_cov.Cinv
            gp_cov.mphi .= Cprime_T * Cinv_T
            # Calculate Kphi without jitter first
            Kphi_dense = gp_cov.Cdoubleprime - gp_cov.mphi * Cprime_T'
            # Add jitter and store the jittered version
            Kphi_sym_jittered = Symmetric(Kphi_dense + jitter * I)
            gp_cov.Kphi .= Matrix(Kphi_sym_jittered) # Store jittered Kphi

            # Check eigenvalues *after* potential fix, optional warning
            eig_Kphi_jittered = eigen(Kphi_sym_jittered)
            min_eig_jittered = minimum(eig_Kphi_jittered.values)
            if min_eig_jittered <= 0
                @warn "Kphi (AFTER jitter & potential fix) still has non-positive eigenvalues. Check derivative calculations or increase jitter." min_eigenvalue=min_eig_jittered jitter=jitter kernel=kernel phi=phi
            end

            # Calculate Kinv from jittered Kphi
            cholesky_K = cholesky(PositiveFactorizations.Positive, Kphi_sym_jittered)
            gp_cov.Kinv .= inv(cholesky_K)
        else
            # Fallback if complexity < 2 or derivatives failed/are zero
            gp_cov.mphi .= 0.0
            Kphi_sym_jittered = Symmetric(Matrix(jitter * I, n, n))
            gp_cov.Kphi .= Matrix(Kphi_sym_jittered)
            try
                cholesky_K_fallback = cholesky(PositiveFactorizations.Positive, Kphi_sym_jittered)
                gp_cov.Kinv .= inv(cholesky_K_fallback)
            catch e_fallback
                @error "Cholesky failed even for fallback jittered identity Kphi." jitter=jitter exception=(e_fallback, catch_backtrace())
                rethrow(e_fallback)
            end
        end

    catch e
        matrix_name = "C"
        matrix_for_debug = C_sym_jittered
        # Determine which matrix failed Cholesky if Kphi was involved
        if @isdefined(Kphi_sym_jittered) && !(derivatives_calculated && !all(iszero, gp_cov.Cprime) && !all(iszero, gp_cov.Cdoubleprime))
             matrix_name = "Kphi (Fallback Identity)"
             matrix_for_debug = Kphi_sym_jittered
        elseif @isdefined(Kphi_sym_jittered)
             matrix_name = "Kphi (Calculated)"
             matrix_for_debug = Kphi_sym_jittered
        end

        if isa(e, PosDefException) || contains(string(lowercase(string(e))), "positive definite") || contains(string(lowercase(string(e))), "cholesky")
            @error "Cholesky decomposition failed for matrix '$matrix_name'. Matrix might be ill-conditioned or derivatives unstable." exception=(e, catch_backtrace()) jitter=jitter kernel=kernel phi=phi complexity=complexity derivatives_calculated=derivatives_calculated
            # Avoid erroring if diag fails on empty matrix
            if size(matrix_for_debug, 1) > 0
                println("Matrix '$matrix_name' diagonal sample: ", diag(matrix_for_debug)[1:min(5,n)])
            end
        else
            @error "An unexpected error occurred during Cholesky/inverse calculation:" exception=(e, catch_backtrace()) matrix_name=matrix_name
        end
        rethrow(e)
    end

    # 4. Create Banded Matrix Representations
    gp_cov.CinvBand = mat2band(gp_cov.Cinv, l, u)
    gp_cov.mphiBand = mat2band(gp_cov.mphi, l, u)
    gp_cov.KinvBand = mat2band(gp_cov.Kinv, l, u)

    return nothing
end


end # module GaussianProcess