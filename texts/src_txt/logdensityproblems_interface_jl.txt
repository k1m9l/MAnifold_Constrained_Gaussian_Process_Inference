# In src/logdensityproblems_interface.jl (or src/samplers.jl)
module LogDensityProblemsInterface

using ..Likelihoods: log_likelihood_and_gradient_banded # Your likelihood function
using ..GaussianProcess: GPCov # Your GPCov struct
using ..ODEModels # To reference the function types if needed
using LogDensityProblems
using LinearAlgebra # For reshape

export MagiTarget

# Struct to hold all necessary fixed data for the likelihood calculation
struct MagiTarget{T <: Real, GPCovVec <: AbstractVector{GPCov}, OdeFunc, OdeJacX, OdeJacP}
    yobs::Matrix{T}
    gp_cov_all_dims::GPCovVec
    ode_f!::OdeFunc
    ode_dfdx!::OdeJacX
    ode_dfdp::OdeJacP
    sigma::Vector{T} # Assuming sigma is fixed for now. If sampled, it moves to the parameter vector.
    prior_temperature::Vector{T}
    n_times::Int
    n_dims::Int
    n_params_ode::Int
end

# Required method: Return the total dimension of the parameter vector being sampled
function LogDensityProblems.dimension(target::MagiTarget)
    return target.n_times * target.n_dims + target.n_params_ode
end

# Required method: Specify that the struct provides log density and gradient (order 1)
function LogDensityProblems.capabilities(::Type{<:MagiTarget})
    return LogDensityProblems.LogDensityOrder{1}()
end

# Helper to unpack the parameter vector
function _unpack_params(target::MagiTarget, params::AbstractVector{T}) where {T}
    xlatent_flat = @view params[1:(target.n_times * target.n_dims)]
    theta = @view params[(target.n_times * target.n_dims + 1):end]
    # It's often safer to copy if the downstream function might modify,
    # but log_likelihood_and_gradient_banded seems to take AbstractMatrix/Vector
    xlatent = reshape(xlatent_flat, target.n_times, target.n_dims)
    return xlatent, theta
end

# Required method: Calculate log density only
function LogDensityProblems.logdensity(target::MagiTarget, params::AbstractVector{T}) where {T}
    xlatent, theta = _unpack_params(target, params)

    # Call your existing function, ignoring the gradient
    # Add priors here if necessary: log_prior_val = logpdf(prior_dist, params)
    ll, _ = log_likelihood_and_gradient_banded(
        xlatent,
        theta,
        target.sigma,
        target.yobs,
        target.gp_cov_all_dims,
        target.ode_f!,
        target.ode_dfdx!,
        target.ode_dfdp;
        prior_temperature = target.prior_temperature
    )
    # return ll + log_prior_val # if priors are used
    return ll
end

# Required method: Calculate log density and gradient
function LogDensityProblems.logdensity_and_gradient(target::MagiTarget, params::AbstractVector{T}) where {T}
    xlatent, theta = _unpack_params(target, params)

    # Call your existing function
    # Add priors here if necessary: log_prior_val, grad_prior = logpdf_with_gradient(prior_dist, params)
    ll, grad = log_likelihood_and_gradient_banded(
        xlatent,
        theta,
        target.sigma,
        target.yobs,
        target.gp_cov_all_dims,
        target.ode_f!,
        target.ode_dfdx!,
        target.ode_dfdp;
        prior_temperature = target.prior_temperature
    )
    # return ll + log_prior_val, grad .+ grad_prior # if priors are used
    return ll, grad
end

end # module LogDensityProblemsInterface