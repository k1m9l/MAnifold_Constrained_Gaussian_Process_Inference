# src/logdensityproblems_interface.jl
module LogDensityProblemsInterface

using ..Likelihoods: log_likelihood_and_gradient_banded
using ..GaussianProcess: GPCov
using ..ODEModels
using LogDensityProblems
using LinearAlgebra

export MagiTarget

"""
    MagiTarget

A struct that implements the LogDensityProblems interface for MAGI.
This allows using standard MCMC samplers with the MAGI log-posterior.

# Fields
- `yobs`: Matrix of observations
- `gp_cov_all_dims`: Vector of pre-calculated GPCov structs
- `ode_f!`: The ODE function
- `ode_dfdx!`: The ODE state Jacobian
- `ode_dfdp`: The ODE parameter Jacobian
- `sigma`: Vector of noise standard deviations
- `prior_temperature`: Vector of temperatures
- `n_times`: Number of time points
- `n_dims`: Number of system dimensions
- `n_params_ode`: Number of ODE parameters
"""
struct MagiTarget{T <: Real, GPCovVec <: AbstractVector{GPCov}, OdeFunc, OdeJacX, OdeJacP}
    yobs::Matrix{T}
    gp_cov_all_dims::GPCovVec
    ode_f!::OdeFunc
    ode_dfdx!::OdeJacX
    ode_dfdp::OdeJacP
    sigma::Vector{T}
    prior_temperature::Vector{T}
    n_times::Int
    n_dims::Int
    n_params_ode::Int
end

"""
    LogDensityProblems.dimension(target::MagiTarget)

Returns the total dimension of the parameter vector for the MAGI target.
"""
function LogDensityProblems.dimension(target::MagiTarget)
    return target.n_times * target.n_dims + target.n_params_ode
end

"""
    LogDensityProblems.capabilities(::Type{<:MagiTarget})

Specifies that the MagiTarget provides log density and gradient (order 1).
"""
function LogDensityProblems.capabilities(::Type{<:MagiTarget})
    return LogDensityProblems.LogDensityOrder{1}()
end

"""
Helper function to unpack parameters from vector to matrices/vectors.
"""
function _unpack_params(target::MagiTarget, params::AbstractVector{T}) where {T}
    xlatent_flat = @view params[1:(target.n_times * target.n_dims)]
    theta = @view params[(target.n_times * target.n_dims + 1):end]
    xlatent = reshape(xlatent_flat, target.n_times, target.n_dims)
    return xlatent, theta
end

"""
    LogDensityProblems.logdensity(target::MagiTarget, params::AbstractVector{T})

Calculates the log density for the MAGI target at the given parameter values.
"""
function LogDensityProblems.logdensity(target::MagiTarget, params::AbstractVector{T}) where {T}
    xlatent, theta = _unpack_params(target, params)
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
    return ll
end

"""
    LogDensityProblems.logdensity_and_gradient(target::MagiTarget, params::AbstractVector{T})

Calculates both the log density and its gradient for the MAGI target.
"""
function LogDensityProblems.logdensity_and_gradient(target::MagiTarget, params::AbstractVector{T}) where {T}
    xlatent, theta = _unpack_params(target, params)
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
    return ll, grad
end

end # module LogDensityProblemsInterface