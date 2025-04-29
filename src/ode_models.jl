# src/ode_models.jl

module ODEModels

# --- Ensure this export line is inside the module ---
export fn_ode!, hes1_ode!, hes1log_ode!, hes1log_ode_fixg!, hes1log_ode_fixf!, hiv_ode!, ptrans_ode!
# -----------------------------------------------------

# Notes on Translation:
# - C++ `theta` vector maps to Julia `p` (parameter vector/tuple).
# - C++ `x` matrix (columns are states) maps to Julia `u` (state vector).
# - Assumes parameters `p` are passed as a vector or tuple in the same order as `theta` in C++.
# - Uses standard Julia math functions and broadcasting where appropriate.


"""
FitzHugh-Nagumo ODE system definition (in-place).

Arguments:
- du: Derivative vector (output). du[1] = dV/dt, du[2] = dR/dt
- u: State vector (input). u[1] = V, u[2] = R
- p: Parameter vector/tuple (input). p[1]=a, p[2]=b, p[3]=c (maps to theta[0], theta[1], theta[2])
- t: Time (input, often unused).
"""
function fn_ode!(du, u, p, t)
    V = u[1]
    R = u[2]
    a, b, c = p # Or p[1], p[2], p[3] if p is a Vector/Tuple

    du[1] = c * (V - (V^3) / 3.0 + R)
    du[2] = -1.0/c * (V - a + b * R)
    return nothing
end



"""
Hes1 ODE system definition (in-place).

Arguments:
- du: Derivative vector (output). du[1]=dP/dt, du[2]=dM/dt, du[3]=dH/dt
- u: State vector (input). u[1]=P, u[2]=M, u[3]=H
- p: Parameter vector/tuple (input). p[1..7] maps to theta[0..6]
- t: Time (input, often unused).
"""
function hes1_ode!(du, u, p, t)
    P = u[1]
    M = u[2]
    H = u[3]
    # p = (p1, p2, ..., p7) maps to theta(0..6)

    du[1] = -p[1]*P*H + p[2]*M - p[3]*P
    du[2] = -p[4]*M + p[5]/(1 + P^2)
    du[3] = -p[1]*P*H + p[6]/(1 + P^2) - p[7]*H
    return nothing
end


"""
Hes1 log-transformed ODE system definition (in-place).
States u are log(P), log(M), log(H).

Arguments:
- du: Derivative vector (output). du[1]=d(logP)/dt, du[2]=d(logM)/dt, du[3]=d(logH)/dt
- u: State vector (input). u[1]=logP, u[2]=logM, u[3]=logH
- p: Parameter vector/tuple (input). p[1..7] maps to theta[0..6]
- t: Time (input, often unused).
"""
function hes1log_ode!(du, u, p, t)
    logP = u[1]
    logM = u[2]
    logH = u[3]
    # p = (p1, p2, ..., p7) maps to theta(0..6)

    P = exp(logP)
    M = exp(logM)
    H = exp(logH)

    one_plus_Psq = 1 + P^2

    # Note: C++ calculated d(logX)/dt = (dX/dt) / X
    # dP/dt = -p[1]*P*H + p[2]*M - p[3]*P
    du[1] = -p[1]*H + p[2]*M/P - p[3]
    # dM/dt = -p[4]*M + p[5]/(1+P^2)
    du[2] = -p[4] + p[5]/(one_plus_Psq * M)
    # dH/dt = -p[1]*P*H + p[6]/(1+P^2) - p[7]*H
    du[3] = -p[1]*P + p[6]/(one_plus_Psq * H) - p[7]
    return nothing
end


"""
Hes1 log-transformed ODE system definition with fixed gamma (p[7]=0.3) (in-place).
States u are log(P), log(M), log(H).

Arguments:
- du: Derivative vector (output). du[1]=d(logP)/dt, du[2]=d(logM)/dt, du[3]=d(logH)/dt
- u: State vector (input). u[1]=logP, u[2]=logM, u[3]=logH
- p: Parameter vector/tuple (input). p[1..6] maps to theta[0..5]
- t: Time (input, often unused).
"""
function hes1log_ode_fixg!(du, u, p, t)
    logP = u[1]
    logM = u[2]
    logH = u[3]
    # p = (p1, p2, ..., p6) maps to theta(0..5)
    # theta(6) is fixed at 0.3

    P = exp(logP)
    M = exp(logM)
    H = exp(logH)

    one_plus_Psq = 1 + P^2
    gamma_fixed = 0.3

    # Note: C++ calculated d(logX)/dt = (dX/dt) / X
    du[1] = -p[1]*H + p[2]*M/P - p[3]
    du[2] = -p[4] + p[5]/(one_plus_Psq * M)
    du[3] = -p[1]*P + p[6]/(one_plus_Psq * H) - gamma_fixed
    return nothing
end

"""
Hes1 log-transformed ODE system definition with fixed f (p[6]=20.0) (in-place).
States u are log(P), log(M), log(H).

Arguments:
- du: Derivative vector (output). du[1]=d(logP)/dt, du[2]=d(logM)/dt, du[3]=d(logH)/dt
- u: State vector (input). u[1]=logP, u[2]=logM, u[3]=logH
- p: Parameter vector/tuple (input). p[1..5] and p[6] map to theta[0..4] and theta[5] (skipping theta[5] in C++ naming which corresponds to p[6] here).
- t: Time (input, often unused).
"""
function hes1log_ode_fixf!(du, u, p, t)
    logP = u[1]
    logM = u[2]
    logH = u[3]
    # p = (p1, p2, p3, p4, p5, p6) maps to theta(0..4) and theta(6) in C++ code. theta(5) is fixed at 20.0
    # theta[0]=p[1], theta[1]=p[2], theta[2]=p[3], theta[3]=p[4], theta[4]=p[5], theta[6]=p[6]

    P = exp(logP)
    M = exp(logM)
    H = exp(logH)

    one_plus_Psq = 1 + P^2
    f_fixed = 20.0

    du[1] = -p[1]*H + p[2]*M/P - p[3]
    du[2] = -p[4] + p[5]/(one_plus_Psq * M)
    du[3] = -p[1]*P + f_fixed/(one_plus_Psq * H) - p[6] # p[6] maps to theta(6) in C++ code
    return nothing
end


"""
HIV log-transformed ODE system definition (in-place).
States u are log(T), log(Tm), log(Tw), log(Tmw).

Arguments:
- du: Derivative vector (output). du[1]=d(logT)/dt, ..., du[4]=d(logTmw)/dt
- u: State vector (input). u[1]=logT, u[2]=logTm, u[3]=logTw, u[4]=logTmw
- p: Parameter vector/tuple (input). p[1..9] maps to theta[0..8]. Note scaling factor 1e-6 in C++.
- t: Time (input, often unused).
"""
function hiv_ode!(du, u, p, t)
    logT   = u[1]
    logTm  = u[2]
    logTw  = u[3]
    logTmw = u[4]
    # p = (p1, ..., p9) maps to theta(0..8)

    T   = exp(logT)
    Tm  = exp(logTm)
    Tw  = exp(logTw)
    Tmw = exp(logTmw)

    # Scaling factor from C++ code
    sf = 1e-6

    # Note: C++ calculated d(logX)/dt = (dX/dt) / X
    # dT/dt = T*(theta(0) - sf*theta(1)*Tm - sf*theta(2)*Tw - sf*theta(3)*Tmw)
    du[1] = p[1] - sf*p[2]*Tm - sf*p[3]*Tw - sf*p[4]*Tmw

    # dTm/dt = Tm*(theta(6) + sf*theta(1)*T - sf*theta(4)*Tw) + sf*0.25*theta(3)*Tmw*T
    du[2] = p[7] + sf*p[2]*T - sf*p[5]*Tw + sf*0.25*p[4]*Tmw*T/Tm

    # dTw/dt = Tw*(theta(7) + sf*theta(2)*T - sf*theta(5)*Tm) + sf*0.25*theta(3)*Tmw*T
    du[3] = p[8] + sf*p[3]*T - sf*p[6]*Tm + sf*0.25*p[4]*Tmw*T/Tw # C++ theta(5) maps to p[6] here

    # dTmw/dt = Tmw*(theta(8) + 0.5*sf*theta(3)*T) + (sf*theta(4)+sf*theta(5))*Tw*Tm
    du[4] = p[9] + 0.5*sf*p[4]*T + (sf*p[5]+sf*p[6])*Tw*Tm/Tmw # C++ theta(4)->p[5], theta(5)->p[6]

    return nothing
end


"""
Protein Transduction ODE system definition (in-place).

Arguments:
- du: Derivative vector (output). du[1]=dS/dt, ..., du[5]=dRPP/dt
- u: State vector (input). u[1]=S, u[2]=dS, u[3]=R, u[4]=RS, u[5]=RPP
- p: Parameter vector/tuple (input). p[1..6] maps to theta[0..5]
- t: Time (input, often unused).
"""
function ptrans_ode!(du, u, p, t)
    S   = u[1]
    # dS  = u[2] # This state seems unused in the ODE definitions? Check C++ usage.
    R   = u[3]
    RS  = u[4]
    RPP = u[5]
    # p = (p1, ..., p6) maps to theta(0..5)

    du[1] = -p[1]*S - p[2] * S * R + p[3] * RS
    du[2] = p[1]*S  # Assuming this calculates dS/dt based on C++ code structure
    du[3] = -p[2]*S*R + p[3]*RS + p[5] * RPP / (p[6]+RPP) # C++ theta(4)->p[5], theta(5)->p[6]
    du[4] = p[2]*S*R - p[3]* RS - p[4]*RS             # C++ theta(3)->p[4]
    du[5] = p[4]*RS - p[5] * RPP / (p[6]+RPP)             # C++ theta(3)->p[4], theta(4)->p[5], theta(5)->p[6]
    return nothing
end

# --- NEW Jacobian Functions ---

"""
FitzHugh-Nagumo ODE Jacobian w.r.t. states X (in-place).
Calculates d(f_i)/d(x_j) and stores it in J.
J[i, j] = derivative of ith component of f w.r.t jth state variable.

Arguments:
- J: Jacobian matrix (output, DxD). J[1,1]=dV'/dV, J[1,2]=dV'/dR, J[2,1]=dR'/dV, J[2,2]=dR'/dR
- u: State vector (input). u[1]=V, u[2]=R
- p: Parameter vector/tuple (input). p[1]=a, p[2]=b, p[3]=c
- t: Time (input, often unused).
"""
function fn_ode_dx!(J, u, p, t)
    V = u[1]
    # R = u[2] # R not needed for these derivatives
    a, b, c = p

    # d(V')/dV = c * (1 - V^2)
    J[1, 1] = c * (1.0 - V^2)
    # d(V')/dR = c
    J[1, 2] = c
    # d(R')/dV = -1.0/c
    J[2, 1] = -1.0 / c
    # d(R')/dR = -1.0/c * (b) = -b/c
    J[2, 2] = -b / c
    return nothing
end

"""
FitzHugh-Nagumo ODE Jacobian w.r.t. parameters theta (returns Matrix).
Calculates d(f_i)/d(p_j).
Returns D x n_params Matrix. Col 1 is d/da, Col 2 is d/db, Col 3 is d/dc.

Arguments:
- u: State vector (input). u[1]=V, u[2]=R
- p: Parameter vector/tuple (input). p[1]=a, p[2]=b, p[3]=c
- t: Time (input, often unused).
"""
function fn_ode_dtheta(u, p, t)
    V = u[1]
    R = u[2]
    a, b, c = p
    n_params = length(p)
    n_dims = length(u)
    Jp = zeros(n_dims, n_params)

    # Derivatives of V' = c*(V - V^3/3 + R)
    # dV'/da = 0
    Jp[1, 1] = 0.0
    # dV'/db = 0
    Jp[1, 2] = 0.0
    # dV'/dc = V - V^3/3 + R
    Jp[1, 3] = V - (V^3) / 3.0 + R

    # Derivatives of R' = -1/c * (V - a + b*R)
    # dR'/da = -1/c * (-1) = 1/c
    Jp[2, 1] = 1.0 / c
    # dR'/db = -1/c * (R) = -R/c
    Jp[2, 2] = -R / c
    # dR'/dc = +1/(c^2) * (V - a + b*R) = -R' / c
    Jp[2, 3] = (1.0 / c^2) * (V - a + b * R) # Or: Jp[2, 3] = -(-1.0/c * (V - a + b * R)) / c

    return Jp
end


"""
Hes1 ODE Jacobian w.r.t. states X (in-place).
J[i, j] = derivative of ith component of f w.r.t jth state variable.

Arguments:
- J: Jacobian matrix (output, 3x3).
- u: State vector (input). u[1]=P, u[2]=M, u[3]=H
- p: Parameter vector/tuple (input). p[1..7]
- t: Time (input, often unused).
"""
function hes1_ode_dx!(J, u, p, t)
    P = u[1]
    # M = u[2] # M not needed
    H = u[3]
    # p = (p1..p7)

    one_plus_Psq = 1 + P^2

    # Row 1: dP'/dP, dP'/dM, dP'/dH
    J[1, 1] = -p[1]*H - p[3]
    J[1, 2] = p[2]
    J[1, 3] = -p[1]*P

    # Row 2: dM'/dP, dM'/dM, dM'/dH
    J[2, 1] = -p[5] * (2*P) / (one_plus_Psq^2) # Derivative of p5/(1+P^2) w.r.t P
    J[2, 2] = -p[4]
    J[2, 3] = 0.0

    # Row 3: dH'/dP, dH'/dM, dH'/dH
    J[3, 1] = -p[1]*H - p[6] * (2*P) / (one_plus_Psq^2) # Derivative of p6/(1+P^2) w.r.t P
    J[3, 2] = 0.0
    J[3, 3] = -p[1]*P - p[7]

    return nothing
end


"""
Hes1 ODE Jacobian w.r.t. parameters theta (returns Matrix).
Calculates d(f_i)/d(p_j).
Returns 3 x 7 Matrix.

Arguments:
- u: State vector (input). u[1]=P, u[2]=M, u[3]=H
- p: Parameter vector/tuple (input). p[1..7]
- t: Time (input, often unused).
"""
function hes1_ode_dtheta(u, p, t)
    P = u[1]
    M = u[2]
    H = u[3]
    n_params = length(p) # Should be 7
    n_dims = length(u)   # Should be 3
    Jp = zeros(n_dims, n_params)

    one_plus_Psq = 1 + P^2

    # Row 1: dP'/dp1 ... dP'/dp7
    Jp[1, 1] = -P*H
    Jp[1, 2] = M
    Jp[1, 3] = -P
    # Jp[1, 4:7] = 0.0 (already zeros)

    # Row 2: dM'/dp1 ... dM'/dp7
    # Jp[2, 1:3] = 0.0
    Jp[2, 4] = -M
    Jp[2, 5] = 1.0 / one_plus_Psq
    # Jp[2, 6:7] = 0.0

    # Row 3: dH'/dp1 ... dH'/dp7
    Jp[3, 1] = -P*H
    # Jp[3, 2:5] = 0.0
    Jp[3, 6] = 1.0 / one_plus_Psq
    Jp[3, 7] = -H

    return Jp
end


end # module ODEModels