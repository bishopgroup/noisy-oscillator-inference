import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.linalg import solve_banded
from typing import Literal, Optional

def simulate_data(param, N, dt, rng, theta0=0):
    """Simulate latent phase and noisy observations for the linear model."""
    omega, Dtheta, R = param
    
    # time step tk = dt*k for k = 1...N (but python indexes from 0)
    tk = dt*(1+np.arange(N))

    # unobserved phase
    thetak = omega*tk + theta0 + np.cumsum(rng.normal(0,np.sqrt(2*Dtheta*dt),(N,))) 

    # measured phase
    yk = thetak + rng.normal(0, np.sqrt(R), (N,))

    return tk, thetak, yk


def kalman_filter(param, data, priors):    
    """Run scalar Kalman filtering for latent phase states."""
    omega, Dtheta, R = param
    tk, yk = data
    mu_theta, sigma_theta, *_ = priors

    # helper variables
    N = yk.shape[0]
    dt = tk[1]-tk[0]
    Q = 2*Dtheta*dt
    wt = omega*dt
    
    # Filter
    thetakk = np.zeros((N,))     # \hat{theta}_{k|k}
    Pkk = np.zeros((N,))         # P_{k|k}
    thetakk_1 = np.zeros((N,))   # \hat{theta}_{k|k-1}
    Pkk_1 = np.zeros((N,))       # P_{k|k-1}
    
    # initial time point
    k = 0 
    
    # Predict
    thetakk_1[k] = mu_theta
    Pkk_1[k] = sigma_theta**2 + Q
    
    # Update
    Pkk[k] = 1/(1/Pkk_1[k] + 1/R)
    thetakk[k] = Pkk[k]*(thetakk_1[k]/Pkk_1[k] + yk[k]/R)
    
    for k in range(1,N):
        # Predict
        thetakk_1[k] = thetakk[k-1] + wt
        Pkk_1[k] = Pkk[k-1] + Q
    
        # Update
        Pkk[k] = 1/(1/Pkk_1[k] + 1/R)
        thetakk[k] = Pkk[k]*(thetakk_1[k]/Pkk_1[k] + yk[k]/R)

    return thetakk, Pkk, thetakk_1, Pkk_1


def rts_smoother(param, data, kalman_phases):    
    """Run Rauch-Tung-Striebel backward smoothing on filtered states."""
    omega, Dtheta, R = param
    tk, yk = data
    thetakk, Pkk, thetakk_1, Pkk_1 = kalman_phases

    # helper variables
    N = yk.shape[0]
    dt = tk[1]-tk[0]
    Q = 2*Dtheta*dt
    wt = omega*dt

    # Smoother
    thetakN = np.zeros((N,))   # theta_{k|N}
    PkN = np.zeros((N,))       # P_{k|N}
    Sigmakk_1 = np.zeros((N,)) # k,k-1 entry of covariance matrix (note: k=0 entry is meaningless)

    # final time point
    k = N-1
    thetakN[k] = thetakk[k]
    PkN[k] = Pkk[k]

    # k,k-1 entry of covariance matrix (lower diagonal)
    Sigmakk_1[k] = np.linalg.inv(np.array([[1/PkN[k] + 1/Q - 1/Pkk_1[k], -1/Q],
                                           [-1/Q,  1/Q + 1/Pkk[k-1]]]))[0,1]
    
    for k in range(N-2,-1,-1):
        Ck = Pkk[k] / Pkk_1[k+1]
        thetakN[k] = thetakk[k] + Ck*(thetakN[k+1] - thetakk_1[k+1])
        PkN[k] = Pkk[k] + Ck*(PkN[k+1] - Pkk_1[k+1])*Ck
    
        # k,k-1 entry of covariance matrix
        Sigmakk_1[k] = np.linalg.inv(np.array([[1/PkN[k] + 1/Q - 1/Pkk_1[k], -1/Q],
                                               [-1/Q,  1/Q + 1/Pkk[k-1]]]))[1,0]

    return thetakN, PkN, Sigmakk_1


def theta_covariance(param, data, priors):
    omega, Dtheta, R = param
    tk, yk = data
    mu_theta, sigma_theta, *_ = priors

    # helper variables
    N = yk.shape[0]
    dt = tk[1]-tk[0]
    Q = 2*Dtheta*dt

    # Construct the banded matrix
    ab = np.zeros((3, N))
    ab[0, 1:] = -1/Q      # Superdiagonal (row 0, offset by +1)
    ab[1, :] = 2/Q + 1/R  # Main diagonal
    ab[1, 0] = 1/Q + 1/R + 1/sigma_theta**2
    ab[1, -1] = 1/Q + 1/R
    ab[2, :-1] = -1/Q     # Subdiagonal (row 2, offset by -1)

    # Solve Ax = rhs
    Sigma = solve_banded((1,1), ab, np.eye(N))

    return Sigma

    
def objective(param, priors, statistics):
    omega, lnDtheta = param
    _, _, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R = priors
    N, dt, S0, S1, Sy = statistics

    Dtheta = np.exp(lnDtheta)
    wdt = omega*dt

    return (omega - mu_omega)**2 / (2*sigma_omega**2) \
         + (alpha_D + 1)*np.log(Dtheta) + beta_D / Dtheta \
         + 0.5*N*np.log(Dtheta) + (S0 - 2*S1*wdt + (N-1)*wdt**2) / (4*Dtheta*dt)


def jac(param, priors, statistics):
    omega, lnDtheta = param
    _, _, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R = priors
    N, dt, S0, S1, Sy = statistics
    
    Dtheta = np.exp(lnDtheta)
    wdt = omega*dt

    return np.array([
        (omega - mu_omega) / sigma_omega**2 + (-S1 + (N-1)*wdt) / (2*Dtheta),
        (alpha_D + 1 + 0.5*N) - (beta_D + (S0 - 2*S1*wdt + (N-1)*wdt**2)/(4*dt))/Dtheta  
    ])
    

def update_parameters(old_param, data, priors):
    """Perform one EM/MAP parameter update step."""
    tk, yk = data
    _, _, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R = priors

    #### Filter and Smooth
    kalman_phases = kalman_filter(old_param, data, priors)
    thetakN, PkN, Sigmakk_1 = rts_smoother(old_param, data, kalman_phases)
    
    #### Statistics 
    N = yk.shape[0]
    dt = tk[1]-tk[0]
    S0 = np.sum((thetakN[1:] - thetakN[:-1])**2 + PkN[1:] + PkN[:-1] - 2*Sigmakk_1[1:])
    S1 = thetakN[-1] - thetakN[0]
    Sy = np.sum((thetakN-yk)**2 + PkN)
    statistics = (N, dt, S0, S1, Sy)
    
    #### Update    
    # noise variance
    R_new = (2*beta_R + Sy) / (2*(alpha_R + 1) + N)

    # frequency
    omega_new = S1 / ((N-1)*dt)

    # phase diffusivity
    Dtheta_new = S0 / (2*(N+2)*dt)

    # Optimize    
    tol = 1e-5
    guess = [omega_new, np.log(Dtheta_new)]
    res = minimize(objective, guess, args=(priors,statistics), jac=jac, tol=tol, method='L-BFGS-B')

    omega_new = res.x[0]
    Dtheta_new = np.exp(res.x[1])
    
    return [omega_new, Dtheta_new, R_new]


def sample_phases(param, data, priors, rng, N_samples=1000):
    """Draw latent phase trajectories with forward-filter backward-sampling."""
    #### Kalman Filter
    thetakk, Pkk, thetakk_1, Pkk_1 = kalman_filter(param, data, priors)
    
    #### Forward-Filter Backward-Sample (FFBS)
    N = thetakk.shape[0]
    theta_samples = np.zeros((N_samples,N))

    # initialize for k = N
    k = N-1
    theta_samples[:,k] = rng.normal(loc=thetakk[k], scale=np.sqrt(Pkk[k]), size=N_samples)
    
    # iterate backward   
    Jk = Pkk[:-1] / Pkk_1[1:]
    for k in range(N-1,0,-1):
        # mean
        mean = thetakk[k-1] + Jk[k-1]*(theta_samples[:,k] - thetakk_1[k])
        # variance
        var = Pkk[k-1] - Jk[k-1] * Pkk_1[k] * Jk[k-1]
        # sample
        theta_samples[:,k-1] = rng.normal(loc=mean, scale=np.sqrt(var))

    return theta_samples


def prior_information(lambda_map, priors):
    omega, Dtheta, R = lambda_map
    _, _, mu_omega, sigma_omega, alpha_D, beta_D, alpha_R, beta_R = priors

    # Prior information matrix
    I_prior = np.zeros((3,3))
    I_prior[0,0] = 1/sigma_omega**2
    I_prior[1,1] = beta_D/Dtheta
    I_prior[2,2] = beta_R/R

    return I_prior


def complete_data_information(lambda_map, statistics):
    omega, Dtheta, R = lambda_map
    N, dt, S0, S1, Sy = statistics

    wdt = omega*dt

    # Complete-data information matrix
    I_comp = np.zeros((3,3))
    I_comp[0,0] = (N-1)*dt/(2*Dtheta)
    I_comp[0,1] = (S1 - (N-1)*wdt)/(2*Dtheta)
    I_comp[1,1] = (S0 - 2*S1*wdt + (N-1)*wdt**2)/(4*Dtheta*dt)
    I_comp[2,2] = Sy/(2*R) 
    I_comp = I_comp + np.triu(I_comp, 1).T

    return I_comp



def missing_information_MC(lambda_map, data, priors, rng, N_samples):
    omega, Dtheta, R = lambda_map
    tk, yk = data
    
    N = tk.shape[0]
    dt = tk[1] - tk[0]
    wdt = omega*dt
    
    ## Sample phases 
    theta_samples = sample_phases(lambda_map, data, priors, rng, N_samples)
    S0_samples = np.sum((theta_samples[:,1:] - theta_samples[:,:-1])**2, axis=1)
    S1_samples = np.sum((theta_samples[:,1:] - theta_samples[:,:-1]), axis=1)
    Sy_samples = np.sum((yk[None,:] - theta_samples)**2, axis=1)
    
    ## Missing information matrix
    score = np.zeros((S1_samples.shape[0], 3))
    score[:,0] = (S1_samples - (N-1)*wdt)/(2*Dtheta)
    score[:,1] = (S0_samples - 2*N*Dtheta*dt - 2*S1_samples*wdt + (N-1)*wdt**2)/(4*Dtheta*dt)
    score[:,2] = (Sy_samples - N*R)/(2*R)
    
    I_miss_MC = np.cov(score, rowvar=False, ddof=1)

    return I_miss_MC
    

def missing_information(
    y: np.ndarray,
    m: np.ndarray,
    omega: float,
    D: float,
    R: float,
    dt: float,
    sigma_theta2: float,
    *,
    compute_traces: bool = True,
) -> np.ndarray:
    """
    Exact missing-information matrix I_miss = Cov(score | y, MAP) for parameters
    [omega, ln D_theta, ln R] in the linear phase model with prior on theta_1:

        theta_1 ~ N(mu_theta, sigma_theta2)      (mu_theta not needed here; fixed)
        theta_{k+1} = theta_k + omega*dt + eps_k,   eps_k ~ N(0, 2 D dt)
        y_k = theta_k + eta_k,                     eta_k ~ N(0, R)

    Inputs
    ------
    y : (N,) array
        Observations y_1..y_N.
    m : (N,) array
        Smoother mean E[theta_1..theta_N | y, MAP].
    omega, D, R, dt : floats
        MAP values.
    sigma_theta2 : float
        Fixed prior variance for theta_1.
    compute_traces : bool
        If False, raises NotImplementedError (kept for future approx variants).

    Returns
    -------
    I_miss : (3,3) array
        Missing information matrix for [omega, ln D, ln R], evaluated at MAP.

    Notes
    -----
    This avoids forming the dense covariance Sigma explicitly. It uses:
      - banded solves with the posterior precision J (tridiagonal)
      - streaming solves to compute trace(Sigma^2), trace(A0 Sigma^2), trace(Sigma_u^2)
    Complexity ~ O(N^2) time, O(N) memory for scalar phase.
    """

    if not compute_traces:
        raise NotImplementedError("Exact I_miss here uses trace terms; set compute_traces=True.")

    y = np.asarray(y, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    if y.shape != m.shape:
        raise ValueError("y and m must have the same shape (N,).")
    N = y.size
    if N < 2:
        raise ValueError("Need N>=2.")
    M = N - 1  # number of increments

    # --- Build banded posterior precision J for theta_1..theta_N ---
    # q = process variance per step for theta_{k+1}-theta_k
    q = 2.0 * D * dt
    if q <= 0 or R <= 0 or sigma_theta2 <= 0:
        raise ValueError("Require D>0, R>0, dt>0, sigma_theta2>0.")

    invR = 1.0 / R
    invq = 1.0 / q
    invsig = 1.0 / sigma_theta2

    # Tridiagonal J: lower diag (l), main diag (d), upper diag (u)
    d = np.empty(N, dtype=float)
    u = np.full(N - 1, -invq, dtype=float)
    l = np.full(N - 1, -invq, dtype=float)

    d[0] = invR + invq + invsig         # k=1 endpoint + prior
    if N > 2:
        d[1:N-1] = invR + 2.0 * invq    # interior
    d[N-1] = invR + invq                # k=N endpoint

    # Banded storage for solve_banded: (l+u+1, N) with (1,1) band => shape (3,N)
    # ab[0,1:] = upper diag, ab[1,:] = main, ab[2,:-1] = lower diag
    ab = np.zeros((3, N), dtype=float)
    ab[0, 1:] = u
    ab[1, :] = d
    ab[2, :-1] = l

    # --- Helpers: apply Sigma = J^{-1} to a vector via banded solve ---
    def apply_sigma(b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float).ravel()
        if b.size != N:
            raise ValueError("apply_sigma: b must have length N.")
        return solve_banded((1, 1), ab, b)

    # --- Tridiagonal multiply by A0 = B^T B (increment stiffness) on theta_1..theta_N ---
    # A0 has diag: [1,2,2,...,2,1], offdiag: -1
    def apply_A0(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        if x.size != N:
            raise ValueError("apply_A0: x must have length N.")
        out = np.empty_like(x)
        if N == 2:
            # A0 = [[1,-1],[-1,1]]
            out[0] = x[0] - x[1]
            out[1] = -x[0] + x[1]
            return out
        out[0] = x[0] - x[1]
        out[1:N-1] = -x[0:N-2] + 2.0 * x[1:N-1] - x[2:N]
        out[N-1] = -x[N-2] + x[N-1]
        return out

    # --- Difference operator B and its transpose (no explicit matrix) ---
    # u = B theta where u_k = theta_{k+1} - theta_k, k=1..M  (0-based: k=0..M-1)
    def apply_B(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != N:
            raise ValueError("apply_B: theta must have length N.")
        return theta[1:] - theta[:-1]  # length M

    # w = B^T v where v length M, returns length N:
    # w_1 = -v_1
    # w_k = v_{k-1} - v_k for k=2..N-1
    # w_N = v_{N-1}
    def apply_Bt(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).ravel()
        if v.size != M:
            raise ValueError("apply_Bt: v must have length N-1.")
        w = np.zeros(N, dtype=float)
        w[0] = -v[0]
        if N > 2:
            w[1:N-1] = v[0:M-1] - v[1:M]
        w[N-1] = v[M-1]
        return w

    # Apply Sigma_u = B Sigma B^T to a vector in increment space (length M)
    def apply_sigma_u(v: np.ndarray) -> np.ndarray:
        w = apply_Bt(v)        # length N
        x = apply_sigma(w)     # Sigma w
        return apply_B(x)      # B x, length M

    # --- Common vectors ---
    r = y - m
    c = np.zeros(N, dtype=float)
    c[0] = -1.0
    c[-1] = +1.0
    onesM = np.ones(M, dtype=float)

    # --- Increment mean mu_u ---
    mu_u = apply_B(m)  # length M

    # --- Quadratic forms / covariances via solves ---
    # Var(S1) = c^T Sigma c
    Sig_c = apply_sigma(c)
    V_S1 = float(c @ Sig_c)

    # Cov(S1, Sy) = 2 c^T Sigma (m - y) = -2 c^T Sigma r
    Sig_r = apply_sigma(r)
    C_S1_Sy = float(-2.0 * (c @ Sig_r))

    # Work in increment space for S0/S1 moments
    # V_S1_alt = ones^T Sigma_u ones (should match V_S1 up to numerical tolerance)
    SigU_ones = apply_sigma_u(onesM)
    V_S1_inc = float(onesM @ SigU_ones)

    # Cov(S1, S0) = 2 ones^T Sigma_u mu_u
    SigU_mu = apply_sigma_u(mu_u)
    C_S1_S0 = float(2.0 * (onesM @ SigU_mu))

    # mu_u^T Sigma_u mu_u
    mu_SigU_mu = float(mu_u @ SigU_mu)

    # --- Exact trace(Sigma^2) and trace(A0 Sigma^2) via streaming basis solves ---
    tr_Sig2 = 0.0
    tr_A0Sig2 = 0.0
    # Solve J x_i = e_i for i=0..N-1
    e = np.zeros(N, dtype=float)
    for i in range(N):
        e.fill(0.0)
        e[i] = 1.0
        x = solve_banded((1, 1), ab, e)   # x = Sigma e_i
        tr_Sig2 += float(x @ x)
        Ax = apply_A0(x)
        tr_A0Sig2 += float(x @ Ax)

    # --- Exact trace(Sigma_u^2) via streaming basis in increment space ---
    tr_SigU2 = 0.0
    ej = np.zeros(M, dtype=float)
    for j in range(M):
        ej.fill(0.0)
        ej[j] = 1.0
        z = apply_sigma_u(ej)   # z = Sigma_u e_j
        tr_SigU2 += float(z @ z)

    # --- Assemble needed moment variances ---
    # Var(S0) = 2 tr(Sigma_u^2) + 4 mu_u^T Sigma_u mu_u
    Var_S0 = 2.0 * tr_SigU2 + 4.0 * mu_SigU_mu

    # Var(Sy) = 2 tr(Sigma^2) + 4 r^T Sigma r
    r_Sig_r = float(r @ Sig_r)
    Var_Sy = 2.0 * tr_Sig2 + 4.0 * r_Sig_r

    # --- Cov(Q, Sy) for Cov(lnD, lnR) ---
    # Cov(Q, Sy) =
    #   2 tr(A0 Sigma^2)
    # + 4 m^T A0 Sigma m
    # - 4 y^T Sigma A0 m
    # - 4 omega dt c^T Sigma m
    # + 4 omega dt c^T Sigma y
    Sig_m = apply_sigma(m)
    term_m_A0_Sig_m = float(m @ apply_A0(Sig_m))  # m^T A0 (Sigma m)

    A0m = apply_A0(m)
    Sig_A0m = apply_sigma(A0m)
    term_y_Sig_A0m = float(y @ Sig_A0m)

    term_c_Sig_m = float(c @ Sig_m)

    Sig_y = apply_sigma(y)
    term_c_Sig_y = float(c @ Sig_y)

    Cov_Q_Sy = (
        2.0 * tr_A0Sig2
        + 4.0 * term_m_A0_Sig_m
        - 4.0 * term_y_Sig_A0m
        - 4.0 * omega * dt * term_c_Sig_m
        + 4.0 * omega * dt * term_c_Sig_y
    )

    # --- Now compute I_miss entries for [omega, ln D, ln R] ---
    # s_omega = (S1 - M omega dt)/(2D)
    # s_lnD  = const + Q/(4 D dt)
    # s_lnR  = const + Sy/(2 R)

    I = np.zeros((3, 3), dtype=float)

    # Var(s_omega)
    I[0, 0] = V_S1 / (4.0 * D * D)

    # Cov(s_omega, s_lnD) = (1/(2D))*(1/(4Ddt)) * Cov(S1, Q)
    # Cov(S1,Q) = Cov(S1,S0) - 2 omega dt Var(S1)
    Cov_S1_Q = C_S1_S0 - 2.0 * omega * dt * V_S1
    I[0, 1] = Cov_S1_Q / (8.0 * D * D * dt)
    I[1, 0] = I[0, 1]

    # Var(s_lnD) = (1/(4Ddt))^2 Var(Q)
    # Var(Q) = Var(S0 - 2 omega dt S1)
    Var_Q = Var_S0 + 4.0 * (omega * dt) ** 2 * V_S1 - 4.0 * omega * dt * C_S1_S0
    I[1, 1] = Var_Q / ((4.0 * D * dt) ** 2)

    # Var(s_lnR) = (1/(2R))^2 Var(Sy) = Var(Sy)/(4R^2)
    I[2, 2] = Var_Sy / (4.0 * R * R)

    # Cov(s_omega, s_lnR) = (1/(2D))*(1/(2R)) Cov(S1,Sy)
    I[0, 2] = C_S1_Sy / (4.0 * D * R)
    I[2, 0] = I[0, 2]

    # Cov(s_lnD, s_lnR) = (1/(4Ddt))*(1/(2R)) Cov(Q,Sy)
    I[1, 2] = Cov_Q_Sy / (8.0 * D * R * dt)
    I[2, 1] = I[1, 2]

    # Optional sanity check: increment-space Var(S1) should match endpoint-space Var(S1)
    # (difference due to numerical error only)
    # You can uncomment this if you like:
    # if not np.isfinite(V_S1_inc) or abs(V_S1 - V_S1_inc) > 1e-6 * (1 + abs(V_S1)):
    #     print("Warning: Var(S1) mismatch between endpoint and increment computations:",
    #           V_S1, V_S1_inc)

    return I

    
def estimate_uncertainty(
    lambda_map,
    data,
    priors,
    statistics,
    method: Literal["MC", "exact"] = "exact",
    rng: Optional[np.random.Generator] = None,
    N_samples: Optional[int] = None,
):
    """Estimate covariance of inferred parameters and EM convergence rate.

    `method="exact"` uses analytic missing-information terms.
    `method="MC"` uses phase-trajectory sampling and requires `rng` and
    `N_samples`.
    """
    omega, Dtheta, R = lambda_map
    tk, yk = data
    mu_theta, sigma_theta, *_ = priors
    dt = tk[1]-tk[0]

    # Prior information matrix
    I_prior = prior_information(lambda_map, priors)

    # Complete-data information matrix
    I_comp = complete_data_information(lambda_map, statistics)
    
    # Missing information matrix
    if method=="exact":
        #### Filter and Smooth
        kalman_phases = kalman_filter(lambda_map, data, priors)
        thetakN, PkN, Sigmakk_1 = rts_smoother(lambda_map, data, kalman_phases)
        
        I_miss = missing_information(
            y=yk,       # shape (N,)
            m=thetakN,  # smoother mean at MAP, shape (N,)
            omega=omega,
            D=Dtheta,
            R=R,
            dt=dt,
            sigma_theta2=sigma_theta**2)

    elif method=="MC":
        if rng is None or N_samples is None:
            raise ValueError("For method='MC', provide rng and N_samples.")
        I_miss = missing_information_MC(lambda_map, data, priors, rng, N_samples)
        
    else:
        raise ValueError("method must be 'MC' or 'exact'.")
      
    # Observed information matrix
    I_obs = I_prior + I_comp - I_miss

    # Convergence rate
    rho = np.max(np.linalg.eigvals(I_miss@np.linalg.inv(I_prior + I_comp)))
    
    return np.linalg.inv(I_obs), -np.log(rho)


def infer_MAP_parameters(
    initial_params,
    data,
    priors,
    max_iter=500,
    tol=1e-5,
    verbose=True,
    print_every=10
):
    """Iterate EM/MAP updates until relative parameter-change convergence."""

    lambda_iter = np.zeros((max_iter,3))

    # Initialize
    lambda_iter[0,:] = np.array(initial_params)

    if verbose:
        print('iter\tomega\t\tDtheta\t\tR')

    converged = False  # track convergence

    for i in range(1, max_iter):

        # Update
        lambda_iter[i] = update_parameters(lambda_iter[i-1], data, priors)

        # Compute relative change
        denom = np.maximum(np.abs(lambda_iter[i-1]), 1e-12)  # avoid divide-by-zero
        rel_change = np.max(np.abs(lambda_iter[i] - lambda_iter[i-1]) / denom)

        # Print progress
        if verbose and (i % print_every == 0 or i == 1):
            print('%d\t%.5e\t%.5e\t%.5e' %
                  (i,
                   lambda_iter[i,0],
                   lambda_iter[i,1],
                   lambda_iter[i,2]))

        # Check convergence
        if rel_change < tol:
            converged = True
            if verbose:
                print(f'\nConverged at iteration {i} '
                      f'(max relative change = {rel_change:.2e})')
            break

    # If loop ended without convergence
    if not converged:
        warnings.warn(
            f"MAP inference did not converge within {max_iter} iterations "
            f"(last relative change = {rel_change:.2e}).",
            RuntimeWarning
        )
        
    # Truncate arrays to actual number of iterations used
    lambda_iter = lambda_iter[:i+1,:]

    return lambda_iter
    

def simulate_and_infer(param, N, dt, rng, priors, 
                       N_samples=1000, 
                       max_iter=500, 
                       tol=1e-5):
    """Simulate data, infer MAP parameters, and estimate uncertainty."""
        
    ## Simulate unwrapped phase data
    tk, thetak, yk = simulate_data(param, N, dt, rng, theta0 = 0)

    ## Compute MAP estimates
    data = (tk, yk)
    lambda_iter = infer_MAP_parameters(param, data, priors, max_iter=max_iter, tol=tol, verbose=False)
    lambda_map = lambda_iter[-1,:]

    # Filter and Smooth
    kalman_phases = kalman_filter(lambda_map, data, priors)
    thetakN, PkN, Sigmakk_1 = rts_smoother(lambda_map, data, kalman_phases)

    ## Statistics 
    N = yk.shape[0]
    dt = tk[1]-tk[0]
    S0 = np.sum((thetakN[1:] - thetakN[:-1])**2 + PkN[1:] + PkN[:-1] - 2*Sigmakk_1[1:])
    S1 = np.sum(thetakN[1:] - thetakN[:-1])
    Sy = np.sum((yk - thetakN)**2 + PkN)
    statistics = (N, dt, S0, S1, Sy)

    # ## Sample phases 
    # theta_samples = sample_phases(lambda_map, (tk,yk), rng, N_samples)

    # theta1_samples = theta_samples[:,0]
    # S0_samples = np.sum((theta_samples[:,1:] - theta_samples[:,:-1])**2, axis=1)
    # S1_samples = np.sum((theta_samples[:,1:] - theta_samples[:,:-1]), axis=1)
    # Sy_samples = np.sum((yk[None,:] - theta_samples)**2, axis=1)
    # samples = (theta1_samples, S0_samples, S1_samples, Sy_samples)

    # Estimate uncertainty
    lambda_cov, em_rate = estimate_uncertainty(lambda_map, data, priors, statistics)
   
    return lambda_map, lambda_cov, em_rate

def cov_mc_se_wishart(I_mc: np.ndarray, S: int) -> np.ndarray:
    """Approx SE of sample covariance entries under Wishart/CLT."""
    I_mc = np.asarray(I_mc, float)
    p = I_mc.shape[0]
    se = np.zeros_like(I_mc)
    for i in range(p):
        for j in range(p):
            se[i, j] = np.sqrt((I_mc[i, i]*I_mc[j, j] + I_mc[i, j]**2) / S)
    return se

def assess_diff_vs_mc(I_analytic, I_mc, N_samples):
    # score_samples = np.asarray(score_samples, float)
    # S, p = score_samples.shape
    # if p != 3:
    #     raise ValueError("Expected score_samples shape (S,3)")

    # if I_mc is None:
    #     I_mc = np.cov(score_samples, rowvar=False, bias=False)

    diff = I_analytic - I_mc
    se = cov_mc_se_wishart(I_mc, N_samples)
    z = diff / se

    print("S =", N_samples)
    print("\nEntrywise SE (approx):\n", se)
    print("\nZ-scores (diff / SE):\n", z)
    print("\nMax |Z|:", np.max(np.abs(z)))

    # A simple global summary: RMS Z over unique entries
    idx = np.triu_indices(3)
    rms_z = np.sqrt(np.mean(z[idx]**2))
    print("RMS Z (upper triangle):", rms_z)

    return {"I_mc": I_mc, "diff": diff, "se": se, "z": z}
