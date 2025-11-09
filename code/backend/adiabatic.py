import numpy as np
from scipy.linalg import expm

def uniform_plus_state(n: int) -> np.ndarray:
    dim = 2 ** n
    return (1 / np.sqrt(dim)) * np.ones(dim, dtype=complex)

def simulate_continuous_adiabatic(H_P, H_M, T=6.0, steps=100):
    """
    Continuous-time adiabatic evolution with linear schedule:
      H(s) = (1 - s) H_M + s H_P,  s = t/T \in [0, 1]
    Returns:
      times        : np.array of times
      fidelities   : fidelity to ground state of H_P vs time
      final_state  : state at t = T
    """
    dim = H_P.shape[0]
    n = int(np.log2(dim))
    psi = uniform_plus_state(n)

    # Ground state of H_P for fidelity reference (lowest eigenvalue)
    evals, evecs = np.linalg.eigh(H_P)
    gs = evecs[:, np.argmin(evals)]

    dt = T / steps
    times = np.linspace(0, T, steps + 1)
    fidelities = np.zeros(steps + 1)
    fidelities[0] = np.abs(np.vdot(gs, psi)) ** 2

    for t_idx in range(steps):
        t = times[t_idx]
        s = t / T
        H_t = (1.0 - s) * H_M + s * H_P
        U_dt = expm(-1j * H_t * dt)
        psi = U_dt @ psi
        fidelities[t_idx + 1] = np.abs(np.vdot(gs, psi)) ** 2

    return times, fidelities, psi

def tqa_params(p: int, T: float = 5.0):
    """
    Returns TQA (discretized adiabatic) parameters for depth p:
      γ_k = s_k * Δt, β_k = (1 - s_k) * Δt,  s_k = (k+1)/p
    """
    gammas = np.zeros(p)
    betas = np.zeros(p)
    dt = T / p
    for k in range(p):
        s_k = (k + 1) / p
        gammas[k] = s_k * dt
        betas[k] = (1.0 - s_k) * dt
    return gammas, betas
