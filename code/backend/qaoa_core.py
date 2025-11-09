import numpy as np
from scipy.linalg import expm
from functools import lru_cache

@lru_cache(maxsize=None)
def _cached_exp(H_key, angle, dim):
    H = np.array(H_key, dtype=np.complex128).reshape((dim, dim))
    return expm(-1j * angle * H)

def qaoa_state(params, p, H_P, H_M):
    gammas = params[:p]
    betas = params[p:]
    dim = H_P.shape[0]
    psi = (1 / np.sqrt(dim)) * np.ones(dim, dtype=complex)
    diag_HP = np.diag(H_P)

    for k in range(p):
        # diagonal exponential = vectorized element-wise multiply
        psi *= np.exp(-1j * gammas[k] * diag_HP)
        U_M = _cached_exp(tuple(H_M.flatten()), betas[k], H_M.shape[0])
        psi = U_M @ psi

    return psi

def qaoa_expectation(params, p, H_P, H_M):
    psi = qaoa_state(params, p, H_P, H_M)
    return np.real(np.vdot(psi, H_P @ psi))