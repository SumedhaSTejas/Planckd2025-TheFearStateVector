import numpy as np

def fourier_heuristic_params(p, base_amp=1.0):
    k = np.arange(1, p + 1)
    gammas = base_amp * np.sin(np.pi * k / (2 * p))
    betas = base_amp * np.cos(np.pi * k / (2 * p))
    return np.concatenate([gammas, betas])
