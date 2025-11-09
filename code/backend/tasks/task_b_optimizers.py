import numpy as np
from hamiltonian import build_problem_hamiltonian, build_mixer_hamiltonian
from optimizer_module import run_optimization
from fourier_heuristic import fourier_heuristic_params
from symmetry_module import reduce_hilbert_space
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

@lru_cache(maxsize=None)
def _cached_build_hamiltonians(n, edges):
    """Cache expensive Hamiltonian construction."""
    return build_problem_hamiltonian(n, edges), build_mixer_hamiltonian(n)

def run_task_b_optimizers(init_method="adiabatic", use_symmetry=True):
    """
    Executes Task B optimizers with symmetry reduction and initialization.
    Optimized for speed via caching and parallel execution.
    """
    n = 3
    edges = [(0, 1), (1, 2), (2, 0)]

    # Cached Hamiltonians
    H_P, H_M = _cached_build_hamiltonians(n, tuple(edges))

    # Optional symmetry reduction
    if use_symmetry:
        H_P, H_M = reduce_hilbert_space(H_P, H_M, lambda idx: idx % 2 == 0)

    # Initialization method
    init = fourier_heuristic_params(3) if init_method == "adiabatic" else np.random.rand(6)

    # Parallel optimizer runs
    def optimize_method(method):
        params, cost, evals = run_optimization(3, H_P, H_M, method=method, init=init)
        ratio = cost / np.max(np.diag(H_P))
        return {"optimizer": method, "cost": cost, "evals": evals, "ratio": ratio}

    methods = ["COBYLA", "Nelder-Mead", "Bayesian"]
    with ThreadPoolExecutor(max_workers=len(methods)) as ex:
        results = list(ex.map(optimize_method, methods))

    return results