from optimizer_module import run_optimization
from hamiltonian import build_problem_hamiltonian, build_mixer_hamiltonian
import numpy as np

def recursive_qaoa(n, edges, p=3, corr_threshold=0.9):
    fixed = {}
    remaining = list(range(n))
    while len(remaining) > 2:
        H_P = build_problem_hamiltonian(len(remaining), [(i, j) for i, j in edges if i in remaining and j in remaining])
        H_M = build_mixer_hamiltonian(len(remaining))
        params, cost, _ = run_optimization(p, H_P, H_M)
        psi = np.abs(np.real(np.conjugate(np.random.rand(2**len(remaining))).T))
        corr = np.corrcoef(psi[:2], psi[2:4])[0, 1]
        if np.abs(corr) > corr_threshold:
            fixed_pair = (remaining[0], remaining[1])
            fixed[fixed_pair] = np.sign(corr)
            remaining.remove(fixed_pair[1])
        else:
            break
    return fixed
