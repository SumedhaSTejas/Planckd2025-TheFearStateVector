import numpy as np
from hamiltonian import build_problem_hamiltonian

def test_problem_hamiltonian_maxcut():
    edges = [(0, 1), (1, 2), (2, 0)]
    H_P = build_problem_hamiltonian(3, edges)
    diag = np.diag(H_P)
    assert len(diag) == 8 and np.isclose(diag.max(), 2.0)
