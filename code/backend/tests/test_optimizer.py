import numpy as np
from hamiltonian import build_problem_hamiltonian, build_mixer_hamiltonian
from optimizer_module import run_optimization

def test_cobyla_optimization():
    n = 3
    edges = [(0, 1), (1, 2), (2, 0)]
    H_P = build_problem_hamiltonian(n, edges)
    H_M = build_mixer_hamiltonian(n)
    params, cost, evals = run_optimization(1, H_P, H_M, method="COBYLA")
    assert cost > 0
