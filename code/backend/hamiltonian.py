import numpy as np

def build_operator_kron(n, op_map):
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    op_dict = {'X': X, 'Z': Z, 'I': I}
    ops = [op_dict.get(op_map.get(i, 'I')) for i in range(n)]
    op = ops[0]
    for i in range(1, n):
        op = np.kron(op, ops[i])
    return op

def build_problem_hamiltonian(n, edges):
    dim = 2**n
    I_full = np.eye(dim)
    H_P = np.zeros((dim, dim))
    for (i, j) in edges:
        H_P += 0.5 * (I_full - build_operator_kron(n, {i: 'Z', j: 'Z'}))
    return H_P

def build_mixer_hamiltonian(n):
    H_M = np.zeros((2**n, 2**n))
    for i in range(n):
        H_M += build_operator_kron(n, {i: 'X'})
    return H_M
