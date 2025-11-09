from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit import QuantumCircuit
import numpy as np

def get_noise_model(prob=0.02):
    model = NoiseModel()
    dep_err = depolarizing_error(prob, 1)
    for gate in ['u1', 'u2', 'u3']:
        model.add_all_qubit_quantum_error(dep_err, gate)
    return model

def apply_noise_to_state(psi, prob=0.02):
    return (1 - prob) * psi + prob * np.random.normal(0, 0.05, psi.shape)
