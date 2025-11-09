# Quantum QAOA: Adiabatic Connection Simulation

### Features
- Discretized adiabatic evolution (Trotter-Suzuki)
- Multi-optimizer support (COBYLA, Nelder-Mead, Bayesian)
- Fourier heuristic parameter initialization
- Noise-aware simulation (Depolarizing Error)
- Symmetry-Aware reduction of Hilbert space
- Recursive QAOA (RQAOA) problem-size reduction

---

### Requirements
Install dependencies
os, numpy, matplotlib, qaoa_core, adiabatic, hamiltonian, optimizer_module, fourier_heuristic, symmetry_module, concurrent.futures, functools, scipy, qiskit_aer, qiskit, skopt, sklearn, flask, flask_cors, visualization_module, 

### Execution
Backend simulation:
python code/backend/server.py

Frontend Visualization:
open the frontend on http://127.0.0.1:5000

### Result
The result is dynamic with different noise profiles.
It can be viewed seperately as only images and a json file in the output folder

### Our Team

Name:- The Fear State Vector

Members:-
Paarth Sarthi ( Team leader ) 2nd year 
Sumedha S. Tejas 2nd year
Anshreet 1st year
Aman 1st year
