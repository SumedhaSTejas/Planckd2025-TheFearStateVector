import numpy as np
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from qaoa_core import qaoa_expectation
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def run_bayesian_opt(objective, p):
    """
    Runs Bayesian optimization using Gaussian Process surrogate with a broad kernel range.
    """
    bounds = [Real(0.0, np.pi, name=f"param_{i}") for i in range(2 * p)]

    @use_named_args(bounds)
    def objective_wrapper(**params):
        x = np.array(list(params.values()))
        return objective(x)

    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e5))

    res = gp_minimize(
        func=objective_wrapper,
        dimensions=bounds,
        acq_func="EI",
        n_calls=25,
        random_state=42,
        n_initial_points=10,
        noise=1e-6,
        base_estimator=None
    )
    return np.array(res.x), -res.fun, len(res.func_vals)


def run_optimization(p, H_P, H_M, method="COBYLA", init=None):
    if init is None:
        init = np.random.uniform(0, np.pi, 2 * p)

    objective = lambda params: -qaoa_expectation(params, p, H_P, H_M)

    if method == "Nelder-Mead":
        res = minimize(objective, init, method=method,
                       options={"maxiter": 100, "fatol": 1e-4})
        return res.x, -res.fun, res.nfev

    elif method == "COBYLA":
        res = minimize(objective, init, method=method,
                       options={"maxiter": 100, "tol": 1e-4})
        return res.x, -res.fun, res.nfev

    elif method == "Bayesian":
        return run_bayesian_opt(objective, p)

    else:
        raise ValueError(f"Unknown optimization method: {method}")
