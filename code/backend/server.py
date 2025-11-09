# server.py
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np, os, time, json, traceback, shutil, threading

from hamiltonian import build_problem_hamiltonian, build_mixer_hamiltonian
from optimizer_module import run_optimization
from fourier_heuristic import fourier_heuristic_params
from recursive_qaoa import recursive_qaoa
from visualization_module import (
    plot_energy_landscape, plot_correlation_heatmap,
    plot_noise_vs_ratio, plot_adiabatic_fidelity,
    plot_qaoa_vs_tqa, plot_param_schedules
)
from qaoa_core import qaoa_expectation
from adiabatic import tqa_params

# ---------------------------------------------------------------------
# PATH SETUP — handles the correct directory for output outside /code

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # root above /code
BACKEND_STATIC_DIR = os.path.join(BASE_DIR, "output", "static")
os.makedirs(BACKEND_STATIC_DIR, exist_ok=True)

RESULT_FILE = os.path.join(BACKEND_STATIC_DIR, "results.json")

# ---------------------------------------------------------------------

app = Flask(
    __name__,
    static_folder="../frontend",  # Serve frontend folder
    static_url_path=""            # Available at root
)
CORS(app, resources={r"/*": {"origins": "*"}})

is_computing = False
lock = threading.Lock()

# ---------------------------------------------------------------------

def compute_results():
    """Main QAOA + visualization run."""
    try:
        # Cleanup old files
        for f in os.listdir(BACKEND_STATIC_DIR):
            fp = os.path.join(BACKEND_STATIC_DIR, f)
            if os.path.isfile(fp) or os.path.islink(fp):
                os.unlink(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)

        n = 3
        edges = [(0, 1), (1, 2), (2, 0)]
        H_P = build_problem_hamiltonian(n, edges)
        H_M = build_mixer_hamiltonian(n)
        C_max = float(np.max(np.diag(H_P)))

        p_values = [1, 2, 3]
        ratios_qaoa, ratios_tqa = [], []
        best_params_for_plot = None
        start = time.time()

        for p in p_values:
            init = fourier_heuristic_params(p)
            best_cost, best_params = -1e9, None
            for method in ["COBYLA", "Nelder-Mead", "Bayesian"]:
                params, cost, _ = run_optimization(p, H_P, H_M, method=method, init=init)
                if cost > best_cost:
                    best_cost, best_params = cost, np.array(params)
            ratios_qaoa.append(float(best_cost / C_max))
            if p == 3:
                best_params_for_plot = best_params

            gammas_tqa, betas_tqa = tqa_params(p, T=5.0)
            params_tqa = np.concatenate([gammas_tqa, betas_tqa])
            tqa_cost = qaoa_expectation(params_tqa, p, H_P, H_M)
            ratios_tqa.append(float(tqa_cost / C_max))

        duration = time.time() - start
        corrs = recursive_qaoa(n, edges)
        corrs_clean = {k: float(v) for k, v in corrs.items()} if isinstance(corrs, dict) else {}

        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
        base_ratio = ratios_qaoa[-1]
        noisy_ratios = [max(0.0, base_ratio * (1 - 1.5 * nl)) for nl in noise_levels]

        # --- generate plots (all saved to BACKEND_STATIC_DIR) ---
        plot_energy_landscape(H_P, H_M, p=1)
        plot_correlation_heatmap(corrs_clean)
        plot_noise_vs_ratio(noise_levels, noisy_ratios)
        plot_adiabatic_fidelity(H_P, H_M, T=10.0, steps=300)
        plot_qaoa_vs_tqa(p_values, ratios_qaoa, ratios_tqa)

        if best_params_for_plot is not None:
            p = 3
            gammas = best_params_for_plot[:p]
            betas = best_params_for_plot[p:2*p]
            plot_param_schedules(gammas, betas)

        optimizer_labels = ["COBYLA", "Nelder-Mead", "Bayesian"]
        optimizer_scores = []
        init_p3 = fourier_heuristic_params(3)
        for method in optimizer_labels:
            params, cost, _ = run_optimization(3, H_P, H_M, method=method, init=init_p3)
            optimizer_scores.append(float(cost / C_max))

        result = {
            "status": "done",
            "best_ratio": float(max(ratios_qaoa)),
            "execution_time": float(duration),
            "optimizer": "Hybrid (best of COBYLA/NM/Bayesian)",
            "performance_ratios": ratios_qaoa,
            "tqa_ratios": ratios_tqa,
            "p_values": p_values,
            "optimizer_labels": optimizer_labels,
            "optimizer_scores": optimizer_scores,
            "plots": {
                "energy_landscape": "energy_landscape.png",
                "correlation_heatmap": "correlation_heatmap.png",
                "noise_sensitivity": "noise_sensitivity.png",
                "adiabatic_fidelity": "adiabatic_fidelity.png",
                "qaoa_vs_tqa": "qaoa_vs_tqa.png",
                "params_schedule": "params_schedule.png",
            },
        }

        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        traceback.print_exc()
        err = {"status": "error", "message": str(e)}
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(err, f)
        return err

# ---------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/output/static/<path:filename>")
def serve_backend_static(filename):
    """Serve images and results from /output/static (outside /code)."""
    abs_path = os.path.abspath(os.path.join(BACKEND_STATIC_DIR, filename))
    return send_from_directory(BACKEND_STATIC_DIR, os.path.basename(abs_path))

@app.route("/results", methods=["GET"])
def results():
    global is_computing

    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "done":
                return jsonify(data)
        except json.JSONDecodeError:
            pass

    with lock:
        if is_computing:
            return jsonify({"status": "processing"})
        is_computing = True

    def background_task():
        global is_computing
        try:
            compute_results()
        finally:
            with lock:
                is_computing = False

    threading.Thread(target=background_task, daemon=True).start()
    return jsonify({"status": "processing"})

@app.route("/status", methods=["GET"])
def status():
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        except json.JSONDecodeError:
            return jsonify({"status": "processing"})
    return jsonify({"status": "processing"})

@app.route("/reset", methods=["POST"])
def reset_results():
    try:
        for f in os.listdir(BACKEND_STATIC_DIR):
            fp = os.path.join(BACKEND_STATIC_DIR, f)
            if os.path.isfile(fp) or os.path.islink(fp):
                os.unlink(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump({"status": "processing"}, f)
        return jsonify({"status": "reset"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
