import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend (thread-safe for Flask)
import matplotlib.pyplot as plt
from qaoa_core import qaoa_expectation
from adiabatic import simulate_continuous_adiabatic

os.makedirs("backend/static", exist_ok=True)

# ---------- Energy Landscape ----------
def plot_energy_landscape(H_P, H_M, p=1, gamma_range=(0, np.pi), beta_range=(0, np.pi), points=35):
    """
    Plots ⟨C⟩(γ, β) as a heatmap for QAOA layer depth p=1.
    Thread-safe, prevents invalid array shape errors.
    """
    gammas = np.linspace(*gamma_range, points)
    betas = np.linspace(*beta_range, points)
    Z = np.zeros((len(gammas), len(betas)))

    for i, g in enumerate(gammas):
        for j, b in enumerate(betas):
            params = np.array([g, b])
            try:
                Z[i, j] = qaoa_expectation(params, p, H_P, H_M)
            except Exception:
                Z[i, j] = np.nan

    if np.isnan(Z).any():
        Z = np.nan_to_num(Z, nan=np.nanmean(Z[np.isfinite(Z)]))

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        Z.T,
        extent=[gamma_range[0], gamma_range[1], beta_range[0], beta_range[1]],
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )
    plt.xlabel(r"Gamma (γ)")
    plt.ylabel(r"Beta (β)")
    plt.title(r"QAOA Energy Landscape <C>(γ, β)")
    plt.colorbar(im, label="Expected Cost")
    plt.tight_layout()
    plt.savefig("backend/static/energy_landscape.png", dpi=120)
    plt.close()


# ---------- Correlation Heatmap ----------
def plot_correlation_heatmap(corrs, title="RQAOA Correlation Heatmap"):
    """
    Plot ⟨Z_i Z_j⟩ correlation matrix for visualization in RQAOA.
    """
    plt.figure(figsize=(6, 5))
    if not corrs:
        plt.text(0.5, 0.5, "No correlations measured",
                 ha="center", va="center", fontsize=12)
        plt.axis("off")
    else:
        m = len(corrs)
        n = int((1 + np.sqrt(1 + 8 * m)) / 2)
        mat = np.zeros((n, n))
        for (i, j), val in corrs.items():
            mat[i, j] = mat[j, i] = val
        im = plt.imshow(mat, cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(im, label="⟨ZᵢZⱼ⟩")
        plt.xlabel("Qubit j")
        plt.ylabel("Qubit i")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("backend/static/correlation_heatmap.png", dpi=120)
    plt.close()


# ---------- Noise Sensitivity ----------
def plot_noise_vs_ratio(noise_levels, ratios):
    """
    Plot performance degradation due to depolarizing noise.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(noise_levels, ratios, 'o-')
    plt.xlabel("Depolarizing Probability")
    plt.ylabel("Approximation Ratio")
    plt.title("Noise Sensitivity of QAOA")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("backend/static/noise_sensitivity.png", dpi=120)
    plt.close()


# ---------- Fidelity vs Time (Adiabatic Evolution) ----------
def plot_adiabatic_fidelity(H_P, H_M, T=10.0, steps=200):
    """
    Plot the ground-state fidelity over time for adiabatic evolution.
    """
    times, fidelities, _ = simulate_continuous_adiabatic(H_P, H_M, T=T, steps=steps)
    plt.figure(figsize=(7, 4))
    plt.plot(times, fidelities, '-')
    plt.xlabel("Time")
    plt.ylabel("Ground-state Fidelity")
    plt.title("Continuous Adiabatic Evolution: Fidelity vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("backend/static/adiabatic_fidelity.png", dpi=120)
    plt.close()


# ---------- QAOA vs TQA ----------
def plot_qaoa_vs_tqa(p_values, ratios_qaoa, ratios_tqa):
    """
    Compare optimized QAOA performance with TQA baseline.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(p_values, ratios_qaoa, 'o-', label="QAOA (optimized)")
    plt.plot(p_values, ratios_tqa, 'x--', label="TQA (adiabatic)")
    plt.axhline(1.0, linestyle="--", color="gray")
    plt.xlabel("Depth p")
    plt.ylabel("Approximation Ratio ⟨C⟩ / C_max")
    plt.title("QAOA vs TQA Performance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("backend/static/qaoa_vs_tqa.png", dpi=120)
    plt.close()


# ---------- Parameter Schedules ----------
def plot_param_schedules(gammas, betas, title="Optimized parameter schedules"):
    """
    Plot optimized γ_k and β_k parameter evolution across QAOA layers.
    """
    p = len(gammas)
    ks = np.arange(1, p + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(ks, gammas, 'o-', label='γ_k')
    plt.plot(ks, betas, 'x--', label='β_k')
    plt.xlabel("Layer k")
    plt.ylabel("Parameter value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("backend/static/params_schedule.png", dpi=120)
    plt.close()