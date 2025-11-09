document.addEventListener("DOMContentLoaded", () => {
  const summary = document.getElementById("summary");
  const runButton = document.getElementById("runButton");
  const qaoaChartCanvas = document.getElementById("qaoaChart");
  const tqaChartCanvas = document.getElementById("tqaChart");
  const optimizerChartCanvas = document.getElementById("optimizerChart");
  const BACKEND = "http://127.0.0.1:5000";

  let isRunning = false;
  runButton.addEventListener("click", startComputation);

  async function startComputation() {
    if (isRunning) {
      summary.textContent = "Simulation already running. Wait until it finishes.";
      return;
    }
    isRunning = true;
    runButton.disabled = true;
    runButton.textContent = "Running…";
    summary.textContent = "Resetting environment…";

    try {
      const resetRes = await fetch(`${BACKEND}/reset`, { method: "POST" });
      if (!resetRes.ok) throw new Error("reset failed");
      await new Promise(r => setTimeout(r, 800));
    } catch (err) {
      summary.textContent = "Reset failed.";
      console.error(err);
      finishRun();
      return;
    }

    summary.textContent = "Starting computation…";
    try {
      const r = await fetch(`${BACKEND}/results`);
      if (!r.ok) throw new Error(`results call failed: ${r.status}`);
      const j = await r.json();
      if (j.status === "processing") {
        summary.textContent = "Processing…";
        pollStatus();
      } else if (j.status === "done") {
        simulateWaitThenShow(j);
      } else {
        summary.textContent = `Unexpected response: ${JSON.stringify(j)}`;
        finishRun();
      }
    } catch (err) {
      summary.textContent = "Failed to reach backend.";
      console.error(err);
      finishRun();
    }
  }

  async function pollStatus() {
    try {
      const resp = await fetch(`${BACKEND}/status`);
      if (!resp.ok) throw new Error("status failed");
      const data = await resp.json();

      if (data.status === "processing") {
        summary.textContent = summary.textContent.endsWith("...") ? "Processing" : summary.textContent + ".";
        setTimeout(pollStatus, 1500);
        return;
      }
      if (data.status === "done") {
        simulateWaitThenShow(data);
        return;
      }
      summary.textContent = "Unexpected backend status.";
      finishRun();
    } catch (err) {
      summary.textContent = "Error polling backend.";
      console.error(err);
      finishRun();
    }
  }

  async function waitForPlots(data) {
    const plotKeys = Object.keys(data.plots || {});
    const maxWait = 10000;
    const start = Date.now();
    while (Date.now() - start < maxWait) {
      const ok = await Promise.all(
        plotKeys.map(k =>
          fetch(`${BACKEND}${data.plots[k]}?t=${Date.now()}`, { method: "HEAD" })
            .then(r => r.ok)
            .catch(() => false)
        )
      );
      if (ok.every(Boolean)) return true;
      await new Promise(r => setTimeout(r, 500));
    }
    return false;
  }

  function simulateWaitThenShow(data) {
    const execSeconds = Math.max(1.0, Number(data.execution_time || 3));
    const total = Math.max(2000, Math.round(execSeconds * 1000));
    let elapsed = 0;
    const step = 250;
    summary.textContent = "Finalizing calculations... 0%";
    const timer = setInterval(() => {
      elapsed += step;
      const pct = Math.min(100, Math.round((elapsed / total) * 100));
      summary.textContent = `Finalizing calculations... ${pct}%`;
      if (elapsed >= total) {
        clearInterval(timer);
        waitForPlots(data).then(() => {
          renderResults(data);
          finishRun();
        });
      }
    }, step);
  }

  function renderResults(data) {
    summary.textContent =
      `Best Approximation Ratio: ${Number(data.best_ratio).toFixed(3)}\n` +
      `Execution Time: ${Number(data.execution_time).toFixed(2)} s\n` +
      `Best Optimizer: ${data.optimizer}\n\n` +
      `QAOA Ratios: ${data.performance_ratios.map(v => Number(v).toFixed(3)).join(", ")}\n` +
      `TQA Ratios: ${data.tqa_ratios.map(v => Number(v).toFixed(3)).join(", ")}`;

    if (window.qaoaChartInstance) window.qaoaChartInstance.destroy();
    if (window.tqaChartInstance) window.tqaChartInstance.destroy();
    if (window.optChartInstance) window.optChartInstance.destroy();

    // QAOA chart
    const qctx = qaoaChartCanvas.getContext("2d");
    window.qaoaChartInstance = new Chart(qctx, {
      type: "bar",
      data: { labels: data.p_values, datasets: [{ label: "QAOA", data: data.performance_ratios }] },
      options: { scales: { y: { beginAtZero: true, max: 1.05 } } }
    });

    // TQA chart
    const tctx = tqaChartCanvas.getContext("2d");
    window.tqaChartInstance = new Chart(tctx, {
      type: "bar",
      data: { labels: data.p_values, datasets: [{ label: "TQA", data: data.tqa_ratios }] },
      options: { scales: { y: { beginAtZero: true, max: 1.05 } } }
    });

    // Optimizer performance chart
    const octx = optimizerChartCanvas.getContext("2d");
    window.optChartInstance = new Chart(octx, {
      type: "bar",
      data: {
        labels: data.optimizer_labels,
        datasets: [
          { label: "Optimizer Scores", data: data.optimizer_scores, backgroundColor: "rgba(0,255,0,0.6)" }
        ]
      },
      options: { scales: { y: { beginAtZero: true, max: 1.05 } } }
    });

    // Load static image plots
    const map = {
      energy_landscape: "energyPlot",
      correlation_heatmap: "corrPlot",
      noise_sensitivity: "noisePlot",
      adiabatic_fidelity: "fidelityPlot",
      qaoa_vs_tqa: "qaoaTqaPlot",
      params_schedule: "paramsPlot",
    };
    Object.entries(map).forEach(([k, v]) => {
      const el = document.getElementById(v);
      if (el && data.plots[k]) el.src = `${BACKEND}/../output/static/${data.plots[k]}?t=${Date.now()}`;
    });
  }

  function finishRun() {
    isRunning = false;
    runButton.disabled = false;
    runButton.textContent = "▶ Run Simulation";
  }
});
