// charts & state 
let forecastChart, historyChart, residualsChart;
let latestData = {};

// helpers 
function $(id) {
  return document.getElementById(id);
}
function showLoader(v) {
  const el = $("fullscreenLoader");
  if (el) el.classList.toggle("d-none", !v);
}

// Update the "Days" label
$("forecastDays").addEventListener("input", function () {
  const v = this.value;
  $("dayRangeLabel").innerText = v === "1" ? "1 Day" : `${v} Days`;
});

// Optional theme toggle if you want to use it somewhere
function toggleDarkMode() {
  document.body.classList.toggle("dark-mode");
}

// Card stat helper for metrics panel
function cardStat(name, val) {
  return `
    <div class="col-6 col-md-3">
      <div class="border rounded p-3 h-100">
        <div class="text-muted small">${name}</div>
        <div class="fs-5 fw-semibold">${val}</div>
      </div>
    </div>`;
}

// Render metrics panel if DOM exists
function renderMetrics(evalObj) {
  const panel = $("metricsPanel");
  const grid = $("metricsGrid");
  const win = $("metricsWindow");
  if (!panel || !grid) return;

  if (!evalObj) {
    panel.style.display = "none";
    return;
  }
  const m = evalObj.metrics || {};

  // ---- dynamic card title "last X days" ----
  const hours = m.window_hours ?? 168;
  const days = hours / 24;
  const titleEl = panel.querySelector("h5");
  if (titleEl) {
    titleEl.textContent = `Forecast Accuracy (last ${
      Number.isInteger(days) ? days : days.toFixed(1)
    } day${days === 1 ? "" : "s"})`;
  }

  grid.innerHTML = [
    cardStat("RMSE", m.RMSE ?? "—"),
    cardStat("MAE", m.MAE ?? "—"),
    cardStat("MAPE %", m["MAPE_%"] ?? "—"),
    cardStat("sMAPE %", m["sMAPE_%"] ?? "—"),
    cardStat("R²", m.R2 ?? "—"),
    cardStat("Within ±10% %", m["Within10pct_%"] ?? "—"),
    cardStat("Window (hours)", hours),
  ].join("");

  if (win) {
    const start = evalObj.start ? formatDatePretty(evalObj.start) : "—";
    const end = evalObj.end ? formatDatePretty(evalObj.end) : "—";
    win.innerText = `Evaluation window: ${start} → ${end}`;
  }

  panel.style.display = "block";
}

// Render residuals chart if DOM exists
function renderResiduals(evalObj) {
  const cv = $("residualsChart");
  if (!cv || !evalObj || !Array.isArray(evalObj.residuals)) return;

  const labels = Array.from(
    { length: evalObj.residuals.length },
    (_, i) => `t-${evalObj.residuals.length - i}`
  );
  const ctx = cv.getContext("2d");
  if (residualsChart) residualsChart.destroy();

  residualsChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Residual (actual − predicted)",
          data: evalObj.residuals,
          borderWidth: 2,
          tension: 0.2,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "Past hours" } },
        y: { title: { display: true, text: "kW (approx.)" } },
      },
    },
  });
}

// Training info (optional)
function renderTrainingInfo(trainInfo) {
  const panel = $("trainingPanel");
  const pre = $("trainingInfo");
  if (!panel || !pre) return;

  if (trainInfo) {
    pre.textContent = JSON.stringify(trainInfo, null, 2);
    panel.style.display = "block";
  } else {
    panel.style.display = "none";
  }
}

// Line chart renderer
function renderChart(labels, data, canvasId, chartInstance) {
  if (chartInstance) chartInstance.destroy();
  const ctx = $(canvasId).getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "kW Consumption",
          data,
          borderColor: "#007bff",
          backgroundColor: "rgba(0,123,255,0.1)",
          fill: true,
          tension: 0.4,
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true } },
    },
  });
}

// main submit 
async function submitInput() {
  const size = parseFloat($("householdSize").value || "1");
  const preference = $("preference").value;
  const peakRaw = $("peakHours").value;
  const forecastDays = parseInt($("forecastDays").value || "1", 10);
  const peak_hours = (peakRaw || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  // persist UI state
  const evalWindowHours = parseInt(document.getElementById("evalWindow").value, 10);
  const withinPct = parseFloat(document.getElementById("withinPct").value);
  
  localStorage.setItem(
    "inputData",
    JSON.stringify({ 
      size, 
      preference, 
      peakRaw, 
      forecastDays, 
      evalWindowHours, 
      withinPct 
    })
  );

  showLoader(true);
  try {
    // Get current evaluation window settings
    const evalWindowHours = parseInt(document.getElementById("evalWindow").value, 10);
    const withinPct = parseFloat(document.getElementById("withinPct").value);
    
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        household_size: size,
        preference,
        peak_hours,
        forecast_days: forecastDays,
        eval_window_hours: evalWindowHours,
        within_pct: withinPct,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");

    latestData = data;

    // Forecast
    renderChart(data.hours, data.values, "forecastChart", forecastChart);
    forecastChart = Chart.getChart("forecastChart");

    // History (last 24)
    renderChart(
      Array.from(
        { length: (data.history || []).length },
        (_, i) => `Hour ${i + 1}`
      ),
      data.history || [],
      "historyChart",
      historyChart
    );
    historyChart = Chart.getChart("historyChart");

    // Recommendation + summary
    if ($("recommendation")) {
      $("recommendation").textContent = data.recommendation || "";
      $("recommendation").style.display = data.recommendation
        ? "block"
        : "none";
    }
    if ($("summaryStats")) {
      const s = data.summary || {};
      $("summaryStats").innerHTML = `Average: ${s.avg ?? "—"} kW | Max: ${
        s.max ?? "—"
      } kW | Min: ${s.min ?? "—"} kW`;
      $("summaryStats").style.display = "block";
    }

    // NEW: metrics, residuals, training info
    renderMetrics(data.evaluation);
    renderResiduals(data.evaluation);
    renderTrainingInfo(data.training_info);
  } catch (err) {
    alert(err.message || "Something went wrong");
  } finally {
    showLoader(false);
  }
}

// CSV download
function downloadCSV() {
  if (!latestData.values) return alert("No forecast data available");
  const csv = [
    "Hour,Predicted(kW)",
    ...latestData.hours.map((h, i) => `${h},${latestData.values[i]}`),
  ].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "forecast.csv";
  a.click();
  URL.revokeObjectURL(url);
}

// restore saved UI & auto-run 
const saved = JSON.parse(localStorage.getItem("inputData") || "{}");
if (saved.size) $("householdSize").value = saved.size;
if (saved.preference) $("preference").value = saved.preference;
if (saved.peakRaw) $("peakHours").value = saved.peakRaw;
if (saved.forecastDays) {
  $("forecastDays").value = saved.forecastDays;
  $("dayRangeLabel").innerText =
    saved.forecastDays === 1 ? "1 Day" : `${saved.forecastDays} Days`;
}
if (saved.evalWindowHours) document.getElementById("evalWindow").value = saved.evalWindowHours;
if (saved.withinPct) document.getElementById("withinPct").value = saved.withinPct;

async function refreshMetrics() {
  const hours = parseInt(document.getElementById("evalWindow").value, 10);
  const pct = parseFloat(document.getElementById("withinPct").value);

  // Hit a small endpoint or reuse /predict with a flag; if you prefer not to
  // change the backend, you can recompute client-side from latestData.evaluation,
  // but server-side is more correct.
  showLoader(true);
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        household_size: document.getElementById("householdSize").value,
        preference: document.getElementById("preference").value,
        peak_hours: (document.getElementById("peakHours").value || "")
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        forecast_days: parseInt(
          document.getElementById("forecastDays").value,
          10
        ),
        eval_window_hours: hours, // < add to backend if you want
        within_pct: pct, // < add to backend if you want
      }),
    });
    const data = await res.json();
    latestData = data;
    renderMetrics(data.evaluation);
    renderResiduals(data.evaluation);
    renderBaseline(data.baseline);
  } finally {
    showLoader(false);
  }
}

function formatDatePretty(iso) {
  // Robust against plain 'YYYY-MM-DDTHH:mm:ss'
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

// Kick off an initial forecast
submitInput();
