let forecastChart, historyChart;
let latestData = {};

document.getElementById("forecastDays").addEventListener("input", function () {
  document.getElementById("dayRangeLabel").innerText = `${this.value} Day(s)`;
});

function toggleDarkMode() {
  document.body.classList.toggle("dark-mode");
}

function submitInput() {
  const size = document.getElementById("householdSize").value;
  const preference = document.getElementById("preference").value;
  const peakRaw = document.getElementById("peakHours").value;
  const forecastDays = parseInt(document.getElementById("forecastDays").value);
  const peak_hours = peakRaw
    .split(",")
    .map((s) => s.trim())
    .filter((x) => x);

  localStorage.setItem(
    "inputData",
    JSON.stringify({ size, preference, peakRaw, forecastDays })
  );
}

 document.getElementById("fullscreenLoader").classList.remove("d-none");

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      household_size: size,
      preference,
      peak_hours,
      forecast_days: forecastDays,
    }),
  })
    .then((res) => res.json())
    .then((data) => {
      latestData = data;

      renderChart(data.hours, data.values, "forecastChart", forecastChart);
      renderChart(
        Array.from({ length: 24 }, (_, i) => `Hour ${i + 1}`),
        data.history,
        "historyChart",
        historyChart
      );

      forecastChart = Chart.getChart("forecastChart");
      historyChart = Chart.getChart("historyChart");

      document.getElementById("recommendation").textContent =
        data.recommendation;
      document.getElementById("recommendation").style.display = "block";

      document.getElementById(
        "summaryStats"
      ).innerHTML = `Average: ${data.summary.avg} kW | Max: ${data.summary.max} kW | Min: ${data.summary.min} kW`;
      document.getElementById("summaryStats").style.display = "block";

      document.getElementById("fullscreenLoader").classList.add("d-none");
    });



function renderChart(labels, data, canvasId, chartInstance) {
  if (chartInstance) chartInstance.destroy();
  const ctx = document.getElementById(canvasId).getContext("2d");
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
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true } },
    },
  });
}

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
const saved = JSON.parse(localStorage.getItem("inputData") || "{}");
if (saved.size) document.getElementById("householdSize").value = saved.size;
if (saved.preference)
  document.getElementById("preference").value = saved.preference;
if (saved.peakRaw) document.getElementById("peakHours").value = saved.peakRaw;
if (saved.forecastDays) {
  document.getElementById("forecastDays").value = saved.forecastDays;
  document.getElementById(
    "dayRangeLabel"
  ).innerText = `${saved.forecastDays} Day(s)`;
}
document.getElementById("forecastDays").addEventListener("input", function () {
  const label = document.getElementById("dayRangeLabel");
  label.innerText = this.value === "1" ? "1 Day" : `${this.value} Days`;
});

submitInput();