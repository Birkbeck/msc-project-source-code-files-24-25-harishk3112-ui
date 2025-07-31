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