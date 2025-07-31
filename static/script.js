

document.getElementById("forecastDays").addEventListener("input", function () {
  document.getElementById("dayRangeLabel").innerText = `${this.value} Day(s)`;
});

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