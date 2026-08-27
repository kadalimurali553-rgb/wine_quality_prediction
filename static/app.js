
const form = document.getElementById("predictForm");
const resetBtn = document.getElementById("resetBtn");
const empty = document.getElementById("resultEmpty");
const result = document.getElementById("resultValue");
const quality = document.getElementById("quality");
const qualityLabel = document.getElementById("qualityLabel");
const meter = document.getElementById("meterFill");
const resultText = document.getElementById("resultText");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(form).entries());
  for (const k in data) data[k] = Number(data[k]);

  const btn = form.querySelector(".predict-btn");
  btn.disabled = true;
  btn.innerHTML = "Running SVM <span>…</span>";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(data)
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Prediction failed.");

    empty.classList.add("hidden");
    result.classList.remove("hidden");
    quality.textContent = payload.quality;
    qualityLabel.textContent = payload.label;
    meter.style.width = `${Math.min(100, payload.quality * 10)}%`;
    resultText.textContent = `The SVM classified this sample as ${payload.label.toLowerCase()} quality.`;
  } catch (err) {
    empty.classList.remove("hidden");
    result.classList.add("hidden");
    empty.querySelector("h3").textContent = "Model unavailable";
    empty.querySelector("p").textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Predict wine quality <span>→</span>';
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  empty.classList.remove("hidden");
  result.classList.add("hidden");
  empty.querySelector("h3").textContent = "Awaiting a sample";
  empty.querySelector("p").textContent = "Enter the wine's chemical properties and run the SVM model.";
  meter.style.width = "0";
});
