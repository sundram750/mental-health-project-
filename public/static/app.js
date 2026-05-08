const form = document.querySelector("#prediction-form");
const errorBox = document.querySelector("#form-error");
const submitButton = form.querySelector("button[type='submit']");
const modelPill = document.querySelector("#model-pill");
const emptyState = document.querySelector("#empty-state");
const resultView = document.querySelector("#result-view");
const details = document.querySelector("#details");
const historyKey = "neuropulse_readings_v2";

function setText(selector, value) {
  const node = document.querySelector(selector);
  if (node) node.textContent = value;
}

function listItems(selector, items) {
  const node = document.querySelector(selector);
  if (!node) return;
  node.innerHTML = "";
  (items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    node.appendChild(li);
  });
}

function setError(message) {
  errorBox.textContent = message || "";
}

function getFormData() {
  const values = Object.fromEntries(new FormData(form).entries());
  return {
    heart_rate: Number(values.heart_rate),
    hrv: Number(values.hrv),
    respiration: Number(values.respiration),
    skin_temp: Number(values.skin_temp),
    bp_systolic: Number(values.bp_systolic),
    bp_diastolic: Number(values.bp_diastolic),
    cognitive_state: Number(values.cognitive_state),
    emotional_state: Number(values.emotional_state),
  };
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(historyKey) || "[]");
  } catch (_error) {
    return [];
  }
}

function saveReading(reading) {
  const history = readHistory();
  history.push(reading);
  while (history.length > 20) history.shift();
  localStorage.setItem(historyKey, JSON.stringify(history));
}

function renderTrend() {
  const chart = document.querySelector("#trend-chart");
  if (!chart) return;
  const history = readHistory().slice(-8);
  if (!history.length) {
    chart.textContent = "No local readings yet.";
    return;
  }

  const width = 700;
  const height = 190;
  const pad = 24;
  const points = history.map((item, index) => {
    const x = history.length === 1
      ? width / 2
      : pad + (index / (history.length - 1)) * (width - pad * 2);
    const y = height - pad - (Number(item.mli) / 100) * (height - pad * 2);
    return { x, y, item };
  });

  const path = points.map((point, index) => {
    return `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
  }).join(" ");

  const dots = points.map((point) => {
    const label = `${point.item.condition}: ${point.item.mli}`;
    return `<circle cx="${point.x}" cy="${point.y}" r="5"><title>${label}</title></circle>`;
  }).join("");

  chart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Mental Load Index trend">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#cbd5e1" />
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#cbd5e1" />
      <path d="${path}" fill="none" stroke="#0f9f9a" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
      <g fill="#d95d45" stroke="#ffffff" stroke-width="2">${dots}</g>
      <text x="${pad}" y="16" fill="#687385" font-size="13">MLI 100</text>
      <text x="${pad}" y="${height - 6}" fill="#687385" font-size="13">MLI 0</text>
    </svg>
  `;
}

function renderProbabilities(probabilities) {
  const container = document.querySelector("#probabilities");
  container.innerHTML = "";
  Object.entries(probabilities || {}).forEach(([label, probability]) => {
    const pct = Math.round(Number(probability) * 1000) / 10;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span>${label}</span>
      <div class="bar"><span style="width: ${pct}%"></span></div>
      <strong>${pct}%</strong>
    `;
    container.appendChild(row);
  });
}

function renderResult(result) {
  const mli = result.mental_load_index || {};
  emptyState.hidden = true;
  resultView.hidden = false;
  details.hidden = false;

  setText("#timestamp", new Date(result.timestamp).toLocaleString());
  setText("#condition", result.condition || "--");
  setText("#stress-level", result.stress_level || "--");
  setText("#confidence", `${result.confidence || 0}%`);
  setText("#mli-category", mli.category || "--");
  setText("#mli-score", mli.score ?? "--");

  const gauge = document.querySelector(".gauge");
  gauge.style.setProperty("--score", Number(mli.score || 0));

  renderProbabilities(result.probabilities);
  listItems("#advisory-list", result.advisory && result.advisory.recommendations);
  setText("#warning-message", result.early_warning && result.early_warning.escalation_risk);
  listItems("#natural-list", result.recommendations && result.recommendations.natural_interventions);
  listItems("#otc-list", result.recommendations && result.recommendations.otc_options);
  listItems("#professional-list", result.recommendations && result.recommendations.professional_services);

  saveReading({
    timestamp: result.timestamp,
    mli: Number(mli.score || 0),
    condition: result.condition || "Unknown",
  });
  renderTrend();
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    modelPill.classList.remove("ready", "degraded");
    if (health.status === "ok") {
      modelPill.textContent = "Model files ready";
      modelPill.classList.add("ready");
    } else {
      modelPill.textContent = "Model files missing";
      modelPill.classList.add("degraded");
    }
  } catch (_error) {
    modelPill.textContent = "Health check unavailable";
    modelPill.classList.add("degraded");
  }
}

form.querySelectorAll("input[type='range']").forEach((slider) => {
  const output = slider.parentElement.querySelector("output");
  const sync = () => {
    output.textContent = slider.value;
  };
  slider.addEventListener("input", sync);
  sync();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  submitButton.disabled = true;
  submitButton.textContent = "Analyzing...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getFormData()),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed");
    }
    renderResult(payload);
  } catch (error) {
    setError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Analyze stress level";
  }
});

document.querySelector("#clear-history").addEventListener("click", () => {
  localStorage.removeItem(historyKey);
  renderTrend();
});

checkHealth();
renderTrend();
