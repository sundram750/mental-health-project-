from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request

try:
    from predictor import MentalHealthPredictor
    from recommendations import RecommendationEngine
except ImportError:
    from api.predictor import MentalHealthPredictor
    from api.recommendations import RecommendationEngine


# ── Path resolution ──────────────────────────────────────────────────────────
# Works for both local dev (project root) and Vercel (/var/task/).
_THIS_FILE = Path(__file__).resolve()
_API_DIR   = _THIS_FILE.parent          # …/api/
PROJECT_ROOT = _API_DIR.parent          # …/  (repo root / /var/task)

# Allow override via env var for flexibility
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "model")))

PUBLIC_STATIC = PROJECT_ROOT / "public" / "static"

MODEL_FILES = {
    "model":    MODEL_DIR / "mental_health_model.pkl",
    "scaler":   MODEL_DIR / "mental_health_model_scaler.pkl",
    "encoder":  MODEL_DIR / "mental_health_model_encoder.pkl",
    "features": MODEL_DIR / "mental_health_model_features.pkl",
}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=str(PUBLIC_STATIC),
    static_url_path="/static",
)

# ── Field definitions ─────────────────────────────────────────────────────────
REQUIRED_FIELDS = {
    "heart_rate":     "Heart_Rate",
    "hrv":            "HRV",
    "respiration":    "Respiration",
    "skin_temp":      "Skin_Temp",
    "bp_systolic":    "BP_Systolic",
    "bp_diastolic":   "BP_Diastolic",
    "cognitive_state":"Cognitive_State",
    "emotional_state":"Emotional_State",
}

ALIASES = {
    "temperature":      "skin_temp",
    "skin_temperature": "skin_temp",
    "skinTemp":         "skin_temp",
    "heartRate":        "heart_rate",
    "bpSystolic":       "bp_systolic",
    "bpDiastolic":      "bp_diastolic",
    "cognitiveState":   "cognitive_state",
    "emotionalState":   "emotional_state",
    "Heart_Rate":       "heart_rate",
    "HRV":              "hrv",
    "Respiration":      "respiration",
    "Skin_Temp":        "skin_temp",
    "BP_Systolic":      "bp_systolic",
    "BP_Diastolic":     "bp_diastolic",
    "Cognitive_State":  "cognitive_state",
    "Emotional_State":  "emotional_state",
}

FIELD_LIMITS = {
    "heart_rate":     (35, 220),
    "hrv":            (0, 500),
    "respiration":    (6, 45),
    "skin_temp":      (30, 42),
    "bp_systolic":    (70, 220),
    "bp_diastolic":   (35, 140),
    "cognitive_state":(1, 5),
    "emotional_state":(1, 5),
}


# ── Cached singletons ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_predictor() -> MentalHealthPredictor:
    return MentalHealthPredictor(
        model_path=MODEL_FILES["model"],
        scaler_path=MODEL_FILES["scaler"],
        encoder_path=MODEL_FILES["encoder"],
        features_path=MODEL_FILES["features"],
    )


@lru_cache(maxsize=1)
def get_recommendation_engine() -> RecommendationEngine:
    return RecommendationEngine()


# ── Helpers ───────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_files_status() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None,
        }
        for name, path in MODEL_FILES.items()
    }


def normalize_payload(payload: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        canonical = ALIASES.get(key, key)
        normalized[canonical] = value

    missing = [field for field in REQUIRED_FIELDS if field not in normalized]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    parsed: dict[str, float] = {}
    for field in REQUIRED_FIELDS:
        try:
            value = float(normalized[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a number") from None

        lower, upper = FIELD_LIMITS[field]
        if not lower <= value <= upper:
            raise ValueError(f"{field} must be between {lower} and {upper}")

        parsed[field] = round(value) if field in {"cognitive_state", "emotional_state"} else value

    return parsed


def to_model_features(data: dict[str, float]) -> dict[str, float]:
    return {model_key: data[field] for field, model_key in REQUIRED_FIELDS.items()}


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def stress_from_label(stress_level: str) -> str:
    label = str(stress_level).strip().lower()
    if label in {"low", "calm", "0"}:
        return "Low"
    if label in {"high", "stressed", "3"}:
        return "High"
    return "Moderate"


def condition_from_stress(stress_level: str) -> str:
    grouped = stress_from_label(stress_level)
    if grouped == "Low":
        return "Calm"
    if grouped == "High":
        return "Stressed"
    return "Moderate"


def compute_mli(data: dict[str, float]) -> dict[str, Any]:
    hr       = data["heart_rate"]
    hrv      = data["hrv"]
    resp     = data["respiration"]
    temp     = data["skin_temp"]
    sys_bp   = data["bp_systolic"]
    dia_bp   = data["bp_diastolic"]
    cognitive = data["cognitive_state"]
    emotional = data["emotional_state"]

    components = {
        "heart_rate":        clamp((hr - 60) / 70 * 100),
        "hrv":               clamp((110 - hrv) / 110 * 100),
        "respiration":       clamp((resp - 12) / 18 * 100),
        "skin_temperature":  clamp(abs(temp - 36.5) / 2.8 * 100),
        "blood_pressure":    clamp((((sys_bp - 110) / 55) + ((dia_bp - 72) / 35)) * 50),
        "cognitive_load":    clamp((cognitive - 1) / 4 * 100),
        "emotional_load":    clamp((emotional - 1) / 4 * 100),
    }
    weights = {
        "heart_rate": 0.18, "hrv": 0.17, "respiration": 0.12,
        "skin_temperature": 0.10, "blood_pressure": 0.16,
        "cognitive_load": 0.13, "emotional_load": 0.14,
    }
    score = round(sum(components[name] * weights[name] for name in weights))

    if score <= 30:
        status, category = "Calm", "Calm"
    elif score <= 50:
        status, category = "Monitor", "Moderate-Low"
    elif score <= 70:
        status, category = "Action advised", "Moderate-High"
    else:
        status, category = "Elevated cognitive load", "Stressed"

    return {
        "score": int(score),
        "status": status,
        "category": category,
        "components": {key: round(value, 1) for key, value in components.items()},
    }


def build_advisory(condition: str, mli_score: int) -> dict[str, Any]:
    if condition == "Calm":
        items    = [
            "Maintain the current routine and preserve recovery time.",
            "Continue light physical activity and hydration.",
            "Use this reading as a stable personal baseline.",
        ]
        priority = "Low"
    elif condition == "Moderate":
        items    = [
            "Take a short recovery break before continuing demanding work.",
            "Reduce multitasking for the next hour.",
            "Use slow breathing or a brief grounding exercise.",
        ]
        priority = "Medium"
    else:
        items    = [
            "Pause high-demand tasks and move into a lower-stimulation setting.",
            "Use an immediate relaxation protocol for 10 to 15 minutes.",
            "Consider professional support if this state persists or feels unsafe.",
        ]
        priority = "High"

    return {
        "message": f"{condition} condition detected with Mental Load Index {mli_score}.",
        "recommendations": items,
        "implementation_duration": "Immediate",
        "priority": priority,
    }


def build_warning(condition: str, mli_score: int) -> dict[str, Any]:
    if condition == "Calm":
        risk, message, confidence = (
            "Low",
            "No immediate escalation signal detected.",
            0.72,
        )
    elif condition == "Moderate":
        risk, message, confidence = (
            "Moderate",
            "If this pattern continues, high stress may develop within 48 to 72 hours.",
            0.78,
        )
    else:
        risk, message, confidence = (
            "High",
            "High burnout risk if the current pattern persists. Immediate intervention is recommended.",
            0.86,
        )

    return {
        "burnout_risk_level":    risk,
        "burnout_risk_score":    min(100, max(0, int(mli_score))),
        "escalation_risk":       message,
        "escalation_confidence": confidence,
        "timeframe":             "48-72 hours",
    }


def prediction_response(data: dict[str, float]) -> dict[str, Any]:
    predictor    = get_predictor()
    prediction   = predictor.predict_stress_level(to_model_features(data))
    stress_level = str(prediction["stress_level"])
    condition    = condition_from_stress(stress_level)
    mli          = compute_mli(data)
    advisory     = build_advisory(condition, mli["score"])
    warning      = build_warning(condition, mli["score"])
    rec_level    = stress_from_label(stress_level)
    recommendations = get_recommendation_engine().generate(rec_level, mli["score"])

    confidence    = round(float(prediction["confidence"]) * 100, 1)
    probabilities = {
        str(label): round(float(prob), 4)
        for label, prob in prediction["probabilities"].items()
    }

    return {
        "timestamp":        now_iso(),
        "input_summary":    data,
        "stress_level":     stress_level,
        "condition":        condition,
        "confidence":       confidence,
        "probabilities":    probabilities,
        "ml_prediction": {
            "stress_level": stress_level,
            "condition":    condition,
            "confidence":   confidence,
            "probabilities":probabilities,
        },
        "mental_load_index": mli,
        "advisory":          advisory,
        "early_warning":     warning,
        "recommendations":   recommendations,
    }


# ── Templates (inline so no template_folder needed on Vercel) ────────────────
INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NeuroPulse Mental Health Monitor</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="/">NeuroPulse</a>
    <nav aria-label="Primary">
      <a href="/" aria-current="page">Dashboard</a>
      <a href="/about">Model</a>
    </nav>
  </header>
  <main class="shell">
    <section class="hero">
      <div>
        <p class="eyebrow">IoT-enabled wearable sensor research</p>
        <h1>Mental health monitoring from biometric and psychological signals</h1>
        <p class="lede">Enter a current reading to classify stress level, estimate mental load,
           and generate tiered recommendations using the saved voting ensemble model.</p>
      </div>
      <img src="/static/brain.png" alt="Neural brain visualization" class="hero-image">
    </section>

    <section class="workspace">
      <form id="prediction-form" class="panel" autocomplete="off">
        <div class="panel-head">
          <h2>Current Reading</h2>
          <span id="model-pill" class="pill">Model ready check pending</span>
        </div>
        <div class="field-grid">
          <label>Heart rate <input name="heart_rate" type="number" min="35" max="220" value="78" required><span>BPM</span></label>
          <label>HRV <input name="hrv" type="number" min="0" max="500" value="64" step="0.1" required><span>ms</span></label>
          <label>Respiration <input name="respiration" type="number" min="6" max="45" value="16" step="0.1" required><span>breaths/min</span></label>
          <label>Skin temp <input name="skin_temp" type="number" min="30" max="42" value="36.4" step="0.1" required><span>C</span></label>
          <label>BP systolic <input name="bp_systolic" type="number" min="70" max="220" value="119" required><span>mmHg</span></label>
          <label>BP diastolic <input name="bp_diastolic" type="number" min="35" max="140" value="78" required><span>mmHg</span></label>
        </div>
        <div class="slider-grid">
          <label>Cognitive fatigue <input name="cognitive_state" type="range" min="1" max="5" value="3"><output>3</output></label>
          <label>Emotional strain <input name="emotional_state" type="range" min="1" max="5" value="3"><output>3</output></label>
        </div>
        <button class="primary-button" type="submit">Analyze stress level</button>
        <p id="form-error" class="form-error" role="alert"></p>
      </form>

      <section class="panel results-panel" aria-live="polite">
        <div class="panel-head">
          <h2>Assessment</h2>
          <span id="timestamp" class="muted">No reading yet</span>
        </div>
        <div id="empty-state" class="empty-state">Submit a reading to see condition, mental load, probabilities, warning status, and recommendations.</div>
        <div id="result-view" class="result-view" hidden>
          <div class="result-hero">
            <div>
              <span class="muted">Condition</span>
              <strong id="condition">--</strong>
            </div>
            <div class="gauge" style="--score:0">
              <span id="mli-score">0</span>
              <small>MLI</small>
            </div>
          </div>
          <div class="summary-grid">
            <div><span>Stress level</span><strong id="stress-level">--</strong></div>
            <div><span>Confidence</span><strong id="confidence">--</strong></div>
            <div><span>Category</span><strong id="mli-category">--</strong></div>
          </div>
          <div class="probability-bars" id="probabilities"></div>
        </div>
      </section>
    </section>

    <section id="details" class="details" hidden>
      <article class="panel">
        <h2>Advisory</h2>
        <ul id="advisory-list" class="clean-list"></ul>
      </article>
      <article class="panel">
        <h2>Early Warning</h2>
        <p id="warning-message" class="warning-text"></p>
      </article>
      <article class="panel wide">
        <div class="panel-head">
          <h2>Local Trend</h2>
          <button id="clear-history" class="ghost-button" type="button">Clear</button>
        </div>
        <div id="trend-chart" class="trend-chart"></div>
      </article>
      <article class="panel wide">
        <h2>Personalized Recommendations</h2>
        <div class="recommendation-grid">
          <section><h3>Natural Interventions</h3><ul id="natural-list" class="clean-list"></ul></section>
          <section><h3>OTC Options</h3><ul id="otc-list" class="clean-list"></ul></section>
          <section><h3>Professional Support</h3><ul id="professional-list" class="clean-list"></ul></section>
        </div>
        <p class="disclaimer">Research prototype only. This system does not replace professional medical diagnosis or emergency care.</p>
      </article>
    </section>
  </main>
  <script src="/static/app.js" defer></script>
</body>
</html>
"""

ABOUT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Details - NeuroPulse</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="/">NeuroPulse</a>
    <nav aria-label="Primary">
      <a href="/">Dashboard</a>
      <a href="/about" aria-current="page">Model</a>
    </nav>
  </header>
  <main class="shell">
    <section class="hero compact">
      <div>
        <p class="eyebrow">Saved model retained</p>
        <h1>Voting ensemble stress classifier</h1>
        <p class="lede">The deployment keeps the original joblib model artifacts and uses a lighter Vercel-compatible inference wrapper.</p>
      </div>
      <img src="/static/brain.png" alt="Neural brain visualization" class="hero-image">
    </section>
    <section class="metrics">
      <article class="panel"><span>Accuracy</span><strong>90.28%</strong></article>
      <article class="panel"><span>Model type</span><strong>Voting Ensemble</strong></article>
      <article class="panel"><span>Features</span><strong>16</strong></article>
      <article class="panel"><span>Training samples</span><strong>1,200</strong></article>
    </section>
    <section class="details">
      <article class="panel wide">
        <h2>Input Features</h2>
        <div class="feature-columns">
          <ul class="clean-list">
            <li>Heart rate</li><li>Heart rate variability</li>
            <li>Respiration rate</li><li>Skin temperature</li>
            <li>Blood pressure systolic</li><li>Blood pressure diastolic</li>
            <li>Cognitive state</li><li>Emotional state</li>
          </ul>
          <ul class="clean-list">
            <li>Heart rate to HRV ratio</li><li>Average blood pressure</li>
            <li>Pulse pressure</li><li>Psychological score</li>
            <li>Heart rate to respiration ratio</li><li>Temperature deviation</li>
            <li>Normalized HRV</li><li>Heart variability index</li>
          </ul>
        </div>
      </article>
      <article class="panel image-panel">
        <h2>Confusion Matrix</h2>
        <img src="/static/confusion_matrix.png" alt="Confusion matrix">
      </article>
      <article class="panel image-panel">
        <h2>ROC AUC Curve</h2>
        <img src="/static/roc_auc_curve.png" alt="ROC AUC curve">
      </article>
    </section>
  </main>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def index() -> str:
    return render_template_string(INDEX_TEMPLATE)


@app.get("/about")
def about() -> str:
    return render_template_string(ABOUT_TEMPLATE)


@app.get("/api/health")
def health():
    files = model_files_status()
    all_present = all(item["exists"] for item in files.values())
    return jsonify({
        "status":       "ok" if all_present else "degraded",
        "model_files":  files,
        "model_loaded": get_predictor.cache_info().currsize > 0,
        "timestamp":    now_iso(),
    })


@app.get("/api/model-info")
def model_info():
    try:
        predictor = get_predictor()
        return jsonify({
            "status":       "ready",
            "model_type":   "Voting Ensemble",
            "accuracy":     "90.28%",
            "features":     list(predictor.feature_names),
            "stress_levels":[str(label) for label in predictor.classes],
            "model_files":  model_files_status(),
        })
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc), "model_files": model_files_status()}), 503


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        data = normalize_payload(payload)
        return jsonify(prediction_response(data))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc), "model_files": model_files_status()}), 503
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "Internal server error"}), 500


# Vercel expects the WSGI app to be named `app`
# Local dev entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
