from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

try:
    from predictor import MentalHealthPredictor
    from recommendations import RecommendationEngine
except ImportError:  # Local package execution, for tests and editors.
    from api.predictor import MentalHealthPredictor
    from api.recommendations import RecommendationEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
PUBLIC_STATIC = PROJECT_ROOT / "public" / "static"

MODEL_FILES = {
    "model": MODEL_DIR / "mental_health_model.pkl",
    "scaler": MODEL_DIR / "mental_health_model_scaler.pkl",
    "encoder": MODEL_DIR / "mental_health_model_encoder.pkl",
    "features": MODEL_DIR / "mental_health_model_features.pkl",
}

app = Flask(
    __name__,
    static_folder=str(PUBLIC_STATIC),
    static_url_path="/static",
)


REQUIRED_FIELDS = {
    "heart_rate": "Heart_Rate",
    "hrv": "HRV",
    "respiration": "Respiration",
    "skin_temp": "Skin_Temp",
    "bp_systolic": "BP_Systolic",
    "bp_diastolic": "BP_Diastolic",
    "cognitive_state": "Cognitive_State",
    "emotional_state": "Emotional_State",
}

ALIASES = {
    "temperature": "skin_temp",
    "skin_temperature": "skin_temp",
    "skinTemp": "skin_temp",
    "heartRate": "heart_rate",
    "bpSystolic": "bp_systolic",
    "bpDiastolic": "bp_diastolic",
    "cognitiveState": "cognitive_state",
    "emotionalState": "emotional_state",
    "Heart_Rate": "heart_rate",
    "HRV": "hrv",
    "Respiration": "respiration",
    "Skin_Temp": "skin_temp",
    "BP_Systolic": "bp_systolic",
    "BP_Diastolic": "bp_diastolic",
    "Cognitive_State": "cognitive_state",
    "Emotional_State": "emotional_state",
}

FIELD_LIMITS = {
    "heart_rate": (35, 220),
    "hrv": (0, 500),
    "respiration": (6, 45),
    "skin_temp": (30, 42),
    "bp_systolic": (70, 220),
    "bp_diastolic": (35, 140),
    "cognitive_state": (1, 5),
    "emotional_state": (1, 5),
}


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_files_status() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path.relative_to(PROJECT_ROOT)),
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
    hr = data["heart_rate"]
    hrv = data["hrv"]
    resp = data["respiration"]
    temp = data["skin_temp"]
    sys_bp = data["bp_systolic"]
    dia_bp = data["bp_diastolic"]
    cognitive = data["cognitive_state"]
    emotional = data["emotional_state"]

    components = {
        "heart_rate": clamp((hr - 60) / 70 * 100),
        "hrv": clamp((110 - hrv) / 110 * 100),
        "respiration": clamp((resp - 12) / 18 * 100),
        "skin_temperature": clamp(abs(temp - 36.5) / 2.8 * 100),
        "blood_pressure": clamp((((sys_bp - 110) / 55) + ((dia_bp - 72) / 35)) * 50),
        "cognitive_load": clamp((cognitive - 1) / 4 * 100),
        "emotional_load": clamp((emotional - 1) / 4 * 100),
    }
    weights = {
        "heart_rate": 0.18,
        "hrv": 0.17,
        "respiration": 0.12,
        "skin_temperature": 0.10,
        "blood_pressure": 0.16,
        "cognitive_load": 0.13,
        "emotional_load": 0.14,
    }
    score = round(sum(components[name] * weights[name] for name in weights))

    if score <= 30:
        status = "Calm"
        category = "Calm"
    elif score <= 50:
        status = "Monitor"
        category = "Moderate-Low"
    elif score <= 70:
        status = "Action advised"
        category = "Moderate-High"
    else:
        status = "Elevated cognitive load"
        category = "Stressed"

    return {
        "score": int(score),
        "status": status,
        "category": category,
        "components": {key: round(value, 1) for key, value in components.items()},
    }


def build_advisory(condition: str, mli_score: int) -> dict[str, Any]:
    if condition == "Calm":
        items = [
            "Maintain the current routine and preserve recovery time.",
            "Continue light physical activity and hydration.",
            "Use this reading as a stable personal baseline.",
        ]
        priority = "Low"
    elif condition == "Moderate":
        items = [
            "Take a short recovery break before continuing demanding work.",
            "Reduce multitasking for the next hour.",
            "Use slow breathing or a brief grounding exercise.",
        ]
        priority = "Medium"
    else:
        items = [
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
        risk = "Low"
        message = "No immediate escalation signal detected."
        confidence = 0.72
    elif condition == "Moderate":
        risk = "Moderate"
        message = "If this pattern continues, high stress may develop within 48 to 72 hours."
        confidence = 0.78
    else:
        risk = "High"
        message = "High burnout risk if the current pattern persists. Immediate intervention is recommended."
        confidence = 0.86

    return {
        "burnout_risk_level": risk,
        "burnout_risk_score": min(100, max(0, int(mli_score))),
        "escalation_risk": message,
        "escalation_confidence": confidence,
        "timeframe": "48-72 hours",
    }


def prediction_response(data: dict[str, float]) -> dict[str, Any]:
    predictor = get_predictor()
    prediction = predictor.predict_stress_level(to_model_features(data))
    stress_level = str(prediction["stress_level"])
    condition = condition_from_stress(stress_level)
    mli = compute_mli(data)
    advisory = build_advisory(condition, mli["score"])
    warning = build_warning(condition, mli["score"])
    rec_level = stress_from_label(stress_level)
    recommendations = get_recommendation_engine().generate(rec_level, mli["score"])

    confidence = round(float(prediction["confidence"]) * 100, 1)
    probabilities = {
        str(label): round(float(probability), 4)
        for label, probability in prediction["probabilities"].items()
    }

    return {
        "timestamp": now_iso(),
        "input_summary": data,
        "stress_level": stress_level,
        "condition": condition,
        "confidence": confidence,
        "probabilities": probabilities,
        "ml_prediction": {
            "stress_level": stress_level,
            "condition": condition,
            "confidence": confidence,
            "probabilities": probabilities,
        },
        "mental_load_index": mli,
        "advisory": advisory,
        "early_warning": warning,
        "recommendations": recommendations,
    }


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/about")
def about() -> str:
    return render_template("about.html")


@app.get("/api/health")
def health():
    files = model_files_status()
    return jsonify(
        {
            "status": "ok" if all(item["exists"] for item in files.values()) else "degraded",
            "model_files": files,
            "model_loaded": get_predictor.cache_info().currsize > 0,
            "timestamp": now_iso(),
        }
    )


@app.get("/api/model-info")
def model_info():
    predictor = get_predictor()
    return jsonify(
        {
            "status": "ready",
            "model_type": "Voting Ensemble",
            "accuracy": "90.28%",
            "features": list(predictor.feature_names),
            "stress_levels": [str(label) for label in predictor.classes],
            "model_files": model_files_status(),
        }
    )


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


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
