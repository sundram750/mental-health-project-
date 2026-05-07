"""
Mental Health Monitoring System - Local Development Server
Uses the saved Voting Ensemble model (90.28% accuracy).
Run this file directly for local development.
For Vercel deployment use api/index.py instead.
"""

from __future__ import annotations

import errno
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR     = Path(__file__).resolve().parent   # …/web/  (or wherever this file lives)
PROJECT_DIR = APP_DIR.parent                    # project root
MODEL_DIR   = PROJECT_DIR / "model"

# Allow the project root on the import path so that
# "from model.predictor import …" resolves correctly.
sys.path.insert(0, str(PROJECT_DIR))

from model.predictor import MentalHealthPredictor  # noqa: E402  (after sys.path update)

# RecommendationEngine lives at utils/recommendation_engine.py locally,
# but the standalone api/recommendations.py is also acceptable.
try:
    from utils.recommendation_engine import RecommendationEngine
except ImportError:
    # Fallback: use the self-contained module copied into api/
    sys.path.insert(0, str(PROJECT_DIR / "api"))
    from recommendations import RecommendationEngine  # type: ignore[no-redef]

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)

# ── Load model ────────────────────────────────────────────────────────────────
print("\n[Loading Model Components…]")

MODEL_FILES = {
    "model":    MODEL_DIR / "mental_health_model.pkl",
    "scaler":   MODEL_DIR / "mental_health_model_scaler.pkl",
    "encoder":  MODEL_DIR / "mental_health_model_encoder.pkl",
    "features": MODEL_DIR / "mental_health_model_features.pkl",
}

predictor   = None
MODEL_READY = False

try:
    predictor = MentalHealthPredictor(
        model_path   = MODEL_FILES["model"],
        scaler_path  = MODEL_FILES["scaler"],
        encoder_path = MODEL_FILES["encoder"],
        features_path= MODEL_FILES["features"],
    )
    print(f"✓ Model loaded  — Voting Ensemble, 90.28% accuracy")
    print(f"✓ Features      — {len(predictor.feature_names)} total")
    MODEL_READY = True
except FileNotFoundError as exc:
    print(f"✗ Model files missing: {exc}")
    print("  Put the four .pkl files in  project_root/model/  and restart.")
except Exception as exc:
    print(f"✗ Model loading failed: {exc}")

# ── Recommendation engine ─────────────────────────────────────────────────────
recommendation_engine = RecommendationEngine()
print("✓ Recommendation Engine initialized")

# ── In-memory session storage ─────────────────────────────────────────────────
user_sessions: dict[str, list] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _model_key_map() -> dict[str, str]:
    """Map snake_case field names → PascalCase model feature names."""
    return {
        "heart_rate":     "Heart_Rate",
        "hrv":            "HRV",
        "respiration":    "Respiration",
        "skin_temp":      "Skin_Temp",
        "bp_systolic":    "BP_Systolic",
        "bp_diastolic":   "BP_Diastolic",
        "cognitive_state":"Cognitive_State",
        "emotional_state":"Emotional_State",
    }


def _normalize_input(raw: dict) -> dict[str, float]:
    """
    Accept both snake_case and PascalCase field names.
    Returns a dict with snake_case keys and float values.
    """
    aliases = {v: k for k, v in _model_key_map().items()}  # PascalCase → snake_case
    normalized: dict[str, float] = {}
    for key, value in raw.items():
        canonical = aliases.get(key, key)  # convert PascalCase → snake_case if needed
        normalized[canonical] = float(value)
    return normalized


def _to_model_features(data: dict[str, float]) -> dict[str, float]:
    return {pascal: data[snake] for snake, pascal in _model_key_map().items()}


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _compute_mli(data: dict[str, float]) -> int:
    """Compute Mental Load Index (0-100) from snake_case data dict."""
    components = {
        "heart_rate":       _clamp((data["heart_rate"] - 60) / 70 * 100),
        "hrv":              _clamp((110 - data["hrv"]) / 110 * 100),
        "respiration":      _clamp((data["respiration"] - 12) / 18 * 100),
        "skin_temperature": _clamp(abs(data["skin_temp"] - 36.5) / 2.8 * 100),
        "blood_pressure":   _clamp((((data["bp_systolic"] - 110) / 55)
                                    + ((data["bp_diastolic"] - 72) / 35)) * 50),
        "cognitive_load":   _clamp((data["cognitive_state"] - 1) / 4 * 100),
        "emotional_load":   _clamp((data["emotional_state"] - 1) / 4 * 100),
    }
    weights = {
        "heart_rate": 0.18, "hrv": 0.17, "respiration": 0.12,
        "skin_temperature": 0.10, "blood_pressure": 0.16,
        "cognitive_load": 0.13, "emotional_load": 0.14,
    }
    return int(round(sum(components[k] * weights[k] for k in weights)))


def _stress_to_rec_level(stress_level: str) -> str:
    """Map model stress label → recommendation tier (Low / Moderate / High)."""
    label = stress_level.strip().lower()
    if label in {"low", "calm"}:
        return "Low"
    if label in {"high", "stressed"}:
        return "High"
    return "Moderate"


def _get_quick_recommendation(stress_label: str) -> dict:
    """Simple inline quick-tip dict shown in the UI alongside full recommendations."""
    tips = {
        "Low": {
            "title": "✓ Maintain Current State",
            "primary": "Keep up your current routine — you're managing stress well!",
            "actions": [
                "✓ Continue regular exercise",
                "✓ Maintain current sleep schedule",
                "✓ Keep healthy eating habits",
            ],
        },
        "Moderate-Low": {
            "title": "⚠ Light Stress Management",
            "primary": "Minor stress detected. Take preventive steps.",
            "actions": [
                "→ Take 5–10 minute breaks",
                "→ Practice deep breathing (4-7-8 technique)",
                "→ Go for a short walk",
            ],
        },
        "Moderate-High": {
            "title": "⚠ Moderate Stress Response",
            "primary": "Noticeable stress. Implement stress management.",
            "actions": [
                "→ Try meditation (10–15 minutes)",
                "→ Do light exercise or yoga",
                "→ Connect with friends/family",
            ],
        },
        "High": {
            "title": "🚨 High Stress Alert",
            "primary": "High stress detected. Seek help if persistent.",
            "actions": [
                "→ Consider talking to a counsellor",
                "→ Practice intensive relaxation techniques",
                "→ Possible medical consultation recommended",
            ],
        },
    }
    return tips.get(stress_label, tips["Moderate-High"])


# ── Core prediction function ──────────────────────────────────────────────────

def predict_stress(raw_input: dict) -> dict:
    """
    Run full stress prediction pipeline using the saved ML model.
    Accepts both snake_case and PascalCase field names.
    Returns a structured result dict.
    """
    if not MODEL_READY or predictor is None:
        return {"error": "Model not ready. Check that model .pkl files exist in model/"}

    try:
        data        = _normalize_input(raw_input)
        model_feats = _to_model_features(data)

        # ── 1. ML model prediction ──────────────────────────────────────────
        ml_result    = predictor.predict_stress_level(model_feats)
        stress_level = str(ml_result["stress_level"])      # e.g. "Low", "High", etc.
        confidence   = round(float(ml_result["confidence"]) * 100, 1)
        probabilities = {
            str(k): round(float(v), 4)
            for k, v in ml_result["probabilities"].items()
        }

        # ── 2. Mental Load Index ────────────────────────────────────────────
        mli = _compute_mli(data)

        # ── 3. UI display fields ────────────────────────────────────────────
        label_lower = stress_level.strip().lower()
        if label_lower in {"low", "calm"}:
            emoji = "😌"; color = "#10b981"; category = "LOW"
        elif label_lower == "high":
            emoji = "😰"; color = "#ef4444"; category = "HIGH"
        elif "high" in label_lower:
            emoji = "😟"; color = "#f97316"; category = "MODERATE-HIGH"
        else:
            emoji = "😐"; color = "#f59e0b"; category = "MODERATE-LOW"

        # ── 4. Recommendations ──────────────────────────────────────────────
        rec_level = _stress_to_rec_level(stress_level)
        # FIX: correct method name is .generate(), not .generate_recommendations()
        full_recs = recommendation_engine.generate(rec_level, mli)

        return {
            "stress_level":    stress_level,
            "stress_category": category,
            "emoji":           emoji,
            "color":           color,
            "confidence":      confidence,
            "probabilities":   probabilities,
            "mental_load_index": mli,
            "recommendation":  _get_quick_recommendation(stress_level),
            "recommendations": {
                "natural_interventions": full_recs.get("natural_interventions", []),
                "otc_options":           full_recs.get("otc_options", []),
                "professional_services": full_recs.get("professional_services", []),
            },
        }

    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        model_accuracy="90.28%",
        model_type="Voting Ensemble",
        features_used=len(predictor.feature_names) if predictor else 16,
    )


@app.route("/about")
def about():
    model_metrics = {
        "accuracy": "90.28%",
        "precision": "89.5%",
        "recall": "90.1%",
        "f1_score": "89.8%",
        "cv_accuracy": "85.00% ± 3.16%",
    }
    algorithms = [
        {"name": "Extra Trees",         "accuracy": "89.2%", "precision": "88.5%", "recall": "89.0%", "f1": "88.7%"},
        {"name": "Logistic Regression", "accuracy": "82.1%", "precision": "81.3%", "recall": "82.0%", "f1": "81.6%"},
        {"name": "Gradient Boosting",   "accuracy": "88.5%", "precision": "87.9%", "recall": "88.3%", "f1": "88.1%"},
        {"name": "K-Nearest Neighbors", "accuracy": "84.7%", "precision": "83.8%", "recall": "84.5%", "f1": "84.1%"},
        {"name": "Random Forest",       "accuracy": "87.3%", "precision": "86.7%", "recall": "87.1%", "f1": "86.9%"},
    ]
    per_class = {
        "Low":           {"accuracy": "87%", "samples": 280},
        "Moderate-Low":  {"accuracy": "88%", "samples": 300},
        "Moderate-High": {"accuracy": "91%", "samples": 320},
        "High":          {"accuracy": "96%", "samples": 300},
    }
    return render_template(
        "about.html",
        model_metrics=model_metrics,
        algorithms=algorithms,
        per_class=per_class,
        model_accuracy="90.28%",
        model_type="Voting Ensemble",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400

        required = list(_model_key_map().keys()) + list(_model_key_map().values())
        has_snake  = all(f in data for f in _model_key_map().keys())
        has_pascal = all(f in data for f in _model_key_map().values())

        if not has_snake and not has_pascal:
            missing = [f for f in _model_key_map().keys() if f not in data]
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        result = predict_stress(data)

        if "error" in result:
            return jsonify(result), 500

        session_id = data.get("session_id")
        if session_id:
            user_sessions.setdefault(session_id, []).append({
                "timestamp": datetime.now().isoformat(),
                "data": result,
            })

        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":      "ok" if MODEL_READY else "degraded",
        "model_ready": MODEL_READY,
        "timestamp":   datetime.now().isoformat(),
        "accuracy":    "90.28%",
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": "Voting Ensemble (5 algorithms)",
        "accuracy": {
            "test_set":          "90.28%",
            "cross_validation":  "85.00% ± 3.16%",
        },
        "algorithms": [
            "Extra Trees", "Logistic Regression", "Gradient Boosting",
            "K-Nearest Neighbors", "Random Forest",
        ],
        "features":         len(predictor.feature_names) if predictor else 16,
        "training_samples": 1200,
        "stress_levels":    [str(c) for c in predictor.classes] if predictor else [],
        "per_class_accuracy": {
            "Low":           "87%",
            "Moderate-Low":  "88%",
            "Moderate-High": "91%",
            "High":          "96%",
        },
    })


@app.route("/api/recommendations", methods=["POST"])
def get_recommendations():
    try:
        data         = request.get_json(force=True, silent=True) or {}
        stress_level = str(data.get("stress_level", "Moderate"))
        mli_score    = data.get("mli_score")
        rec_level    = _stress_to_rec_level(stress_level)
        # FIX: correct method name
        result       = recommendation_engine.generate(rec_level, mli_score)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "Server error"}), 500


# ── Port helpers ──────────────────────────────────────────────────────────────

def find_free_port(host: str, start: int, stop: int = None) -> int:
    stop = stop or start + 20
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError as exc:
                if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                    continue
                raise
    raise OSError(f"No free port found in {start}–{stop}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MENTAL HEALTH MONITORING SYSTEM — LOCAL DEV SERVER")
    print("=" * 70)
    if MODEL_READY:
        print(f"  ✓ Model:        Voting Ensemble  (90.28% accuracy)")
        print(f"  ✓ Features:     {len(predictor.feature_names)}")
        print(f"  ✓ Classes:      {predictor.classes}")
    else:
        print("  ⚠  Model not ready — predictions will return an error.")
    print("=" * 70 + "\n")

    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", os.environ.get("PORT", 5000)))
    try:
        port = find_free_port(host, port)
    except OSError as exc:
        print(f"✗ Could not acquire a free port: {exc}")
        sys.exit(1)

    print(f"  Access at: http://{host}:{port}\n")
    app.run(debug=True, host=host, port=port, use_reloader=False)
