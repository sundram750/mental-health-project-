from flask import Flask, request, jsonify, render_template_string
import os
import sys

# ---------------------------------------------------------------------------
# All dependencies are co-located in the same api/ directory.
# No sys.path tricks needed — Vercel bundles api/ completely.
# ---------------------------------------------------------------------------

# Safely import predictor (local file: api/predictor.py)
try:
    from predictor import MentalHealthPredictor
    _predictor_available = True
except Exception as _e:
    _predictor_available = False
    print(f"[WARN] predictor import failed: {_e}")

# Safely import recommendation engine (local file: api/recommendation_engine.py)
try:
    from recommendation_engine import RecommendationEngine
    _rec_engine_available = True
except Exception as _e2:
    _rec_engine_available = False
    print(f"[WARN] recommendation_engine import failed: {_e2}")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Global model instances
predictor = None
recommendation_engine = None

# Key mapping: frontend snake_case -> predictor PascalCase
_KEY_MAP = {
    'heart_rate':      'Heart_Rate',
    'hrv':             'HRV',
    'respiration':     'Respiration',
    'temperature':     'Skin_Temp',
    'bp_systolic':     'BP_Systolic',
    'bp_diastolic':    'BP_Diastolic',
    'cognitive_state': 'Cognitive_State',
    'emotional_state': 'Emotional_State',
}

def _map_input(data: dict) -> dict:
    """Remap frontend keys to keys expected by MentalHealthPredictor."""
    mapped = {}
    for front_key, pred_key in _KEY_MAP.items():
        if front_key in data:
            mapped[pred_key] = data[front_key]
    # Pass through already-correct keys
    for k, v in data.items():
        if k not in _KEY_MAP and k not in mapped:
            mapped[k] = v
    return mapped


def initialize_app():
    """Load model + engine. Returns True only when ML model is ready."""
    global predictor, recommendation_engine
    model_ready = False

    if _predictor_available:
        try:
            # .pkl files live in api/ alongside this file
            api_dir = os.path.dirname(os.path.abspath(__file__))
            predictor = MentalHealthPredictor(
                model_path=os.path.join(api_dir, 'mental_health_model.pkl'),
                scaler_path=os.path.join(api_dir, 'mental_health_model_scaler.pkl'),
                encoder_path=os.path.join(api_dir, 'mental_health_model_encoder.pkl'),
                features_path=os.path.join(api_dir, 'mental_health_model_features.pkl'),
            )
            model_ready = True
            print("[OK] MentalHealthPredictor loaded")
        except Exception as e:
            print(f"[ERROR] MentalHealthPredictor init failed: {e}")

    if _rec_engine_available:
        try:
            recommendation_engine = RecommendationEngine()
            print("[OK] RecommendationEngine loaded")
        except Exception as e:
            print(f"[ERROR] RecommendationEngine init failed: {e}")

    return model_ready


MODEL_READY = initialize_app()

# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; }
        .form-group { margin: 12px 0; display: flex; align-items: center; gap: 10px; }
        label { min-width: 180px; font-weight: bold; color: #555; }
        input { padding: 8px 12px; border: 1px solid #ddd; border-radius: 5px; width: 160px; }
        button { padding: 12px 30px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        button:hover { background: #2980b9; }
        .result { margin-top: 25px; padding: 20px; border-radius: 8px; border-left: 5px solid #3498db; background: #f0f8ff; }
        .result h3 { margin-top: 0; color: #2c3e50; }
        .error { border-left-color: #e74c3c; background: #fff5f5; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; color: white; }
        .Low { background: #27ae60; }
        .Moderate-Low { background: #f39c12; }
        .Moderate-High { background: #e67e22; }
        .High { background: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>&#129504; Mental Health Monitoring System</h1>
        <p>Enter your biometric readings to assess your current stress level.</p>
        <form id="healthForm">
            <div class="form-group">
                <label>Heart Rate (BPM):</label>
                <input type="number" id="heart_rate" value="75" required>
            </div>
            <div class="form-group">
                <label>HRV (ms):</label>
                <input type="number" id="hrv" step="0.1" value="45" required>
            </div>
            <div class="form-group">
                <label>Respiration Rate:</label>
                <input type="number" id="respiration" value="18" required>
            </div>
            <div class="form-group">
                <label>Skin Temperature (°C):</label>
                <input type="number" id="temperature" step="0.1" value="36.5" required>
            </div>
            <div class="form-group">
                <label>BP Systolic (mmHg):</label>
                <input type="number" id="bp_systolic" value="120" required>
            </div>
            <div class="form-group">
                <label>BP Diastolic (mmHg):</label>
                <input type="number" id="bp_diastolic" value="80" required>
            </div>
            <div class="form-group">
                <label>Cognitive State (1-5):</label>
                <input type="number" id="cognitive" min="1" max="5" value="3" required>
            </div>
            <div class="form-group">
                <label>Emotional State (1-5):</label>
                <input type="number" id="emotional" min="1" max="5" value="3" required>
            </div>
            <button type="submit">Analyze Stress Level</button>
        </form>
        <div id="result" style="display:none;"></div>
    </div>

    <script>
        document.getElementById('healthForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = 'result';
            resultDiv.innerHTML = '<p>Analyzing...</p>';

            const data = {
                heart_rate:      parseFloat(document.getElementById('heart_rate').value),
                hrv:             parseFloat(document.getElementById('hrv').value),
                respiration:     parseFloat(document.getElementById('respiration').value),
                temperature:     parseFloat(document.getElementById('temperature').value),
                bp_systolic:     parseInt(document.getElementById('bp_systolic').value),
                bp_diastolic:    parseInt(document.getElementById('bp_diastolic').value),
                cognitive_state: parseInt(document.getElementById('cognitive').value),
                emotional_state: parseInt(document.getElementById('emotional').value)
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                const level = result.stress_level || 'Unknown';
                const levelClass = level.replace(' ', '-');
                const recs = Array.isArray(result.recommendations)
                    ? result.recommendations.join(' &bull; ')
                    : result.recommendations;
                resultDiv.innerHTML = `
                    <h3>Analysis Result</h3>
                    <p><strong>Stress Level:</strong> <span class="badge ${levelClass}">${level}</span></p>
                    <p><strong>Confidence:</strong> ${result.confidence}%</p>
                    <p><strong>Recommendations:</strong> ${recs}</p>
                    ${result.error ? '<p style="color:#888;font-size:0.9em;">&#9432; ' + result.error + '</p>' : ''}
                `;
            } catch (error) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `<p><strong>Error:</strong> ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/api/predict', methods=['POST'])
def predict():
    # Fallback demo response when model failed to load
    if not MODEL_READY or predictor is None:
        return jsonify({
            'stress_level': 'Moderate-Low',
            'confidence': 75.0,
            'probabilities': {'Low': 0.1, 'Moderate-Low': 0.4, 'Moderate-High': 0.3, 'High': 0.2},
            'recommendations': ['Take a short break', 'Practice deep breathing', 'Stay hydrated'],
            'error': 'Model not ready — showing demo prediction'
        })

    try:
        raw_data = request.get_json(force=True) or {}
        mapped_data = _map_input(raw_data)

        # Call the correct method
        result = predictor.predict_stress_level(mapped_data)

        # Confidence: 0-1 float -> 0-100 percentage
        confidence_pct = round(float(result['confidence']) * 100, 1)

        # Get recommendations
        if recommendation_engine is not None:
            recs = recommendation_engine.get_recommendations(
                result['stress_level'], result['confidence']
            )
        else:
            recs = ['Take a short break', 'Practice deep breathing', 'Stay hydrated']

        return jsonify({
            'stress_level': result['stress_level'],
            'confidence': confidence_pct,
            'probabilities': result.get('probabilities', {}),
            'recommendations': recs
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if MODEL_READY else 'degraded',
        'model_ready': MODEL_READY,
        'predictor_available': _predictor_available,
        'rec_engine_available': _rec_engine_available
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_ready': MODEL_READY,
        'accuracy': '90.28%',
        'model_type': 'Voting Ensemble',
        'features': 16,
        'training_samples': 1200,
        'cross_val_score': '85.00% (±3.16%)'
    })