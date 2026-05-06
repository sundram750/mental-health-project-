from flask import Flask, request, jsonify, render_template_string
import sys
import os
from pathlib import Path

# Add project root to path so model/utils packages are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.insert(0, project_dir)

# Safely import application modules — any import error is caught so
# the serverless function can still start and return a meaningful error
try:
    from model.predictor import MentalHealthPredictor
    _predictor_available = True
except Exception as _import_err:
    _predictor_available = False
    print(f"✗ Could not import MentalHealthPredictor: {_import_err}")

try:
    from utils.recommendation_engine import RecommendationEngine
    _rec_engine_available = True
except Exception as _import_err2:
    _rec_engine_available = False
    print(f"✗ Could not import RecommendationEngine: {_import_err2}")

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and engine
predictor = None
recommendation_engine = None

# Mapping from frontend snake_case keys → predictor PascalCase keys
_KEY_MAP = {
    'heart_rate':     'Heart_Rate',
    'hrv':            'HRV',
    'respiration':    'Respiration',
    'temperature':    'Skin_Temp',
    'bp_systolic':    'BP_Systolic',
    'bp_diastolic':   'BP_Diastolic',
    'cognitive_state':'Cognitive_State',
    'emotional_state':'Emotional_State',
}

def _map_input(data: dict) -> dict:
    """Remap frontend keys to the keys expected by MentalHealthPredictor."""
    mapped = {}
    for front_key, pred_key in _KEY_MAP.items():
        if front_key in data:
            mapped[pred_key] = data[front_key]
    # Also pass through any already-correct keys
    for k, v in data.items():
        if k not in _KEY_MAP and k not in mapped:
            mapped[k] = v
    return mapped

def initialize_app():
    """Initialize model and engines. Returns True only when the ML model is ready."""
    global predictor, recommendation_engine
    model_ready = False

    # --- load ML model ---
    if _predictor_available:
        try:
            api_dir = os.path.dirname(os.path.abspath(__file__))
            predictor = MentalHealthPredictor(
                model_path=os.path.join(api_dir, 'mental_health_model.pkl'),
                scaler_path=os.path.join(api_dir, 'mental_health_model_scaler.pkl'),
                encoder_path=os.path.join(api_dir, 'mental_health_model_encoder.pkl'),
                features_path=os.path.join(api_dir, 'mental_health_model_features.pkl')
            )
            model_ready = True
            print("✓ MentalHealthPredictor loaded")
        except Exception as e:
            print(f"✗ MentalHealthPredictor init failed: {e}")

    # --- load recommendation engine ---
    if _rec_engine_available:
        try:
            recommendation_engine = RecommendationEngine()
            print("✓ RecommendationEngine loaded")
        except Exception as e:
            print(f"✗ RecommendationEngine init failed: {e}")

    return model_ready

# Initialize on startup
MODEL_READY = initialize_app()

# HTML template (simplified version)
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .form-group { margin: 10px 0; }
        input, button { padding: 8px; margin: 5px; }
        .result { margin-top: 20px; padding: 10px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Mental Health Monitoring System</h1>
        <form id="healthForm">
            <div class="form-group">
                <label>Heart Rate (BPM):</label>
                <input type="number" id="heart_rate" required>
            </div>
            <div class="form-group">
                <label>HRV:</label>
                <input type="number" id="hrv" step="0.1" required>
            </div>
            <div class="form-group">
                <label>Respiration Rate:</label>
                <input type="number" id="respiration" required>
            </div>
            <div class="form-group">
                <label>Skin Temperature (°C):</label>
                <input type="number" id="temperature" step="0.1" required>
            </div>
            <div class="form-group">
                <label>BP Systolic:</label>
                <input type="number" id="bp_systolic" required>
            </div>
            <div class="form-group">
                <label>BP Diastolic:</label>
                <input type="number" id="bp_diastolic" required>
            </div>
            <div class="form-group">
                <label>Cognitive State (1-5):</label>
                <input type="number" id="cognitive" min="1" max="5" required>
            </div>
            <div class="form-group">
                <label>Emotional State (1-5):</label>
                <input type="number" id="emotional" min="1" max="5" required>
            </div>
            <button type="submit">Analyze Stress Level</button>
        </form>
        <div id="result" class="result" style="display:none;"></div>
    </div>

    <script>
        document.getElementById('healthForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const data = {
                heart_rate: parseFloat(document.getElementById('heart_rate').value),
                hrv: parseFloat(document.getElementById('hrv').value),
                respiration: parseFloat(document.getElementById('respiration').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                bp_systolic: parseInt(document.getElementById('bp_systolic').value),
                bp_diastolic: parseInt(document.getElementById('bp_diastolic').value),
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
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').innerHTML = `
                    <h3>Analysis Result:</h3>
                    <p><strong>Stress Level:</strong> ${result.stress_level}</p>
                    <p><strong>Confidence:</strong> ${result.confidence}%</p>
                    <p><strong>Recommendations:</strong> ${result.recommendations.join(', ')}</p>
                `;
            } catch (error) {
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/api/predict', methods=['POST'])
def predict():
    # --- fallback mock when model is unavailable ---
    if not MODEL_READY or predictor is None:
        return jsonify({
            'error': 'Model not ready - using demo prediction',
            'stress_level': 'Moderate-Low',
            'confidence': 75,
            'probabilities': {'Low': 0.1, 'Moderate-Low': 0.4, 'Moderate-High': 0.3, 'High': 0.2},
            'recommendations': ['Take a short break', 'Practice deep breathing', 'Stay hydrated']
        })

    try:
        raw_data = request.get_json(force=True) or {}

        # Remap frontend keys → predictor-expected keys
        mapped_data = _map_input(raw_data)

        # Call the correct method name: predict_stress_level (not predict)
        result = predictor.predict_stress_level(mapped_data)

        # Confidence comes back as a 0-1 float; convert to 0-100 integer for UI
        confidence_pct = round(float(result['confidence']) * 100, 1)

        # Get recommendations (engine may be None if import failed)
        if recommendation_engine is not None:
            recommendations = recommendation_engine.get_recommendations(
                result['stress_level'],
                result['confidence']
            )
        else:
            recommendations = ['Take a short break', 'Practice deep breathing', 'Stay hydrated']

        return jsonify({
            'stress_level': result['stress_level'],
            'confidence': confidence_pct,
            'probabilities': result.get('probabilities', {}),
            'recommendations': recommendations
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if MODEL_READY else 'unhealthy',
        'model_ready': MODEL_READY
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    if not MODEL_READY:
        return jsonify({'error': 'Model not ready'}), 500

    return jsonify({
        'accuracy': '90.28%',
        'model_type': 'Voting Ensemble',
        'features': predictor.get_feature_names() if hasattr(predictor, 'get_feature_names') else 16,
        'training_samples': 1200,
        'cross_val_score': '85.00% (±3.16%)'
    })

# Vercel expects the app to be named 'app'
# This is the entry point for Vercel
app = app