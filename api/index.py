from flask import Flask, request, jsonify, render_template_string
import sys
import os
import json
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.insert(0, project_dir)

# Import your modules
from model.predictor import MentalHealthPredictor
from utils.recommendation_engine import RecommendationEngine

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and engine
predictor = None
recommendation_engine = None

def initialize_app():
    """Initialize model and engines"""
    global predictor, recommendation_engine
    try:
        # Initialize predictor with model files in current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predictor = MentalHealthPredictor(
            model_path=os.path.join(current_dir, 'mental_health_model.pkl'),
            scaler_path=os.path.join(current_dir, 'mental_health_model_scaler.pkl'),
            encoder_path=os.path.join(current_dir, 'mental_health_model_encoder.pkl'),
            features_path=os.path.join(current_dir, 'mental_health_model_features.pkl')
        )
        recommendation_engine = RecommendationEngine()
        print("✓ Models initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing models: {e}")
        # Try to continue without models for now
        try:
            recommendation_engine = RecommendationEngine()
            print("✓ Recommendation engine initialized (models failed)")
            return False  # Models not ready but engine is
        except:
            return False

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
    if not MODEL_READY or predictor is None:
        return jsonify({
            'error': 'Model not ready - using mock prediction',
            'stress_level': 'Moderate-Low',
            'confidence': 75,
            'probabilities': {'Low': 0.1, 'Moderate-Low': 0.4, 'Moderate-High': 0.3, 'High': 0.2},
            'recommendations': ['Take a short break', 'Practice deep breathing', 'Stay hydrated']
        })

    try:
        data = request.get_json()

        # Make prediction
        result = predictor.predict(data)

        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(
            result['stress_level'],
            result['confidence']
        )

        return jsonify({
            'stress_level': result['stress_level'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'recommendations': recommendations
        })

    except Exception as e:
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