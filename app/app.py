"""
Mental Health Monitoring System - Flask Web Application
Integrates all modules for interactive stress monitoring and recommendations
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.predictor import MentalHealthPredictor
from utils.mental_load_calculator import MentalLoadCalculator
from utils.advisory_engine import AdvisoryEngine
from utils.early_warning import EarlyWarningSystem
from utils.recommendation_engine import RecommendationEngine

# Initialize Flask app
app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')

# Initialize all system components
try:
    predictor = MentalHealthPredictor()
    print("✓ Predictor loaded successfully")
except Exception as e:
    print(f"✗ Error loading predictor: {e}")
    predictor = None

mli_calculator = MentalLoadCalculator()
advisory_engine = AdvisoryEngine()
early_warning = EarlyWarningSystem()
recommendation_engine = RecommendationEngine()

# Simulated user session data (in production, use database)
user_sessions = {}

class UserSession:
    """Manages individual user session data."""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.predictions_history = []
        self.created_at = datetime.now()
        
    def add_prediction(self, prediction_data):
        """Add prediction to session history."""
        self.predictions_history.append({
            'timestamp': datetime.now().isoformat(),
            'data': prediction_data
        })
        
    def get_7day_history(self):
        """Get predictions from last 7 days."""
        cutoff_date = datetime.now() - timedelta(days=7)
        return [p for p in self.predictions_history 
               if datetime.fromisoformat(p['timestamp']) > cutoff_date]


@app.route('/')
def index():
    """Serve main dashboard."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict_stress():
    """
    Main prediction endpoint.
    Receives physiological and psychological data, returns complete assessment.
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = [
            'heart_rate', 'hrv', 'respiration', 'skin_temp',
            'bp_systolic', 'bp_diastolic', 'cognitive_state', 'emotional_state'
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Prepare features for prediction
        features_dict = {
            'Heart_Rate': float(data['heart_rate']),
            'HRV': float(data['hrv']),
            'Respiration': float(data['respiration']),
            'Skin_Temp': float(data['skin_temp']),
            'BP_Systolic': float(data['bp_systolic']),
            'BP_Diastolic': float(data['bp_diastolic']),
            'Cognitive_State': int(data['cognitive_state']),
            'Emotional_State': int(data['emotional_state'])
        }
        
        # 1. ML Model Prediction
        if predictor:
            ml_prediction = predictor.predict_stress_level(features_dict)
        else:
            # Fallback for demo when model is not available
            ml_prediction = {
                'stress_level': 'Moderate',
                'stress_level_encoded': 2,
                'confidence': 0.5,
                'probabilities': {'Low': 0.2, 'Moderate': 0.5, 'High': 0.3},
                'input_features': features_dict,
                'raw_prediction': 2
            }
        
        # 2. Mental Load Index Calculation
        physiological_data = {k: v for k, v in features_dict.items() 
                             if k != 'Cognitive_State' and k != 'Emotional_State'}
        psychological_data = {
            'Cognitive_State': features_dict['Cognitive_State'],
            'Emotional_State': features_dict['Emotional_State']
        }
        mli_result = mli_calculator.calculate_mli(physiological_data, psychological_data)
        
        # Determine stress level for subsequent modules
        stress_level = ml_prediction['stress_level']
        mli_score = mli_result['mental_load_index']
        
        # 3. Advisory Generation
        advisory = advisory_engine.generate_advisory(stress_level, mli_score, 
                                                    physiological_data)
        
        # 4. Early Warning Assessment
        early_warning.add_reading(stress_level, mli_score, 
                                 features_dict['Heart_Rate'], 
                                 features_dict['HRV'])
        burnout_risk = early_warning.evaluate_burnout_risk()
        escalation = early_warning.predict_escalation_risk()
        
        # 5. Personalized Recommendations
        recommendations = recommendation_engine.generate_recommendations(
            stress_level, mli_score
        )
        
        # Store in user session
        user_id = request.headers.get('X-User-ID', 'default_user')
        if user_id not in user_sessions:
            user_sessions[user_id] = UserSession(user_id)
        
        user_sessions[user_id].add_prediction({
            'features': features_dict,
            'stress_level': stress_level,
            'mli_score': mli_score,
            'confidence': ml_prediction['confidence']
        })
        
        # Compile complete response
        response = {
            'timestamp': datetime.now().isoformat(),
            'input_summary': {
                'heart_rate': features_dict['Heart_Rate'],
                'hrv': features_dict['HRV'],
                'respiration': features_dict['Respiration'],
                'skin_temp': features_dict['Skin_Temp'],
                'bp_systolic': features_dict['BP_Systolic'],
                'bp_diastolic': features_dict['BP_Diastolic'],
                'cognitive_state': features_dict['Cognitive_State'],
                'emotional_state': features_dict['Emotional_State']
            },
            'ml_prediction': {
                'stress_level': ml_prediction['stress_level'],
                'confidence': ml_prediction['confidence'],
                'probabilities': ml_prediction['probabilities']
            },
            'mental_load_index': {
                'score': mli_result['mental_load_index'],
                'status': mli_result['status'],
                'category': mli_result['category'],
                'components': mli_result['component_breakdown']
            },
            'advisory': {
                'message': advisory['main_message'],
                'recommendations': advisory['advice_items'],
                'implementation_duration': advisory['implementation_duration'],
                'priority': advisory['priority_level']
            },
            'early_warning': {
                'burnout_risk_level': burnout_risk['risk_level'],
                'burnout_risk_score': burnout_risk['risk_percentage'],
                'escalation_risk': escalation['escalation_risk'],
                'escalation_confidence': escalation['confidence'],
                'timeframe': escalation['timeframe']
            },
            'recommendations': {
                'stress_level': recommendations['stress_level'],
                'natural_interventions': recommendations['natural_interventions'],
                'otc_options': recommendations['otc_options'],
                'professional_services': recommendations['professional_services'],
                'summary': recommendations['summary']
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/trends', methods=['GET'])
def get_trends():
    """Get 7-day trend data for current user."""
    try:
        user_id = request.headers.get('X-User-ID', 'default_user')
        
        if user_id not in user_sessions:
            return jsonify({'error': 'No data available for this user'}), 404
        
        history = user_sessions[user_id].get_7day_history()
        
        if not history:
            return jsonify({'message': 'No historical data available'}), 200
        
        # Extract trend data
        timestamps = []
        stress_levels = []
        mli_scores = []
        confidences = []
        
        for record in history:
            timestamps.append(record['timestamp'])
            stress_levels.append(record['data']['stress_level'])
            mli_scores.append(record['data']['mli_score'])
            confidences.append(record['data']['confidence'])
        
        summary = early_warning.get_7day_summary()
        
        response = {
            'timestamps': timestamps,
            'stress_levels': stress_levels,
            'mli_scores': mli_scores,
            'confidences': confidences,
            'summary': summary
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/session-info', methods=['GET'])
def get_session_info():
    """Get current session information."""
    try:
        user_id = request.headers.get('X-User-ID', 'default_user')
        
        if user_id not in user_sessions:
            return jsonify({
                'user_id': user_id,
                'predictions_made': 0,
                'session_created': datetime.now().isoformat()
            }), 200
        
        session = user_sessions[user_id]
        
        return jsonify({
            'user_id': user_id,
            'session_created': session.created_at.isoformat(),
            'predictions_made': len(session.predictions_history),
            'last_prediction': (session.predictions_history[-1]['timestamp'] 
                               if session.predictions_history else None)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    """Clear session data for current user."""
    try:
        user_id = request.headers.get('X-User-ID', 'default_user')
        
        if user_id in user_sessions:
            del user_sessions[user_id]
            early_warning.reset_history()
        
        return jsonify({'message': 'Session cleared successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about loaded ML model."""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    response = {
        'model_name': 'Mental Health Stress Classifier',
        'features': predictor.feature_names,
        'stress_classes': list(predictor.label_encoder.classes_),
        'model_path': 'model/mental_health_model.pkl',
        'status': 'Ready'
    }
    
    return jsonify(response), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def create_app():
    """Application factory."""
    return app


if __name__ == '__main__':
    print("="*60)
    print("MENTAL HEALTH MONITORING SYSTEM - FLASK SERVER")
    print("="*60)
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Routes available:")
    print("  GET  /                    - Main dashboard")
    print("  POST /api/predict         - Make predictions")
    print("  GET  /api/trends          - Get 7-day trends")
    print("  GET  /api/session-info    - Session information")
    print("  POST /api/clear-session   - Clear session data")
    print("  GET  /api/model-info      - Model information")
    print("  GET  /api/health          - Health check")
    print("="*60)
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
