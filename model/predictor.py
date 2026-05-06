"""
Mental Health Monitoring System - Predictor Module
Loads trained model and makes stress level predictions
"""

import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

class MentalHealthPredictor:
    """Handles model loading and making predictions."""
    
    def __init__(self, model_path='model/mental_health_model.pkl',
                 scaler_path='model/mental_health_model_scaler.pkl',
                 encoder_path='model/mental_health_model_encoder.pkl',
                 features_path='model/mental_health_model_features.pkl'):
        """
        Initialize predictor with pre-trained model and preprocessing objects.
        
        Args:
            model_path (str): Path to trained model pickle file
            scaler_path (str): Path to feature scaler pickle file
            encoder_path (str): Path to label encoder pickle file
            features_path (str): Path to feature names pickle file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.features_path = features_path
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        self.load_model()
        
    def load_model(self):
        """Load all model components from pickle files."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoder = joblib.load(self.encoder_path)
            self.feature_names = joblib.load(self.features_path)
            
            print("✓ Model loaded successfully")
            print(f"✓ Feature names: {self.feature_names}")
            print(f"✓ Stress level classes: {list(self.label_encoder.classes_)}")
            
        except FileNotFoundError as e:
            print(f"✗ Error loading model: {e}")
            print(f"Please train the model first using train_model.py")
            raise
            
    def _compute_engineered_features(self, features_dict):
        """
        Compute engineered features from base features.
        
        Args:
            features_dict (dict): Dictionary with base physiological features
            
        Returns:
            dict: Updated features_dict with engineered features added
        """
        features = features_dict.copy()
        
        # Create engineered features
        hr = float(features['Heart_Rate'])
        hrv = float(features['HRV'])
        resp = float(features['Respiration'])
        temp = float(features['Skin_Temp'])
        bp_sys = float(features['BP_Systolic'])
        bp_dia = float(features['BP_Diastolic'])
        cog = float(features['Cognitive_State'])
        emo = float(features['Emotional_State'])
        
        # Engineering formulas from train_optimal.py
        features['HR_HRV_Ratio'] = hr / (hrv + 1)
        features['BP_Average'] = (bp_sys + bp_dia) / 2
        features['BP_Diff'] = bp_sys - bp_dia
        features['Psych_Score'] = cog + emo
        features['HR_Resp_Ratio'] = hr / (resp + 0.1)
        features['Temp_Deviation'] = abs(temp - 36.5)
        features['HRV_Norm'] = hrv / 500.0  # Normalize by max HRV
        features['HR_Variability'] = features['HR_HRV_Ratio'] * features['Psych_Score']
        
        return features
    
    def validate_input(self, features_dict):
        """
        Validate input features.
        
        Args:
            features_dict (dict): Dictionary with feature names and values
            
        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        # Get base features (those that are not engineered)
        base_features = ['Heart_Rate', 'HRV', 'Respiration', 'Skin_Temp', 
                        'BP_Systolic', 'BP_Diastolic', 'Cognitive_State', 'Emotional_State']
        
        provided_features = list(features_dict.keys())
        missing_features = [f for f in base_features if f not in provided_features]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Check for non-numeric values
        for feature, value in features_dict.items():
            try:
                float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{feature}' must be numeric, got {value}")
        
        return True
        
    def predict_stress_level(self, features_dict):
        """
        Predict stress level for given input features.
        
        Args:
            features_dict (dict): Dictionary with physiological and psychological features
                                 Example:
                                 {
                                     'Heart_Rate': 75,
                                     'HRV': 45,
                                     'Respiration': 18,
                                     'Skin_Temp': 36.5,
                                     'BP_Systolic': 120,
                                     'BP_Diastolic': 80,
                                     'Cognitive_State': 3,
                                     'Emotional_State': 2
                                 }
        
        Returns:
            dict: Prediction results with stress level, probability, and encoded value
        """
        # Validate input
        self.validate_input(features_dict)
        
        # Compute engineered features
        features_with_engineered = self._compute_engineered_features(features_dict)
        
        # Prepare features in correct order
        X = np.array([[features_with_engineered[f] for f in self.feature_names]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        stress_level_encoded = self.model.predict(X_scaled)[0]
        stress_level = self.label_encoder.inverse_transform([stress_level_encoded])[0]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = np.max(probabilities)
        
        # Create detailed prediction result
        result = {
            'stress_level': stress_level,
            'stress_level_encoded': int(stress_level_encoded),
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'input_features': features_dict,
            'raw_prediction': int(stress_level_encoded)
        }
        
        return result
        
    def batch_predict(self, dataframe):
        """
        Make predictions for multiple samples.
        
        Args:
            dataframe (pd.DataFrame): DataFrame with feature columns
            
        Returns:
            pd.DataFrame: Original data with predictions added
        """
        # Make a copy to add engineered features
        df_with_engineered = dataframe.copy()
        
        # Compute engineered features for each row
        df_with_engineered['HR_HRV_Ratio'] = df_with_engineered['Heart_Rate'] / (df_with_engineered['HRV'] + 1)
        df_with_engineered['BP_Average'] = (df_with_engineered['BP_Systolic'] + df_with_engineered['BP_Diastolic']) / 2
        df_with_engineered['Psych_Score'] = df_with_engineered['Cognitive_State'] + df_with_engineered['Emotional_State']
        
        # Prepare features
        X = df_with_engineered[self.feature_names].values
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidence = np.max(probabilities, axis=1)
        
        # Convert encoded predictions to stress level labels
        stress_levels = self.label_encoder.inverse_transform(predictions)
        
        # Add results to dataframe
        result_df = dataframe.copy()
        result_df['Predicted_Stress_Level'] = stress_levels
        result_df['Stress_Level_Confidence'] = confidence
        
        # Add probability for each class
        for idx, class_name in enumerate(self.label_encoder.classes_):
            result_df[f'Prob_{class_name}'] = probabilities[:, idx]
        
        return result_df
        
    def explain_prediction(self, prediction_result):
        """
        Provide explanation for the prediction.
        
        Args:
            prediction_result (dict): Result from predict_stress_level
            
        Returns:
            str: Explanation text
        """
        stress_level = prediction_result['stress_level']
        confidence = prediction_result['confidence']
        probabilities = prediction_result['probabilities']
        
        explanation = f"""
PREDICTION EXPLANATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted Stress Level: {stress_level}
Confidence Level: {confidence:.2%}

Probability Distribution:
"""
        
        for class_name, prob in probabilities.items():
            bar_length = int(prob * 40)
            bar = '█' * bar_length
            explanation += f"  {class_name:12} {bar} {prob:.4f}\n"
        
        explanation += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        return explanation


def get_sample_input():
    """Return sample input for testing."""
    return {
        'Heart_Rate': 75,
        'HRV': 45,
        'Respiration': 18,
        'Skin_Temp': 36.5,
        'BP_Systolic': 120,
        'BP_Diastolic': 80,
        'Cognitive_State': 3,
        'Emotional_State': 2
    }


if __name__ == "__main__":
    # Initialize predictor
    predictor = MentalHealthPredictor()
    
    # Test with sample input
    print("\n" + "="*60)
    print("TESTING PREDICTOR WITH SAMPLE INPUT")
    print("="*60)
    
    sample_input = get_sample_input()
    print("\nInput Features:")
    for feature, value in sample_input.items():
        print(f"  {feature}: {value}")
    
    # Make prediction
    result = predictor.predict_stress_level(sample_input)
    
    print("\nPrediction Result:")
    print(f"  Stress Level: {result['stress_level']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Probabilities: {result['probabilities']}")
    
    # Explain prediction
    print(predictor.explain_prediction(result))
