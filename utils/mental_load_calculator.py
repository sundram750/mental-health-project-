"""
Mental Health Monitoring System - Mental Load Calculator
Calculates Mental Load Index (MLI) from physiological and psychological features
"""

import numpy as np
import pandas as pd
from datetime import datetime

class MentalLoadCalculator:
    """
    Calculates Mental Load Index (MLI) - a composite metric (0-100) that represents
    the overall mental and physiological stress burden.
    """
    
    # Reference ranges for physiological normalization
    NORMALIZATION_RANGES = {
        'Heart_Rate': {'min': 60, 'max': 100},           # Normal resting HR
        'HRV': {'min': 20, 'max': 100},                  # Heart Rate Variability
        'Respiration': {'min': 12, 'max': 20},           # Breaths per minute
        'Skin_Temp': {'min': 33, 'max': 36},             # Skin temperature in Celsius
        'BP_Systolic': {'min': 90, 'max': 140},          # Systolic BP
        'BP_Diastolic': {'min': 60, 'max': 90},          # Diastolic BP
    }
    
    # Weights for each component in MLI calculation
    MLI_WEIGHTS = {
        'heart_rate_stress': 0.20,      # 20%
        'hrv_stress': 0.15,              # 15%
        'respiration_stress': 0.15,      # 15%
        'temp_stress': 0.10,             # 10%
        'bp_stress': 0.15,               # 15%
        'cognitive_state': 0.12,         # 12%
        'emotional_state': 0.13          # 13%
    }
    
    def __init__(self):
        """Initialize Mental Load Calculator."""
        self.history = []
        
    def normalize_value(self, value, feature_name, reverse=False):
        """
        Normalize a value to 0-1 range using feature-specific ranges.
        
        Args:
            value (float): Raw feature value
            feature_name (str): Name of the feature
            reverse (bool): If True, lower values get higher scores (for HRV, temp)
            
        Returns:
            float: Normalized value (0-1)
        """
        if feature_name not in self.NORMALIZATION_RANGES:
            return 0.5  # Default middle value if range unknown
        
        range_info = self.NORMALIZATION_RANGES[feature_name]
        min_val = range_info['min']
        max_val = range_info['max']
        
        # Clamp value to range
        clipped_value = np.clip(value, min_val, max_val)
        
        # Normalize to 0-1
        normalized = (clipped_value - min_val) / (max_val - min_val)
        
        # Reverse if needed (features where lower is better)
        if reverse:
            normalized = 1 - normalized
        
        return float(normalized)
        
    def calculate_stress_from_heart_rate(self, heart_rate):
        """
        Calculate stress component from heart rate.
        Higher HR = higher stress
        """
        normalized_hr = self.normalize_value(heart_rate, 'Heart_Rate')
        return normalized_hr
        
    def calculate_stress_from_hrv(self, hrv):
        """
        Calculate stress component from HRV.
        Lower HRV = higher stress (reverse=True)
        """
        normalized_hrv = self.normalize_value(hrv, 'HRV', reverse=True)
        return normalized_hrv
        
    def calculate_stress_from_respiration(self, respiration):
        """
        Calculate stress component from respiration.
        Higher respiration rate = higher stress
        """
        normalized_resp = self.normalize_value(respiration, 'Respiration')
        return normalized_resp
        
    def calculate_stress_from_temp(self, skin_temp):
        """
        Calculate stress component from skin temperature.
        Abnormal temperature (both high and low) indicates stress
        """
        # Temperature around 35.5°C is optimal for calm state
        optimal_temp = 35.5
        temp_diff = abs(skin_temp - optimal_temp)
        max_possible_diff = 2.5  # Maximum expected deviation
        
        stress = min(temp_diff / max_possible_diff, 1.0)
        return float(stress)
        
    def calculate_stress_from_bp(self, bp_systolic, bp_diastolic):
        """
        Calculate stress component from blood pressure.
        Higher BP = higher stress
        """
        normalized_sys = self.normalize_value(bp_systolic, 'BP_Systolic')
        normalized_dia = self.normalize_value(bp_diastolic, 'BP_Diastolic')
        
        # Average of systolic and diastolic
        bp_stress = (normalized_sys + normalized_dia) / 2
        return float(bp_stress)
        
    def assess_cognitive_state(self, cognitive_score):
        """
        Convert cognitive state score (1-5) to stress component (0-1).
        Cognitive_State: 1=Very Alert, 5=Completely Overwhelmed
        """
        # Normalize 1-5 scale to 0-1 (where 1=low stress, 5=high stress)
        cognitive_stress = (cognitive_score - 1) / 4
        return float(np.clip(cognitive_stress, 0, 1))
        
    def assess_emotional_state(self, emotional_score):
        """
        Convert emotional state score (1-5) to stress component (0-1).
        Emotional_State: 1=Very Calm, 5=Very Anxious
        """
        # Normalize 1-5 scale to 0-1 (where 1=low stress, 5=high stress)
        emotional_stress = (emotional_score - 1) / 4
        return float(np.clip(emotional_stress, 0, 1))
        
    def calculate_mli(self, physiological_data, psychological_data):
        """
        Calculate Mental Load Index (0-100 scale).
        
        Args:
            physiological_data (dict): Contains Heart_Rate, HRV, Respiration,
                                      Skin_Temp, BP_Systolic, BP_Diastolic
            psychological_data (dict): Contains Cognitive_State, Emotional_State
            
        Returns:
            dict: MCL result with index value and status
        """
        # Calculate stress from physiological features
        hr_stress = self.calculate_stress_from_heart_rate(
            physiological_data.get('Heart_Rate')
        )
        hrv_stress = self.calculate_stress_from_hrv(
            physiological_data.get('HRV')
        )
        resp_stress = self.calculate_stress_from_respiration(
            physiological_data.get('Respiration')
        )
        temp_stress = self.calculate_stress_from_temp(
            physiological_data.get('Skin_Temp')
        )
        bp_stress = self.calculate_stress_from_bp(
            physiological_data.get('BP_Systolic'),
            physiological_data.get('BP_Diastolic')
        )
        
        # Calculate stress from psychological features
        cognitive_stress = self.assess_cognitive_state(
            psychological_data.get('Cognitive_State', 3)
        )
        emotional_stress = self.assess_emotional_state(
            psychological_data.get('Emotional_State', 3)
        )
        
        # Weighted combination
        mli = (
            hr_stress * self.MLI_WEIGHTS['heart_rate_stress'] +
            hrv_stress * self.MLI_WEIGHTS['hrv_stress'] +
            resp_stress * self.MLI_WEIGHTS['respiration_stress'] +
            temp_stress * self.MLI_WEIGHTS['temp_stress'] +
            bp_stress * self.MLI_WEIGHTS['bp_stress'] +
            cognitive_stress * self.MLI_WEIGHTS['cognitive_state'] +
            emotional_stress * self.MLI_WEIGHTS['emotional_state']
        )
        
        # Convert to 0-100 scale
        mli_score = mli * 100
        mli_score = float(np.clip(mli_score, 0, 100))
        
        # Determine MLI status
        if mli_score < 40:
            status = "Stable"
            category = "Low"
        elif mli_score < 70:
            status = "Elevated Load"
            category = "Moderate"
        else:
            status = "Critical Load"
            category = "High"
        
        result = {
            'mental_load_index': mli_score,
            'status': status,
            'category': category,
            'component_breakdown': {
                'heart_rate_stress': float(hr_stress),
                'hrv_stress': float(hrv_stress),
                'respiration_stress': float(resp_stress),
                'temperature_stress': float(temp_stress),
                'blood_pressure_stress': float(bp_stress),
                'cognitive_stress': float(cognitive_stress),
                'emotional_stress': float(emotional_stress)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.history.append(result)
        
        return result
        
    def get_formatted_output(self, mli_result):
        """
        Format MLI result for display.
        
        Args:
            mli_result (dict): Result from calculate_mli
            
        Returns:
            str: Formatted text output
        """
        mli = mli_result['mental_load_index']
        status = mli_result['status']
        category = mli_result['category']
        
        output = f"""
╔════════════════════════════════════════════════════════╗
║          MENTAL LOAD INDEX ASSESSMENT                 ║
╚════════════════════════════════════════════════════════╝

Mental Load Index: {mli:.1f}
Status: {status}
Category: {category}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Component Breakdown:
  Heart Rate Stress:      {mli_result['component_breakdown']['heart_rate_stress']:.3f}
  HRV Stress:             {mli_result['component_breakdown']['hrv_stress']:.3f}
  Respiration Stress:     {mli_result['component_breakdown']['respiration_stress']:.3f}
  Temperature Stress:     {mli_result['component_breakdown']['temperature_stress']:.3f}
  Blood Pressure Stress:  {mli_result['component_breakdown']['blood_pressure_stress']:.3f}
  Cognitive Stress:       {mli_result['component_breakdown']['cognitive_stress']:.3f}
  Emotional Stress:       {mli_result['component_breakdown']['emotional_stress']:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return output
        
    def get_history(self):
        """Return all stored predictions."""
        return self.history
        
    def get_history_dataframe(self):
        """Return history as pandas DataFrame."""
        return pd.DataFrame(self.history)
        
    def clear_history(self):
        """Clear prediction history."""
        self.history = []


def get_sample_physiological_data():
    """Return sample physiological data for testing."""
    return {
        'Heart_Rate': 78,
        'HRV': 42,
        'Respiration': 19,
        'Skin_Temp': 35.8,
        'BP_Systolic': 125,
        'BP_Diastolic': 82
    }


def get_sample_psychological_data():
    """Return sample psychological data for testing."""
    return {
        'Cognitive_State': 3,  # 1-5 scale
        'Emotional_State': 4   # 1-5 scale
    }


if __name__ == "__main__":
    # Test the calculator
    calculator = MentalLoadCalculator()
    
    print("="*60)
    print("MENTAL LOAD INDEX CALCULATOR - TEST")
    print("="*60)
    
    # Test with different input combinations
    test_cases = [
        # Low stress case
        {
            'phys': {'Heart_Rate': 65, 'HRV': 80, 'Respiration': 14, 
                    'Skin_Temp': 35.5, 'BP_Systolic': 110, 'BP_Diastolic': 70},
            'psych': {'Cognitive_State': 1, 'Emotional_State': 1},
            'name': 'LOW STRESS CASE'
        },
        # Moderate stress case
        {
            'phys': {'Heart_Rate': 80, 'HRV': 50, 'Respiration': 18,
                    'Skin_Temp': 35.8, 'BP_Systolic': 125, 'BP_Diastolic': 82},
            'psych': {'Cognitive_State': 3, 'Emotional_State': 3},
            'name': 'MODERATE STRESS CASE'
        },
        # High stress case
        {
            'phys': {'Heart_Rate': 95, 'HRV': 25, 'Respiration': 22,
                    'Skin_Temp': 36.5, 'BP_Systolic': 140, 'BP_Diastolic': 90},
            'psych': {'Cognitive_State': 5, 'Emotional_State': 5},
            'name': 'HIGH STRESS CASE'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        result = calculator.calculate_mli(test_case['phys'], test_case['psych'])
        print(calculator.get_formatted_output(result))
