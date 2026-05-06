"""
Mental Health Monitoring System - Early Warning Module
Predicts stress escalation risk based on trends and patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque

class EarlyWarningSystem:
    """
    Monitors stress trends and predicts early warning signals for burnout risk.
    Analyzes 7-day history and detects concerning patterns.
    """
    
    def __init__(self, history_size=7):
        """
        Initialize Early Warning System.
        
        Args:
            history_size (int): Number of days to track (default: 7)
        """
        self.history_size = history_size
        self.stress_history = deque(maxlen=history_size)
        self.hrv_history = deque(maxlen=history_size)
        self.hr_history = deque(maxlen=history_size)
        self.mli_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
    def add_reading(self, stress_level, mli_score, heart_rate, hrv, timestamp=None):
        """
        Add a new stress reading to history.
        
        Args:
            stress_level (str): 'Low', 'Moderate', 'High'
            mli_score (float): Mental Load Index (0-100)
            heart_rate (float): Current heart rate
            hrv (float): Heart Rate Variability
            timestamp (datetime): Timestamp of reading (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Encode stress level
        stress_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
        stress_encoded = stress_mapping.get(stress_level, 2)
        
        self.stress_history.append(stress_encoded)
        self.mli_history.append(mli_score)
        self.hr_history.append(heart_rate)
        self.hrv_history.append(hrv)
        self.timestamps.append(timestamp)
        
    def calculate_trend_slope(self):
        """
        Calculate the slope of stress trend (increasing/decreasing).
        
        Returns:
            float: Slope value (positive = increasing stress)
        """
        if len(self.stress_history) < 2:
            return 0
        
        x = np.arange(len(self.stress_history))
        y = np.array(self.stress_history)
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
        
    def calculate_mli_trend_slope(self):
        """Calculate slope for MLI trend."""
        if len(self.mli_history) < 2:
            return 0
        
        x = np.arange(len(self.mli_history))
        y = np.array(self.mli_history)
        
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
        
    def calculate_hrv_decline_rate(self):
        """
        Calculate the rate of HRV decline.
        Declining HRV often indicates increasing stress.
        
        Returns:
            float: Rate of decline (negative = worsening)
        """
        if len(self.hrv_history) < 2:
            return 0
        
        x = np.arange(len(self.hrv_history))
        y = np.array(self.hrv_history)
        
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
        
    def detect_trend_pattern(self):
        """
        Detect the overall trend pattern.
        
        Returns:
            str: 'Stable', 'Increasing', or 'Consistently High'
        """
        if len(self.stress_history) < 3:
            return 'Insufficient Data'
        
        current_level = self.stress_history[-1]
        trend_slope = self.calculate_trend_slope()
        
        # Check if consistently high
        recent_high_count = sum(1 for x in list(self.stress_history)[-3:] 
                               if x == 3)  # 3 = High
        
        if recent_high_count >= 2:
            return 'Consistently High'
        
        # Check trend direction
        if trend_slope > 0.1:
            return 'Increasing'
        elif trend_slope < -0.1:
            return 'Decreasing'
        else:
            return 'Stable'
        
    def evaluate_burnout_risk(self):
        """
        Evaluate burnout risk based on stress patterns.
        
        Returns:
            dict: Risk assessment with level and description
        """
        if len(self.stress_history) < 2:
            return {
                'risk_level': 'Insufficient Data',
                'risk_percentage': 0,
                'description': 'Need more data for accurate assessment'
            }
        
        current_stress = self.stress_history[-1]
        trend_slope = self.calculate_trend_slope()
        hrv_decline = self.calculate_hrv_decline_rate()
        mli_slope = self.calculate_mli_trend_slope()
        trend_pattern = self.detect_trend_pattern()
        
        # Calculate risk score (0-100)
        risk_score = 0
        
        # Current stress level contribution (40%)
        if current_stress == 3:  # High
            risk_score += 40
        elif current_stress == 2:  # Moderate
            risk_score += 20
        
        # Trend contribution (30%)
        if trend_slope > 0.15:
            risk_score += 30  # Rapidly increasing
        elif trend_slope > 0.05:
            risk_score += 15  # Gradually increasing
        
        # HRV decline contribution (20%)
        if hrv_decline < -2:
            risk_score += 20  # Significant HRV decline
        elif hrv_decline < 0:
            risk_score += 10  # Some HRV decline
        
        # Pattern contribution (10%)
        if trend_pattern == 'Consistently High':
            risk_score += 10
        
        risk_score = min(risk_score, 100)
        
        # Determine risk level
        if risk_score < 30:
            risk_level = 'Low'
        elif risk_score < 60:
            risk_level = 'Moderate'
        else:
            risk_level = 'High'
        
        return {
            'risk_level': risk_level,
            'risk_percentage': float(risk_score),
            'current_stress': current_stress,
            'trend_pattern': trend_pattern,
            'trend_slope': trend_slope,
            'hrv_decline_rate': hrv_decline,
            'mli_trend_slope': mli_slope
        }
        
    def predict_escalation_risk(self):
        """
        Predict if stress will escalate in next 48-72 hours.
        
        Returns:
            dict: Escalation prediction with timeframe and confidence
        """
        if len(self.stress_history) < 3:
            return {
                'escalation_risk': 'Unknown',
                'timeframe': 'Insufficient data',
                'confidence': 0,
                'description': 'Need 3+ readings for escalation prediction'
            }
        
        trend_slope = self.calculate_trend_slope()
        mli_slope = self.calculate_mli_trend_slopes()
        hrv_decline = self.calculate_hrv_decline_rate()
        current_stress = self.stress_history[-1]
        
        # Calculate escalation risk
        escalation_indicators = 0
        
        if trend_slope > 0.1:  # Stress increasing
            escalation_indicators += 1
        
        if mli_slope > 5:  # MLI increasing rapidly
            escalation_indicators += 1
        
        if hrv_decline < -1:  # HRV declining
            escalation_indicators += 1
        
        if current_stress >= 2:  # Already moderate or high
            escalation_indicators += 1
        
        # Determine escalation probability
        escalation_confidence = min(escalation_indicators / 4 * 100, 100)
        
        if escalation_confidence < 30:
            risk_assessment = 'Low Risk - No immediate escalation predicted'
            timeframe = '7+ days'
        elif escalation_confidence < 60:
            risk_assessment = 'Moderate Risk - Escalation possible within 48-72 hours'
            timeframe = '48-72 hours'
        else:
            risk_assessment = 'High Risk - Significant escalation predicted'
            timeframe = '24-48 hours'
        
        return {
            'escalation_risk': risk_assessment,
            'timeframe': timeframe,
            'confidence': float(escalation_confidence),
            'indicators_detected': escalation_indicators,
            'description': self.get_escalation_description(
                escalation_confidence, escalation_indicators
            )
        }
        
    def calculate_hrv_decline_rates(self):
        """Calculate HRV decline rate (alternate method)."""
        return self.calculate_hrv_decline_rate()
        
    def calculate_mli_trend_slopes(self):
        """Calculate MLI trend slope (alternate method)."""
        return self.calculate_mli_trend_slope()
        
    def get_escalation_description(self, confidence, indicators):
        """Generate description for escalation prediction."""
        if confidence < 30:
            return "Stress levels are stable. Continue current wellness practices."
        elif confidence < 60:
            return f"Detected {indicators} concerning indicators. " \
                   "Consider increasing self-care practices in advance."
        else:
            return f"Multiple stress indicators detected ({indicators} of 4). " \
                   "Urgent intervention recommended. Seek support if needed."
        
    def generate_warning_message(self, current_stress, mli_score):
        """
        Generate early warning message based on current status.
        
        Args:
            current_stress (str): 'Low', 'Moderate', 'High'
            mli_score (float): Current Mental Load Index
            
        Returns:
            str: Warning message
        """
        burnout_risk = self.evaluate_burnout_risk()
        escalation = self.predict_escalation_risk()
        
        if current_stress == 'High':
            if burnout_risk['risk_level'] == 'High':
                return (f"⛔ CRITICAL: Burnout risk elevated if pattern persists. "
                       f"Immediate professional consultation strongly recommended.")
            else:
                return (f"⚠ WARNING: High stress detected. "
                       f"Consider implementing recovery strategies within 24 hours.")
        
        elif current_stress == 'Moderate':
            if escalation['confidence'] > 60:
                return (f"⚠ HIGH STRESS RISK: Escalation possible within 48-72 hours. "
                       f"Proactive stress management advised.")
            else:
                return (f"ℹ NOTICE: Moderate stress detected. "
                       f"Continue monitoring and maintain wellness practices.")
        
        else:  # Low stress
            return (f"✓ SAFE: No immediate risk predicted. "
                   f"Maintain current wellness routine.")
        
    def get_7day_summary(self):
        """Get summary of 7-day history."""
        if not self.stress_history:
            return None
        
        stress_mapping_reverse = {1: 'Low', 2: 'Moderate', 3: 'High'}
        
        summary = {
            'period': f'Last {len(self.stress_history)} days',
            'readings_count': len(self.stress_history),
            'stress_sequence': [stress_mapping_reverse.get(x, 'Unknown') 
                               for x in self.stress_history],
            'average_mli': float(np.mean(self.mli_history)) if self.mli_history else 0,
            'average_hr': float(np.mean(self.hr_history)) if self.hr_history else 0,
            'average_hrv': float(np.mean(self.hrv_history)) if self.hrv_history else 0,
            'trend_pattern': self.detect_trend_pattern(),
            'mli_change': float(self.mli_history[-1] - self.mli_history[0]) 
                         if len(self.mli_history) > 1 else 0
        }
        
        return summary
        
    def reset_history(self):
        """Clear all history."""
        self.stress_history.clear()
        self.hrv_history.clear()
        self.hr_history.clear()
        self.mli_history.clear()
        self.timestamps.clear()


def format_early_warning_output(warning_system):
    """Format early warning system output for display."""
    
    burnout_risk = warning_system.evaluate_burnout_risk()
    escalation = warning_system.predict_escalation_risk()
    summary = warning_system.get_7day_summary()
    
    if not summary:
        return "No historical data available yet."
    
    output = f"""
╔════════════════════════════════════════════════════════╗
║            EARLY WARNING SYSTEM ALERT                  ║
╚════════════════════════════════════════════════════════╝

7-DAY TREND MONITORING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Trend Pattern: {summary['trend_pattern']}
  Average Mental Load: {summary['average_mli']:.1f}/100
  Average Heart Rate: {summary['average_hr']:.0f} BPM
  Average HRV: {summary['average_hrv']:.1f}

BURNOUT RISK ASSESSMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Risk Level: {burnout_risk['risk_level']}
  Risk Score: {burnout_risk['risk_percentage']:.1f}%

ESCALATION PREDICTION (48-72 hours):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {escalation['escalation_risk']}
  Confidence: {escalation['confidence']:.1f}%
  Timeframe: {escalation['timeframe']}

"""
    
    return output


if __name__ == "__main__":
    # Test the early warning system
    import random
    
    warning_system = EarlyWarningSystem()
    
    print("="*60)
    print("EARLY WARNING SYSTEM - TEST")
    print("="*60)
    
    # Simulate 7 days of readings with increasing stress
    stress_levels = ['Low', 'Low', 'Moderate', 'Moderate', 'High', 'High', 'High']
    
    for i, stress in enumerate(stress_levels):
        day = datetime.now() - timedelta(days=7-i)
        mli = 20 + (i * 10)  # Increasing MLI
        hr = 70 + (i * 3)    # Increasing HR
        hrv = 70 - (i * 8)   # Decreasing HRV
        
        warning_system.add_reading(stress, mli, hr, hrv, day)
    
    print(format_early_warning_output(warning_system))
