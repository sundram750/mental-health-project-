"""
Mental Health Monitoring System - Advisory Engine
Generates dynamic advisory messages based on stress level
"""

from datetime import datetime

class AdvisoryEngine:
    """
    Generates personalized advisory messages based on predicted stress level.
    Provides actionable recommendations without numerical data.
    """
    
    # Advisory content organized by stress level
    ADVISORIES = {
        'Low': {
            'message': 'Your mental load is stable. You are managing well!',
            'advice_items': [
                'Maintain your current routine and habits',
                'Light physical activity is recommended',
                'Continue healthy sleep and hydration practices',
                'Engage in activities you enjoy for relaxation',
                'Stay connected with supportive people around you'
            ],
            'duration': 'Continue current practices',
            'priority': 'Maintenance'
        },
        'Moderate': {
            'message': 'Your mental load is elevated. Consider implementing recovery strategies.',
            'advice_items': [
                'Take short recovery breaks every hour',
                'Reduce multitasking and focus on one task at a time',
                'Practice deep breathing exercises (4-7-8 technique)',
                'Ensure adequate sleep (7-9 hours)',
                'Limit caffeine and sugary foods',
                'Spend time in nature or outdoors',
                'Try progressive muscle relaxation techniques'
            ],
            'duration': 'Implement for 3-5 days, reassess',
            'priority': 'Immediate attention needed'
        },
        'High': {
            'message': 'Your mental load is critical. Immediate recovery is advised.',
            'advice_items': [
                'Immediate recovery advised - prioritize self-care',
                'Avoid demanding tasks and meetings for the next few hours',
                'Engage in guided relaxation or meditation',
                'Take scheduled time off if possible',
                'Limit social obligations temporarily',
                'Use calming techniques: breathing, yoga, or tai chi',
                'Seek social support from trusted friends or family',
                'Consider professional consultation if symptoms persist'
            ],
            'duration': '24-48 hours minimum recovery period',
            'priority': 'Urgent - requires immediate action'
        }
    }
    
    def __init__(self):
        """Initialize Advisory Engine."""
        self.history = []
        
    def generate_advisory(self, stress_level, mli_score=None, additional_context=None):
        """
        Generate advisory message based on stress level.
        
        Args:
            stress_level (str): 'Low', 'Moderate', or 'High'
            mli_score (float): Mental Load Index score (optional, for context)
            additional_context (dict): Optional additional context like HRV, HR, etc.
            
        Returns:
            dict: Advisory content with recommendations
        """
        # Normalize stress level input
        stress_level = stress_level.capitalize()
        
        if stress_level not in self.ADVISORIES:
            raise ValueError(f"Invalid stress level: {stress_level}")
        
        advisory_content = self.ADVISORIES[stress_level]
        
        advisory = {
            'stress_level': stress_level,
            'timestamp': datetime.now().isoformat(),
            'main_message': advisory_content['message'],
            'advice_items': advisory_content['advice_items'],
            'implementation_duration': advisory_content['duration'],
            'priority_level': advisory_content['priority'],
            'mli_score': mli_score,
            'additional_context': additional_context or {}
        }
        
        # Store in history
        self.history.append(advisory)
        
        return advisory
        
    def get_formatted_advice(self, advisory):
        """
        Format advisory for display.
        
        Args:
            advisory (dict): Advisory from generate_advisory
            
        Returns:
            str: Formatted advisory text
        """
        stress_level = advisory['stress_level']
        main_message = advisory['main_message']
        advice_items = advisory['advice_items']
        duration = advisory['implementation_duration']
        priority = advisory['priority_level']
        
        # Color coding based on stress level
        emoji_map = {
            'Low': '✓',
            'Moderate': '⚠',
            'High': '⛔'
        }
        emoji = emoji_map.get(stress_level, '•')
        
        output = f"""
╔════════════════════════════════════════════════════════╗
║          {emoji} PERSONALIZED ADVISORY RECOMMENDATIONS {emoji}           ║
╚════════════════════════════════════════════════════════╝

{main_message}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDED ACTIONS:

"""
        for idx, item in enumerate(advice_items, 1):
            output += f"  {idx}. {item}\n"
        
        output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPLEMENTATION TIMELINE: {duration}
PRIORITY LEVEL: {priority}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 NOTE: These are general recommendations based on your
   current stress assessment. Continue lifestyle changes
   gradually to find what works best for you.

"""
        
        return output
        
    def get_quick_advice(self, stress_level):
        """
        Get quick, concise advice message.
        
        Args:
            stress_level (str): 'Low', 'Moderate', or 'High'
            
        Returns:
            str: Short advice message
        """
        stress_level = stress_level.capitalize()
        
        if stress_level not in self.ADVISORIES:
            return "Unable to generate advice for unknown stress level."
        
        return self.ADVISORIES[stress_level]['message']
        
    def get_contextual_advice(self, stress_level, physiological_data):
        """
        Generate advice considering specific physiological markers.
        
        Args:
            stress_level (str): 'Low', 'Moderate', or 'High'
            physiological_data (dict): Contains HR, HRV, respiration, etc.
            
        Returns:
            dict: Contextual advisory with targeted recommendations
        """
        advisory = self.generate_advisory(stress_level)
        
        # Add specific recommendations based on physiological markers
        contextual_recommendations = []
        
        if physiological_data.get('Heart_Rate', 0) > 90:
            contextual_recommendations.append(
                'High heart rate detected - try calming breathing techniques'
            )
        
        if physiological_data.get('HRV', 100) < 30:
            contextual_recommendations.append(
                'Low heart rate variability detected - ensure adequate rest'
            )
        
        if physiological_data.get('Respiration', 16) > 20:
            contextual_recommendations.append(
                'Elevated respiration rate - try deep breathing exercises'
            )
        
        if physiological_data.get('Skin_Temp', 35.5) > 36.2:
            contextual_recommendations.append(
                'Elevated body temperature - try cooling techniques (cool water, ventilation)'
            )
        
        advisory['contextual_recommendations'] = contextual_recommendations
        
        return advisory
        
    def get_history(self):
        """Return advisory history."""
        return self.history
        
    def get_latest_advisory(self):
        """Get the most recent advisory."""
        return self.history[-1] if self.history else None
        
    def clear_history(self):
        """Clear advisory history."""
        self.history = []


def format_advice_for_display(stress_level):
    """
    Simple function to get formatted advice.
    
    Args:
        stress_level (str): 'Low', 'Moderate', or 'High'
        
    Returns:
        dict: Formatted advisory dictionary
    """
    engine = AdvisoryEngine()
    advisory = engine.generate_advisory(stress_level)
    return advisory


if __name__ == "__main__":
    # Test the advisory engine
    engine = AdvisoryEngine()
    
    print("="*60)
    print("ADVISORY ENGINE - TEST")
    print("="*60)
    
    # Test all stress levels
    for stress_level in ['Low', 'Moderate', 'High']:
        advisory = engine.generate_advisory(stress_level, mli_score=50)
        print(engine.get_formatted_advice(advisory))
        print("\n")
