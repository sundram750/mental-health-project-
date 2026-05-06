"""
Mental Health Monitoring System - Recommendation Engine
Generates personalized intervention recommendations based on stress level
"""

from datetime import datetime

class RecommendationEngine:
    """
    Generates tiered personalized recommendations (Natural, OTC, Professional)
    based on stress level and psychological markers.
    """
    
    MEDICAL_DISCLAIMER = """
    ⚠ MEDICAL DISCLAIMER:
    These recommendations are for informational purposes only and should not
    be considered as medical advice. Always consult with a healthcare professional
    before starting any new supplement, medication, or treatment regimen.
    The system is designed to support, not replace, professional medical guidance.
    """
    
    RECOMMENDATIONS = {
        'Low': {
            'natural': [
                'Light yoga or stretching',
                '10-minute mindfulness'
            ],
            'otc': [
                'Not required'
            ],
            'professional': [
                'Not required'
            ],
            'summary': 'Maintain current wellness routine'
        },
        'Moderate': {
            'natural': [
                'Ashwagandha support',
                'Brahmi for cognitive clarity',
                '15-minute breathing practice'
            ],
            'otc': [
                'Magnesium supplementation',
                'Herbal calming formulation'
            ],
            'professional': [
                'Consider consultation if persists for several days'
            ],
            'summary': 'Implement natural and OTC interventions; monitor closely'
        },
        'High': {
            'natural': [
                'Immediate relaxation protocol',
                'Digital detox period'
            ],
            'otc': [
                'Short-term sleep support (if needed)'
            ],
            'professional': [
                'Strongly recommend psychologist consultation',
                'Stress management program enrollment'
            ],
            'summary': 'Urgent professional support strongly recommended'
        }
    }
    
    def __init__(self):
        """Initialize Recommendation Engine."""
        self.history = []
        
    def generate_recommendations(self, stress_level, mli_score=None, 
                                duration_days=None):
        """
        Generate personalized recommendations based on stress level.
        
        Args:
            stress_level (str): 'Low', 'Moderate', 'High'
            mli_score (float): Mental Load Index (optional)
            duration_days (int): How many days at current level (optional)
            
        Returns:
            dict: Tiered recommendations (natural, OTC, professional)
        """
        stress_level = stress_level.capitalize()
        
        if stress_level not in self.RECOMMENDATIONS:
            raise ValueError(f"Invalid stress level: {stress_level}")
        
        recommendations = self.RECOMMENDATIONS[stress_level]
        
        result = {
            'stress_level': stress_level,
            'mli_score': mli_score,
            'duration_at_level': duration_days,
            'timestamp': datetime.now().isoformat(),
            'natural_interventions': recommendations['natural'],
            'otc_options': recommendations['otc'],
            'professional_services': recommendations['professional'],
            'summary': recommendations['summary'],
            'medical_disclaimer': self.MEDICAL_DISCLAIMER
        }
        
        # Store in history
        self.history.append(result)
        
        return result
        
    def get_formatted_recommendations(self, recommendations):
        """
        Format recommendations for display.
        
        Args:
            recommendations (dict): From generate_recommendations
            
        Returns:
            str: Formatted recommendation text
        """
        stress_level = recommendations['stress_level']
        natural = recommendations['natural_interventions']
        otc = recommendations['otc_options']
        professional = recommendations['professional_services']
        summary = recommendations['summary']
        
        # Emoji based on stress level
        emoji_map = {
            'Low': '✓',
            'Moderate': '⚠',
            'High': '🔴'
        }
        emoji = emoji_map.get(stress_level, '•')
        
        output = f"""
╔════════════════════════════════════════════════════════╗
║  {emoji}  PERSONALIZED INTERVENTION RECOMMENDATIONS  {emoji}  ║
╚════════════════════════════════════════════════════════╝

STRESS LEVEL: {stress_level.upper()}
Summary: {summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌿 NATURAL INTERVENTIONS:
"""
        
        if natural:
            for idx, item in enumerate(natural, 1):
                output += f"   {idx}. {item}\n"
        else:
            output += "   (Not required for current stress level)\n"
        
        output += f"""
💊 OTC SUPPLEMENTS & REMEDIES:
"""
        
        if otc:
            for idx, item in enumerate(otc, 1):
                output += f"   {idx}. {item}\n"
        else:
            output += "   (Not required for current stress level)\n"
        
        output += f"""
👨‍⚕️ PROFESSIONAL SERVICES:
"""
        
        if professional:
            for idx, item in enumerate(professional, 1):
                output += f"   {idx}. {item}\n"
        else:
            output += "   (Not required for current stress level)\n"
        
        output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{self.MEDICAL_DISCLAIMER}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return output
        
    def get_quick_recommendation(self, stress_level):
        """
        Get quick recommendation summary.
        
        Args:
            stress_level (str): 'Low', 'Moderate', 'High'
            
        Returns:
            str: Quick recommendation message
        """
        stress_level = stress_level.capitalize()
        
        if stress_level not in self.RECOMMENDATIONS:
            return "Unable to generate recommendations."
        
        return self.RECOMMENDATIONS[stress_level]['summary']
        
    def escalate_recommendation(self, stress_level, previous_recommendation=None):
        """
        Escalate recommendation if stress persists or worsens.
        
        Args:
            stress_level (str): Current stress level
            previous_recommendation (dict): Previous recommendation (optional)
            
        Returns:
            dict: Escalated recommendation
        """
        result = self.generate_recommendations(stress_level)
        
        if stress_level == 'High':
            result['urgency'] = 'CRITICAL'
            result['action_required'] = 'Immediate'
            result['suggested_contacts'] = [
                'Local Mental Health Crisis Hotline',
                'Psychiatrist or Psychologist',
                'Hospital Emergency Department (if severe)',
                'Trusted friend or family member'
            ]
        elif stress_level == 'Moderate' and previous_recommendation:
            if previous_recommendation.get('stress_level') == 'Moderate':
                result['urgency'] = 'Elevated'
                result['action_required'] = 'Within 3-5 days'
                result['escalation_note'] = (
                    'Stress level remaining elevated. '
                    'Increase self-care intensity and consider professional consultation.'
                )
        
        return result
        
    def get_dosage_guidance(self, supplement_name):
        """
        Get dosage guidance for common supplements.
        
        Args:
            supplement_name (str): Name of supplement
            
        Returns:
            str: Dosage and usage information with disclaimer
        """
        dosages = {
            'ashwagandha': '300-600mg daily, take with food, typically shown benefits after 4-8 weeks',
            'brahmi': '300-450mg daily, preferably in divided doses',
            'magnesium': '200-400mg daily, take with evening meal',
            'valerian': '400-900mg 30 minutes before bedtime',
            'l-theanine': '100-200mg daily, can be taken with or without food',
            'passionflower': '0.5-4g fresh herb daily or standardized extract',
            'omega-3': '1000-2000mg daily of combined EPA+DHA'
        }
        
        supplement_lower = supplement_name.lower()
        
        if supplement_lower in dosages:
            return f"""
⚠ DOSAGE INFORMATION FOR {supplement_name.upper()}:
{dosages[supplement_lower]}

IMPORTANT: Consult with a healthcare provider before starting any supplement,
especially if taking other medications or have existing health conditions.
"""
        else:
            return f"Dosage information not available for {supplement_name}"
        
    def get_history(self):
        """Return recommendation history."""
        return self.history
        
    def get_latest_recommendation(self):
        """Get the most recent recommendation."""
        return self.history[-1] if self.history else None
        
    def clear_history(self):
        """Clear recommendation history."""
        self.history = []


def format_recommendations_for_display(stress_level):
    """
    Simple function to get formatted recommendations.
    
    Args:
        stress_level (str): 'Low', 'Moderate', 'High'
        
    Returns:
        str: Formatted recommendation text
    """
    engine = RecommendationEngine()
    recommendations = engine.generate_recommendations(stress_level)
    return engine.get_formatted_recommendations(recommendations)


if __name__ == "__main__":
    # Test the recommendation engine
    engine = RecommendationEngine()
    
    print("="*60)
    print("RECOMMENDATION ENGINE - TEST")
    print("="*60)
    
    # Test all stress levels
    for stress_level in ['Low', 'Moderate', 'High']:
        recommendations = engine.generate_recommendations(stress_level, mli_score=50)
        print(engine.get_formatted_recommendations(recommendations))
        print("\n")
