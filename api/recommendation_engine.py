from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class RecommendationEngine:
    """Tiered recommendations for the mental health research prototype."""

    disclaimer = (
        "These recommendations are informational only and do not replace "
        "professional medical advice, diagnosis, treatment, or emergency care."
    )

    content = {
        "Low": {
            "natural_interventions": [
                "Maintain current sleep and movement routine.",
                "Use light stretching or a short walk to preserve baseline stability.",
                "Continue normal hydration and meal timing.",
            ],
            "otc_options": ["Not required for the current reading."],
            "professional_services": ["Not required for the current reading."],
            "summary": "Maintain current wellness routine.",
        },
        "Moderate": {
            "natural_interventions": [
                "Practice slow breathing for 5 to 10 minutes.",
                "Reduce multitasking and schedule a short recovery break.",
                "Use light yoga, mindfulness, or a quiet walk.",
            ],
            "otc_options": [
                "Consider magnesium only if appropriate for you.",
                "Use caffeine moderation and avoid late stimulant intake.",
            ],
            "professional_services": [
                "Consider consultation if elevated readings persist for several days.",
            ],
            "summary": "Use recovery strategies and monitor the next readings closely.",
        },
        "High": {
            "natural_interventions": [
                "Move into an immediate relaxation protocol.",
                "Reduce digital and environmental stimulation.",
                "Postpone demanding tasks until the reading stabilizes.",
            ],
            "otc_options": [
                "Use sleep or calming supports only with appropriate guidance.",
            ],
            "professional_services": [
                "Strongly consider a psychologist or counselor consultation.",
                "Seek urgent help if symptoms feel unsafe, severe, or escalating.",
            ],
            "summary": "Immediate recovery and professional support are recommended.",
        },
    }

    def generate(self, stress_level: str, mli_score: int | None = None) -> dict[str, Any]:
        key = stress_level if stress_level in self.content else "Moderate"
        result = dict(self.content[key])
        result.update(
            {
                "stress_level": key,
                "mli_score": mli_score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "medical_disclaimer": self.disclaimer,
            }
        )
        return result
