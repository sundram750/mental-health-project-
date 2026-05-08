from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class MentalHealthPredictor:
    """Lightweight inference wrapper for the saved sklearn model artifacts."""

    base_features = (
        "Heart_Rate",
        "HRV",
        "Respiration",
        "Skin_Temp",
        "BP_Systolic",
        "BP_Diastolic",
        "Cognitive_State",
        "Emotional_State",
    )

    def __init__(
        self,
        model_path: str | Path,
        scaler_path: str | Path,
        encoder_path: str | Path,
        features_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.encoder_path = Path(encoder_path)
        self.features_path = Path(features_path)
        self.model: Any = None
        self.scaler: Any = None
        self.label_encoder: Any = None
        self.feature_names: list[str] = []
        self.load()

    @property
    def classes(self) -> list[Any]:
        if self.label_encoder is None:
            return []
        return list(self.label_encoder.classes_)

    def load(self) -> None:
        missing = [
            str(path)
            for path in (self.model_path, self.scaler_path, self.encoder_path, self.features_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing model artifact(s): {', '.join(missing)}")

        try:
            self._load_artifacts()
        except (ModuleNotFoundError, ValueError):
            self._patch_numpy_legacy_paths()
            self._load_artifacts()

    def _load_artifacts(self) -> None:
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.encoder_path)
        self.feature_names = list(joblib.load(self.features_path))

    def _patch_numpy_legacy_paths(self) -> None:
        """Allow older numpy pickle module paths to resolve on newer runtimes."""
        try:
            sys.modules.setdefault("numpy._core", np.core)
            sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
            sys.modules.setdefault("numpy._core.umath", np.core.umath)
            sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
        except Exception:
            return

    def validate(self, values: dict[str, float]) -> None:
        missing = [name for name in self.base_features if name not in values]
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")

        for name, value in values.items():
            try:
                float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Feature {name} must be numeric") from None

    def engineered_features(self, values: dict[str, float]) -> dict[str, float]:
        self.validate(values)
        features = {name: float(value) for name, value in values.items()}

        heart_rate = features["Heart_Rate"]
        hrv = features["HRV"]
        respiration = features["Respiration"]
        skin_temp = features["Skin_Temp"]
        systolic = features["BP_Systolic"]
        diastolic = features["BP_Diastolic"]
        cognitive = features["Cognitive_State"]
        emotional = features["Emotional_State"]

        features["HR_HRV_Ratio"] = heart_rate / (hrv + 1)
        features["BP_Average"] = (systolic + diastolic) / 2
        features["BP_Diff"] = systolic - diastolic
        features["Psych_Score"] = cognitive + emotional
        features["HR_Resp_Ratio"] = heart_rate / (respiration + 0.1)
        features["Temp_Deviation"] = abs(skin_temp - 36.5)
        features["HRV_Norm"] = hrv / 500.0
        features["HR_Variability"] = features["HR_HRV_Ratio"] * features["Psych_Score"]
        return features

    def predict_stress_level(self, values: dict[str, float]) -> dict[str, Any]:
        features = self.engineered_features(values)
        missing_engineered = [name for name in self.feature_names if name not in features]
        if missing_engineered:
            raise ValueError(f"Missing engineered features: {', '.join(missing_engineered)}")

        matrix = np.array([[features[name] for name in self.feature_names]], dtype=float)
        scaled = self.scaler.transform(matrix)
        encoded = self.model.predict(scaled)[0]
        stress_level = self.label_encoder.inverse_transform([encoded])[0]

        probabilities = self.model.predict_proba(scaled)[0]
        confidence = float(np.max(probabilities))
        probability_map = {
            str(label): float(probability)
            for label, probability in zip(self.label_encoder.classes_, probabilities)
        }

        return {
            "stress_level": str(stress_level),
            "stress_level_encoded": int(encoded),
            "confidence": confidence,
            "probabilities": probability_map,
        }
