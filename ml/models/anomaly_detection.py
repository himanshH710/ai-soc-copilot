from typing import Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest


FEATURE_NAMES = [
    "failed_attempts",
    "successful_attempts",
    "request_count",
    "unique_ips",
    "session_duration",
]


class AnomalyDetector:
    """
    Detect unusual security events using Isolation Forest.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, training_features: List[Dict[str, float]]) -> None:
        """
        Fit the detector using multiple reference samples.
        """

        if not training_features:
            raise ValueError("Training data cannot be empty.")

        vectors = [
            self._to_vector(features)[0]
            for features in training_features
        ]

        self.model.fit(np.array(vectors))
        self.is_fitted = True

    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Return anomaly prediction and anomaly score.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "AnomalyDetector must be fitted before prediction."
            )

        vector = self._to_vector(features)

        prediction = self.model.predict(vector)[0]
        decision_score = self.model.decision_function(vector)[0]

        anomaly_detected = prediction == -1

        anomaly_score = float(
            np.clip(0.5 - decision_score, 0.0, 1.0)
        )

        return {
            "anomaly_detected": anomaly_detected,
            "anomaly_score": anomaly_score,
        }

    @staticmethod
    def _to_vector(
        features: Dict[str, float],
    ) -> np.ndarray:
        """
        Convert feature dictionary into a 2D numerical array.
        """

        vector = [
            float(features.get(name, 0.0))
            for name in FEATURE_NAMES
        ]

        return np.array([vector])