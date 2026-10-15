from typing import Dict

from sklearn.ensemble import RandomForestClassifier


class ThreatClassifier:
    """
    Classify security events into threat categories.

    Initial threat categories:
    - normal
    - brute_force
    - suspicious_activity
    - high_request_rate
    """

    def __init__(self, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
        )
        self.is_fitted = False

    def fit(self, training_data, labels) -> None:
        """
        Train the classifier.

        training_data:
            List of numerical feature dictionaries.

        labels:
            Corresponding threat labels.
        """

        feature_names = [
            "failed_attempts",
            "successful_attempts",
            "request_count",
            "unique_ips",
            "session_duration",
            "anomaly_score",
        ]

        vectors = [
            [float(row.get(name, 0.0)) for name in feature_names]
            for row in training_data
        ]

        self.model.fit(vectors, labels)
        self.is_fitted = True

    def predict(
        self,
        features: Dict[str, float],
        anomaly_score: float = 0.0,
    ) -> Dict[str, object]:
        """
        Predict the threat category and confidence.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "ThreatClassifier must be fitted before prediction."
            )

        feature_names = [
            "failed_attempts",
            "successful_attempts",
            "request_count",
            "unique_ips",
            "session_duration",
            "anomaly_score",
        ]

        vector = [
            float(features.get(name, 0.0))
            if name != "anomaly_score"
            else float(anomaly_score)
            for name in feature_names
        ]

        prediction = self.model.predict([vector])[0]

        probabilities = self.model.predict_proba([vector])[0]
        confidence = float(max(probabilities))

        return {
            "threat_type": str(prediction),
            "confidence": confidence,
        }