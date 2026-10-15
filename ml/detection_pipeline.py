from typing import Any, Dict, List

from ml.preprocessing.parser import parse_event
from ml.preprocessing.feature_engineering import extract_features
from ml.models.anomaly_detection import AnomalyDetector
from ml.models.classifier import ThreatClassifier
from ml.models.risk_score import calculate_risk_score

def generate_explanation(
    features,
    threat_type,
    anomaly_score,
) -> str:
    """
    Generate a human-readable explanation for the detection result.
    """

    if threat_type == "brute_force":
        return (
            f"Repeated authentication failures detected "
            f"({features.get('failed_attempts', 0):.0f} failed attempts)."
        )

    if threat_type == "high_request_rate":
        return (
            f"Unusually high request activity detected "
            f"({features.get('request_count', 0):.0f} requests)."
        )

    if threat_type == "suspicious_activity":
        return (
            f"Security activity deviates from the expected pattern "
            f"(anomaly score: {anomaly_score:.2f})."
        )

    if threat_type == "normal":
        return "No significant security threat detected."

    return (
        f"Unusual security activity detected "
        f"(anomaly score: {anomaly_score:.2f})."
    )

class ThreatDetectionPipeline:
    """
    End-to-end security threat detection pipeline.
    """

    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.classifier = ThreatClassifier()

    def train(
        self,
        training_features: List[Dict[str, float]],
        labels: List[str],
    ) -> None:
        """
        Train both the anomaly detector and threat classifier.
        """

        if not training_features:
            raise ValueError("Training data cannot be empty.")

        if len(training_features) != len(labels):
            raise ValueError(
                "Training features and labels must have the same length."
            )

        # Train anomaly detector on reference data.
        self.anomaly_detector.fit(training_features)

        # Train threat classifier.
        classifier_features = []

        for features in training_features:
            classifier_features.append(
                {
                    **features,
                    "anomaly_score": 0.0,
                }
            )

        self.classifier.fit(classifier_features, labels)

    def analyze(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze one security event and return a structured result.
        """

        normalized_event = parse_event(event)

        features = extract_features(normalized_event)

        anomaly_result = self.anomaly_detector.predict(features)

        classification = self.classifier.predict(
            features,
            anomaly_result["anomaly_score"],
        )

        risk = calculate_risk_score(
            anomaly_score=anomaly_result["anomaly_score"],
            threat_type=classification["threat_type"],
            confidence=classification["confidence"],
        )

        explanation = generate_explanation(
    features=features,
    threat_type=classification["threat_type"],
    anomaly_score=anomaly_result["anomaly_score"],
    )

        return {
            "event_id": normalized_event["event_id"],
            "user_id": normalized_event["user_id"],
            "threat_detected": risk["threat_detected"],
            "threat_type": classification["threat_type"],
            "risk_score": risk["risk_score"],
            "severity": risk["severity"],
            "confidence": round(
                classification["confidence"],
                3,
            ),
            "anomaly_score": round(
                anomaly_result["anomaly_score"],
                3,
            ),
            "reason": explanation,
        }