from typing import Any, Dict


def extract_features(event: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a normalized security event into numerical features
    used by the detection models.
    """

    event_features = event.get("features", {})

    features = {
        "failed_attempts": float(
            event_features.get("failed_attempts", 0)
        ),
        "successful_attempts": float(
            event_features.get("successful_attempts", 0)
        ),
        "request_count": float(
            event_features.get("request_count", 0)
        ),
        "unique_ips": float(
            event_features.get("unique_ips", 0)
        ),
        "session_duration": float(
            event_features.get("session_duration", 0)
        ),
    }

    return features