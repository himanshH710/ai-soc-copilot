from typing import Dict


THREAT_WEIGHTS = {
    "normal": 0.0,
    "suspicious_activity": 0.6,
    "high_request_rate": 0.7,
    "brute_force": 0.9,
}


def calculate_risk_score(
    anomaly_score: float,
    threat_type: str,
    confidence: float,
) -> Dict[str, object]:
    """
    Calculate a normalized security risk score.
    """

    anomaly_score = max(0.0, min(1.0, float(anomaly_score)))
    confidence = max(0.0, min(1.0, float(confidence)))

    threat_weight = THREAT_WEIGHTS.get(threat_type, 0.5)

    risk_score = (
        0.45 * anomaly_score
        + 0.35 * threat_weight
        + 0.20 * confidence
    )

    risk_score = max(0.0, min(1.0, risk_score))

    if risk_score >= 0.80:
        severity = "CRITICAL"
    elif risk_score >= 0.60:
        severity = "HIGH"
    elif risk_score >= 0.35:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "risk_score": round(risk_score, 3),
        "severity": severity,
        "threat_detected": (
            threat_type != "normal" and risk_score >= 0.35
        ),
    }