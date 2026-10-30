from ml.models.risk_score import calculate_risk_score


def test_risk_score_range():
    result = calculate_risk_score(
        anomaly_score=0.9,
        threat_type="brute_force",
        confidence=0.94,
    )

    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["severity"] in [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ]
    assert isinstance(
        result["threat_detected"],
        bool,
    )