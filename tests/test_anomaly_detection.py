from ml.models.anomaly_detection import AnomalyDetector


def test_anomaly_detector():
    training_data = [
        {
            "failed_attempts": 1,
            "successful_attempts": 10,
            "request_count": 20,
            "unique_ips": 1,
            "session_duration": 300,
        },
        {
            "failed_attempts": 2,
            "successful_attempts": 8,
            "request_count": 25,
            "unique_ips": 1,
            "session_duration": 280,
        },
        {
            "failed_attempts": 1,
            "successful_attempts": 12,
            "request_count": 18,
            "unique_ips": 1,
            "session_duration": 320,
        },
        {
            "failed_attempts": 3,
            "successful_attempts": 7,
            "request_count": 22,
            "unique_ips": 2,
            "session_duration": 250,
        },
    ]

    detector = AnomalyDetector()

    detector.fit(training_data)

    result = detector.predict(
        {
            "failed_attempts": 20,
            "successful_attempts": 0,
            "request_count": 100,
            "unique_ips": 5,
            "session_duration": 20,
        }
    )

    assert "anomaly_detected" in result
    assert "anomaly_score" in result
    assert 0.0 <= result["anomaly_score"] <= 1.0