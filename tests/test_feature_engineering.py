from ml.preprocessing.feature_engineering import extract_features


def test_extract_features():
    event = {
        "features": {
            "failed_attempts": 10,
            "successful_attempts": 2,
            "request_count": 100,
            "unique_ips": 5,
            "session_duration": 60,
        }
    }

    result = extract_features(event)

    assert result["failed_attempts"] == 10.0
    assert result["successful_attempts"] == 2.0
    assert result["request_count"] == 100.0
    assert result["unique_ips"] == 5.0
    assert result["session_duration"] == 60.0