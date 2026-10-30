from ml.detection_pipeline import ThreatDetectionPipeline


def main():
    pipeline = ThreatDetectionPipeline()

    # Load trained models from disk.
    pipeline.load_models()

    test_event = {
        "event_id": "EVT-TEST-001",
        "timestamp": "2026-10-26T12:00:00",
        "user_id": "U001",
        "source": "login_service",
        "event_type": "login_failed",
        "features": {
            "failed_attempts": 20,
            "successful_attempts": 0,
            "request_count": 100,
            "unique_ips": 5,
            "session_duration": 20,
        },
    }

    result = pipeline.analyze(test_event)

    print("\nDetection Result:")
    print(result)


if __name__ == "__main__":
    main()