from ml.preprocessing.parser import parse_event


def test_parse_valid_event():
    event = {
        "event_id": "EVT-001",
        "timestamp": "2026-10-30T10:00:00",
        "user_id": "U001",
        "source": "login_service",
        "event_type": "login_failed",
        "features": {
            "failed_attempts": 5,
        },
    }

    result = parse_event(event)

    assert result["event_id"] == "EVT-001"
    assert result["user_id"] == "U001"
    assert result["features"]["failed_attempts"] == 5


def test_parse_missing_required_field():
    event = {
        "event_id": "EVT-001",
        "timestamp": "2026-10-30T10:00:00",
        "user_id": "U001",
    }

    try:
        parse_event(event)
        assert False
    except ValueError:
        assert True