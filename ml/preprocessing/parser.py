from typing import Any, Dict


REQUIRED_FIELDS = {
    "event_id",
    "timestamp",
    "user_id",
    "source",
    "event_type",
}


def parse_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize an incoming security event.

    Expected event format:
    {
        "event_id": "EVT-001",
        "timestamp": "2026-10-15T10:30:00",
        "user_id": "U001",
        "source": "login_service",
        "event_type": "login_failed",
        "features": {
            ...
        }
    }
    """

    if not isinstance(event, dict):
        raise TypeError("Event must be a dictionary.")

    missing_fields = REQUIRED_FIELDS - event.keys()

    if missing_fields:
        raise ValueError(
            f"Missing required event fields: {sorted(missing_fields)}"
        )

    normalized_event = {
        "event_id": str(event["event_id"]),
        "timestamp": str(event["timestamp"]),
        "user_id": str(event["user_id"]),
        "source": str(event["source"]),
        "event_type": str(event["event_type"]),
        "features": event.get("features", {}),
    }

    if not isinstance(normalized_event["features"], dict):
        raise TypeError("Event 'features' must be a dictionary.")

    return normalized_event