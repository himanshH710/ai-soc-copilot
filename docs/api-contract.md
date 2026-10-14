# API Contracts

## 1. ML Detection Result

The ML module produces a normalized detection result.

```json
{
  "event_id": "EVT-1001",
  "anomaly_score": 0.91,
  "risk_score": 87,
  "risk_level": "HIGH",
  "attack_type": "BRUTE_FORCE",
  "model_version": "v1.0"
}