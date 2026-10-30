from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.models.anomaly_detection import AnomalyDetector
from ml.models.classifier import ThreatClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "security_events.csv"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"

FEATURE_NAMES = [
    "failed_attempts",
    "successful_attempts",
    "request_count",
    "unique_ips",
    "session_duration",
]


def load_dataset():
    """Load the prototype security dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    """Separate features and labels."""

    X = df[FEATURE_NAMES].copy()
    y = df["threat_type"].copy()

    return X, y


def train_models(X_train, y_train):
    """
    Train the anomaly detector first, then use its
    anomaly scores as an input feature for the classifier.
    """

    training_features = X_train.to_dict(
        orient="records"
    )

    # -------------------------------------------------
    # 1. Train anomaly detector
    # -------------------------------------------------

    anomaly_detector = AnomalyDetector()

    anomaly_detector.fit(
        training_features
    )

    # -------------------------------------------------
    # 2. Generate anomaly scores
    # -------------------------------------------------

    classifier_training_data = []

    for features in training_features:

        anomaly_result = anomaly_detector.predict(
            features
        )

        classifier_training_data.append(
            {
                **features,
                "anomaly_score": anomaly_result[
                    "anomaly_score"
                ],
            }
        )

    # -------------------------------------------------
    # 3. Train threat classifier
    # -------------------------------------------------

    classifier = ThreatClassifier()

    classifier.fit(
        classifier_training_data,
        y_train.tolist(),
    )

    return anomaly_detector, classifier


def save_models(anomaly_detector, classifier):
    """Save trained models to disk."""

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    anomaly_model_path = (
        MODELS_DIR / "anomaly_detector.pkl"
    )

    classifier_model_path = (
        MODELS_DIR / "threat_classifier.pkl"
    )

    joblib.dump(
        anomaly_detector,
        anomaly_model_path,
    )

    joblib.dump(
        classifier,
        classifier_model_path,
    )

    print("\nModels saved successfully.")
    print(f"Anomaly detector: {anomaly_model_path}")
    print(f"Threat classifier: {classifier_model_path}")


def main():
    print("Loading dataset...")

    df = load_dataset()

    print(
        f"Dataset loaded: {len(df)} samples"
    )

    X, y = prepare_data(df)

    print("\nThreat distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("\nData split:")
    print(
        f"Training samples: {len(X_train)}"
    )
    print(
        f"Testing samples: {len(X_test)}"
    )

    print("\nTraining models...")

    anomaly_detector, classifier = train_models(
        X_train,
        y_train,
    )

    print("Models trained successfully.")

    save_models(
        anomaly_detector,
        classifier,
    )


if __name__ == "__main__":
    main()