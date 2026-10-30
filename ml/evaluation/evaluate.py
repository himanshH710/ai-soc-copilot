from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from ml.training.train import (
    load_dataset,
    prepare_data,
    train_models,
)


def main():
    print("Loading dataset...")

    df = load_dataset()

    X, y = prepare_data(df)

    # Use the same split configuration as training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print(
        f"Evaluating on {len(X_test)} unseen samples..."
    )

    # Train only on the training portion.
    anomaly_detector, classifier = train_models(
        X_train,
        y_train,
    )

    # -------------------------------------------------
    # Predict each unseen test sample
    # -------------------------------------------------

    predictions = []

    for _, row in X_test.iterrows():

        features = row.to_dict()

        # First detect whether the event is anomalous.
        anomaly_result = anomaly_detector.predict(
            features
        )

        # Pass the anomaly score to the classifier.
        classification_result = classifier.predict(
            features,
            anomaly_score=anomaly_result[
                "anomaly_score"
            ],
        )

        predictions.append(
            classification_result["threat_type"]
        )

    # -------------------------------------------------
    # Evaluation metrics
    # -------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(
        f"\nTest samples: {len(y_test)}"
    )

    print(
        f"Predictions: {len(predictions)}"
    )

    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # -------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------

    labels = sorted(y.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("Confusion Matrix:")

    print("\nLabels:")
    print(labels)

    print("\nMatrix:")
    print(matrix)


if __name__ == "__main__":
    main()