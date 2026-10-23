from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

from src.data_preprocessing import load_and_prepare_data


SELECTED_THRESHOLD = 0.30


def evaluate_final_model():
    (
        data,
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_and_prepare_data()

    # Train the final Logistic Regression model
    model = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Get malignant probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Apply the locked threshold selected using cross-validation
    predictions = (
        probabilities >= SELECTED_THRESHOLD
    ).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    # Classification metrics
    accuracy = accuracy_score(y_test, predictions)

    sensitivity = tp / (tp + fn)

    specificity = tn / (tn + fp)

    precision = tp / (tp + fp)

    npv = tn / (tn + fn)

    f1 = 2 * precision * sensitivity / (
        precision + sensitivity
    )

    # Probability-based metrics
    brier = brier_score_loss(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    logloss = log_loss(
        y_test,
        probabilities
    )

    print("=" * 60)
    print("FINAL TEST-SET EVALUATION")
    print("=" * 60)

    print(f"\nLocked threshold: {SELECTED_THRESHOLD}")

    print("\nConfusion Matrix:")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    print("\nClassification Metrics:")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"NPV:          {npv:.4f}")
    print(f"F1 Score:     {f1:.4f}")

    print("\nProbability Metrics:")
    print(f"Brier Score:  {brier:.4f}")
    print(f"ROC-AUC:      {roc_auc:.4f}")
    print(f"Log Loss:     {logloss:.4f}")

    return {
        "threshold": SELECTED_THRESHOLD,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "brier": brier,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


if __name__ == "__main__":
    evaluate_final_model()