import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

from src.models import train_models


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification and probability-based metrics.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)

    sensitivity = tp / (tp + fn)

    specificity = tn / (tn + fp)

    precision = tp / (tp + fp)

    f1 = 2 * precision * sensitivity / (
        precision + sensitivity
    )

    brier = brier_score_loss(
        y_true,
        y_prob
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    logloss = log_loss(
        y_true,
        y_prob
    )

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "brier_score": brier,
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def evaluate_models():

    (
        gnb,
        log_reg,
        log_reg_reduced,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_reduced,
        X_test_reduced,
        reduced_features
    ) = train_models()

    # --------------------------------------------------
    # 1. Gaussian Naive Bayes
    # --------------------------------------------------

    gnb_prob = gnb.predict_proba(X_test)[:, 1]

    gnb_pred = (gnb_prob >= 0.5).astype(int)

    gnb_metrics = calculate_metrics(
        y_test,
        gnb_pred,
        gnb_prob
    )

    # --------------------------------------------------
    # 2. Full Logistic Regression
    # --------------------------------------------------

    lr_prob = log_reg.predict_proba(X_test)[:, 1]

    lr_pred = (lr_prob >= 0.5).astype(int)

    lr_metrics = calculate_metrics(
        y_test,
        lr_pred,
        lr_prob
    )

    # --------------------------------------------------
    # 3. Reduced Logistic Regression
    # --------------------------------------------------

    reduced_prob = log_reg_reduced.predict_proba(
        X_test_reduced
    )[:, 1]

    reduced_pred = (
        reduced_prob >= 0.5
    ).astype(int)

    reduced_metrics = calculate_metrics(
        y_test,
        reduced_pred,
        reduced_prob
    )

    # --------------------------------------------------
    # Print Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    models = {
        "Gaussian Naive Bayes": gnb_metrics,
        "Full Logistic Regression": lr_metrics,
        "Reduced Logistic Regression": reduced_metrics
    }

    for model_name, metrics in models.items():

        print("\n" + model_name)
        print("-" * len(model_name))

        print(
            f"Accuracy:      {metrics['accuracy']:.4f}"
        )

        print(
            f"Sensitivity:   {metrics['sensitivity']:.4f}"
        )

        print(
            f"Specificity:   {metrics['specificity']:.4f}"
        )

        print(
            f"Precision:     {metrics['precision']:.4f}"
        )

        print(
            f"F1 Score:      {metrics['f1']:.4f}"
        )

        print(
            f"Brier Score:   {metrics['brier_score']:.4f}"
        )

        print(
            f"ROC-AUC:       {metrics['roc_auc']:.4f}"
        )

        print(
            f"Log Loss:      {metrics['log_loss']:.4f}"
        )

        print(
            f"TN: {metrics['tn']}  "
            f"FP: {metrics['fp']}  "
            f"FN: {metrics['fn']}  "
            f"TP: {metrics['tp']}"
        )

    return models


if __name__ == "__main__":
    evaluate_models()