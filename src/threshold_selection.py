import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from src.data_preprocessing import load_and_prepare_data


def calculate_threshold_metrics(y_true, probabilities, threshold):
    """
    Calculate classification metrics for a given probability threshold.
    """

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    sensitivity = tp / (tp + fn)

    specificity = tn / (tn + fp)

    youden_j = sensitivity + specificity - 1

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_j": youden_j
    }


def select_threshold():

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    data, X, y, X_train, X_test, y_train, y_test = (
        load_and_prepare_data()
    )

    # --------------------------------------------------
    # Thresholds to evaluate
    # --------------------------------------------------

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50
    ]

    # --------------------------------------------------
    # 5-Fold Stratified Cross-Validation
    # --------------------------------------------------

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    threshold_results = {
        threshold: []
        for threshold in thresholds
    }

    for train_index, validation_index in skf.split(
        X_train,
        y_train
    ):

        X_cv_train = X_train.iloc[train_index]
        X_cv_validation = X_train.iloc[validation_index]

        y_cv_train = y_train.iloc[train_index]
        y_cv_validation = y_train.iloc[validation_index]

        # Train logistic regression on CV training fold
        model = LogisticRegression(
            max_iter=10000,
            random_state=42
        )

        model.fit(
            X_cv_train,
            y_cv_train
        )

        # Probability of malignant class
        probabilities = model.predict_proba(
            X_cv_validation
        )[:, 1]

        # Evaluate every threshold
        for threshold in thresholds:

            metrics = calculate_threshold_metrics(
                y_cv_validation,
                probabilities,
                threshold
            )

            threshold_results[threshold].append(
                metrics
            )

    # --------------------------------------------------
    # Average CV results
    # --------------------------------------------------

    results = []

    for threshold in thresholds:

        fold_results = threshold_results[threshold]

        results.append({
            "threshold": threshold,
            "accuracy": np.mean([
                result["accuracy"]
                for result in fold_results
            ]),
            "sensitivity": np.mean([
                result["sensitivity"]
                for result in fold_results
            ]),
            "specificity": np.mean([
                result["specificity"]
                for result in fold_results
            ]),
            "youden_j": np.mean([
                result["youden_j"]
                for result in fold_results
            ])
        })

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # Select threshold using maximum Youden's J
    # --------------------------------------------------

    best_index = results_df["youden_j"].idxmax()

    selected_threshold = results_df.loc[
        best_index,
        "threshold"
    ]

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("THRESHOLD SELECTION")
    print("=" * 60)

    print("\n5-Fold Stratified Cross-Validation")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "accuracy": "{:.4f}".format,
                "sensitivity": "{:.4f}".format,
                "specificity": "{:.4f}".format,
                "youden_j": "{:.4f}".format
            }
        )
    )

    print("\nSelected threshold:", selected_threshold)

    print(
        "Maximum Youden's J:",
        round(
            results_df.loc[
                best_index,
                "youden_j"
            ],
            4
        )
    )

    return (
        selected_threshold,
        results_df
    )


if __name__ == "__main__":
    select_threshold()