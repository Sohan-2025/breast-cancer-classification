import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import load_and_prepare_data
from src.models import train_models
from src.threshold_selection import select_threshold


FIGURES_PATH = (
    Path(__file__).resolve().parent.parent / "figures"
)

FIGURES_PATH.mkdir(exist_ok=True)


def plot_class_distribution(y):
    """
    Plot the distribution of benign and malignant cases.
    """

    counts = y.value_counts().sort_index()

    labels = ["Benign", "Malignant"]

    plt.figure(figsize=(7, 5))

    plt.bar(labels, counts.values)

    plt.title("Breast Cancer Class Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of Cases")

    for i, value in enumerate(counts.values):
        plt.text(
            i,
            value + 5,
            str(value),
            ha="center"
        )

    plt.tight_layout()

    output_path = FIGURES_PATH / "class_distribution.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_correlation_heatmap(X_train):
    """
    Plot Pearson correlation matrix for training features.
    """

    correlation_matrix = X_train.corr()

    plt.figure(figsize=(14, 11))

    plt.imshow(
        correlation_matrix,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(label="Pearson Correlation")

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        fontsize=7
    )

    plt.title(
        "Pearson Correlation Matrix of Predictor Variables"
    )

    plt.tight_layout()

    output_path = FIGURES_PATH / "correlation_heatmap.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

def plot_vif(vif_df):
    """
    Plot Variance Inflation Factor values.
    """

    vif_plot = vif_df.sort_values(
        "VIF",
        ascending=True
    )

    plt.figure(figsize=(10, 10))

    plt.barh(
        vif_plot["feature"],
        vif_plot["VIF"]
    )

    plt.axvline(
        x=10,
        linestyle="--",
        linewidth=1.5,
        label="VIF = 10"
    )

    plt.xlabel("Variance Inflation Factor (VIF)")
    plt.ylabel("Feature")
    plt.title("Variance Inflation Factor of Predictor Variables")

    plt.legend()

    plt.tight_layout()

    output_path = FIGURES_PATH / "vif_plot.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

def plot_roc_curves():
    """
    Plot ROC curves for all three trained models.
    """

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

    # Get predicted probabilities
    gnb_prob = gnb.predict_proba(X_test)[:, 1]

    log_reg_prob = log_reg.predict_proba(X_test)[:, 1]

    reduced_prob = (
        log_reg_reduced.predict_proba(X_test_reduced)[:, 1]
    )

    # ROC curves
    fpr_gnb, tpr_gnb, _ = roc_curve(
        y_test,
        gnb_prob
    )

    fpr_log, tpr_log, _ = roc_curve(
        y_test,
        log_reg_prob
    )

    fpr_reduced, tpr_reduced, _ = roc_curve(
        y_test,
        reduced_prob
    )

    # ROC-AUC
    auc_gnb = roc_auc_score(
        y_test,
        gnb_prob
    )

    auc_log = roc_auc_score(
        y_test,
        log_reg_prob
    )

    auc_reduced = roc_auc_score(
        y_test,
        reduced_prob
    )

    # Plot
    plt.figure(figsize=(8, 7))

    plt.plot(
        fpr_gnb,
        tpr_gnb,
        label=f"Gaussian Naive Bayes (AUC = {auc_gnb:.4f})"
    )

    plt.plot(
        fpr_log,
        tpr_log,
        label=f"Full Logistic Regression (AUC = {auc_log:.4f})"
    )

    plt.plot(
        fpr_reduced,
        tpr_reduced,
        label=f"Reduced Logistic Regression (AUC = {auc_reduced:.4f})"
    )

    # Random classifier reference
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curves for Classification Models")

    plt.legend(loc="lower right")

    plt.tight_layout()

    output_path = FIGURES_PATH / "roc_curves.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

def plot_confusion_matrix():
    """
    Plot the final confusion matrix using the locked threshold of 0.30.
    """


    (
        data,
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_and_prepare_data()

    model = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.30).astype(int)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Benign", "Malignant"]
    )

    plt.yticks(
        [0, 1],
        ["Benign", "Malignant"]
    )

    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("Actual Diagnosis")

    plt.title(
        "Confusion Matrix at Selected Threshold (0.30)"
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    output_path = FIGURES_PATH / "confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

def plot_threshold_youden():
    """
    Plot threshold selection using Youden's J statistic.
    """


    selected_threshold, results_df = select_threshold()

    plt.figure(figsize=(8, 6))

    plt.plot(
        results_df["threshold"],
        results_df["youden_j"],
        marker="o"
    )

    plt.axvline(
        x=selected_threshold,
        linestyle="--",
        label=f"Selected threshold = {selected_threshold:.2f}"
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Youden's J")
    plt.title("Threshold Selection Using Youden's J Statistic")

    plt.legend()

    plt.tight_layout()

    output_path = FIGURES_PATH / "threshold_youden.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

def generate_visualizations():

    (
        data,
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_and_prepare_data()

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Class distribution
    plot_class_distribution(y)

    # Pearson correlation heatmap
    plot_correlation_heatmap(X_train)

    # Calculate VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_results = []

    for i, feature in enumerate(X_train.columns):

        vif = variance_inflation_factor(
            X_train.values,
            i
        )

        vif_results.append({
            "feature": feature,
            "VIF": vif
        })

    vif_df = pd.DataFrame(vif_results)

    # VIF plot
    plot_vif(vif_df)

    # ROC curves
    plot_roc_curves()

    # Final confusion matrix
    plot_confusion_matrix()

    # Threshold selection
    plot_threshold_youden()  

    print("\nAll visualizations generated successfully.")


if __name__ == "__main__":
    generate_visualizations()