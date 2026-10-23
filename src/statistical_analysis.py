import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import shapiro, ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.data_preprocessing import load_and_prepare_data


RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "statistical_analysis.txt"
)


def run_statistical_analysis():

    # Load the same training data used by the project
    data, X, y, X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Combine training features and labels for statistical analysis
    train_data = X_train.copy()
    train_data["diagnosis"] = y_train.values

    # Full dataset for Welch's t-test
    full_data = X.copy()
    full_data["diagnosis"] = y.values

    # --------------------------------------------------
    # 1. Shapiro-Wilk Normality Test
    # --------------------------------------------------

    normality_results = []

    for feature in X_train.columns:

        benign_values = train_data.loc[
            train_data["diagnosis"] == 0, feature
        ]

        malignant_values = train_data.loc[
            train_data["diagnosis"] == 1, feature
        ]

        benign_stat, benign_p = shapiro(benign_values)
        malignant_stat, malignant_p = shapiro(malignant_values)

        normality_results.append({
            "feature": feature,
            "shapiro_p_benign": benign_p,
            "shapiro_p_malignant": malignant_p
        })

    normality_df = pd.DataFrame(normality_results)

    # --------------------------------------------------
    # 2. Welch's Two-Sample t-Test
    # --------------------------------------------------

    ttest_results = []

    for feature in X.columns:

        benign_values = full_data.loc[
            full_data["diagnosis"] == 0, feature
        ]

        malignant_values = full_data.loc[
            full_data["diagnosis"] == 1, feature
        ]

        t_stat, p_value = ttest_ind(
            benign_values,
            malignant_values,
            equal_var=False
        )

        ttest_results.append({
            "feature": feature,
            "t_statistic": t_stat,
            "p_value": p_value
        })

    ttest_df = pd.DataFrame(ttest_results)

    # --------------------------------------------------
    # 3. Pearson Correlation
    # --------------------------------------------------

    correlation_matrix = X_train.corr()

    correlation_pairs = []

    features = correlation_matrix.columns

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            correlation = correlation_matrix.iloc[i, j]

            if abs(correlation) > 0.95:

                correlation_pairs.append({
                    "feature_1": features[i],
                    "feature_2": features[j],
                    "correlation": correlation
                })

    correlation_df = pd.DataFrame(correlation_pairs)

    correlation_df["absolute_correlation"] = (
        correlation_df["correlation"].abs()
    )

    correlation_df = correlation_df.sort_values(
        "absolute_correlation",
        ascending=False
    )

    # --------------------------------------------------
    # 4. Variance Inflation Factor (VIF)
    # --------------------------------------------------

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

    vif_df = vif_df.sort_values(
        "VIF",
        ascending=False
    )

    # --------------------------------------------------
    # 5. Feature Reduction
    # --------------------------------------------------

    removed_features = [
        "perimeter_mean",
        "area_mean",
        "perimeter_se",
        "area_se",
        "radius_worst",
        "perimeter_worst",
        "area_worst"
    ]

    reduced_features = [
        feature
        for feature in X_train.columns
        if feature not in removed_features
    ]

    X_train_reduced = X_train[reduced_features]
    X_test_reduced = X_test[reduced_features]

    # --------------------------------------------------
    # Save Results
    # --------------------------------------------------

    significant = ttest_df[ttest_df["p_value"] < 0.05]

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:

        file.write("=" * 70 + "\n")
        file.write("STATISTICAL ANALYSIS RESULTS\n")
        file.write("=" * 70 + "\n\n")

        file.write("1. SHAPIRO-WILK NORMALITY TEST\n")
        file.write("-" * 70 + "\n")
        file.write(
            normality_df.to_string(index=False)
        )
        file.write("\n\n")

        file.write("2. WELCH'S TWO-SAMPLE t-TEST\n")
        file.write("-" * 70 + "\n")
        file.write(
            ttest_df.to_string(index=False)
        )
        file.write("\n\n")

        file.write(
            f"Significant features (p < 0.05): "
            f"{len(significant)} out of {len(ttest_df)}\n\n"
        )

        file.write("3. HIGHLY CORRELATED FEATURE PAIRS\n")
        file.write("-" * 70 + "\n")
        file.write(
            "Criterion: |r| > 0.95\n\n"
        )
        file.write(
            correlation_df.to_string(index=False)
        )
        file.write("\n\n")

        file.write("4. VARIANCE INFLATION FACTOR\n")
        file.write("-" * 70 + "\n")
        file.write(
            vif_df.to_string(index=False)
        )
        file.write("\n\n")

        file.write("5. FEATURE REDUCTION\n")
        file.write("-" * 70 + "\n")
        file.write(
            f"Original number of features: "
            f"{len(X_train.columns)}\n"
        )
        file.write(
            f"Removed features: "
            f"{', '.join(removed_features)}\n"
        )
        file.write(
            f"Remaining features: "
            f"{len(reduced_features)}\n"
        )

    # --------------------------------------------------
    # Print Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    print("\n1. Shapiro-Wilk Normality Test")
    print(normality_df.to_string(index=False))

    print("\n2. Welch's Two-Sample t-Test")
    print(ttest_df.to_string(index=False))

    print("\nSignificant features (p < 0.05):")
    print(len(significant), "out of", len(ttest_df))

    print("\n3. Highly Correlated Feature Pairs (|r| > 0.95)")
    print(correlation_df.to_string(index=False))

    print("\n4. Variance Inflation Factor")
    print(vif_df.to_string(index=False))

    print("\n5. Feature Reduction")
    print("Original number of features:", len(X_train.columns))
    print("Removed features:", removed_features)
    print("Remaining features:", len(reduced_features))

    print(f"\nResults saved to: {RESULTS_PATH}")

    return (
        normality_df,
        ttest_df,
        correlation_df,
        vif_df,
        X_train_reduced,
        X_test_reduced
    )


if __name__ == "__main__":
    run_statistical_analysis()