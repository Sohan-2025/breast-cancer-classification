import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import load_and_prepare_data


def train_models():

    # Load the same train/test split used throughout the project
    data, X, y, X_train, X_test, y_train, y_test = load_and_prepare_data()

    # --------------------------------------------------
    # 1. Gaussian Naive Bayes
    # --------------------------------------------------

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # --------------------------------------------------
    # 2. Full Logistic Regression
    # --------------------------------------------------

    log_reg = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    log_reg.fit(X_train, y_train)

    # --------------------------------------------------
    # 3. Reduced Feature Logistic Regression
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

    log_reg_reduced = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    log_reg_reduced.fit(
        X_train_reduced,
        y_train
    )

    # --------------------------------------------------
    # Print basic results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    print("\nGaussian Naive Bayes")
    print("Training accuracy:",
          round(gnb.score(X_train, y_train), 4))
    print("Test accuracy:",
          round(gnb.score(X_test, y_test), 4))

    print("\nFull Logistic Regression")
    print("Training accuracy:",
          round(log_reg.score(X_train, y_train), 4))
    print("Test accuracy:",
          round(log_reg.score(X_test, y_test), 4))

    print("\nReduced Logistic Regression")
    print("Training accuracy:",
          round(log_reg_reduced.score(X_train_reduced, y_train), 4))
    print("Test accuracy:",
          round(log_reg_reduced.score(X_test_reduced, y_test), 4))

    print("\nReduced features:", len(reduced_features))

    return (
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
    )


if __name__ == "__main__":
    train_models()