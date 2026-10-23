from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project root/data/wdbc.data
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "wdbc.data"


def load_and_prepare_data():
    columns = [
        "id", "diagnosis",

        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",

        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",

        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst"
    ]

    # Load the dataset
    data = pd.read_csv(DATA_PATH, header=None)
    data.columns = columns

    # Convert diagnosis:
    # B = 0 (Benign), M = 1 (Malignant)
    data["diagnosis"] = data["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    # ID is not used as a predictor
    data = data.drop(columns=["id"])

    # Separate predictors and target
    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"]

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return data, X, y, X_train, X_test, y_train, y_test