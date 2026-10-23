# Probabilistic Breast Cancer Classification

A statistical and probabilistic machine learning study for classifying breast tumors as benign or malignant using the Breast Cancer Wisconsin (Diagnostic) dataset.

The project combines statistical inference, multicollinearity analysis, probabilistic classification, logistic regression, probability evaluation, and threshold optimization into a reproducible Python workflow.

---

## Project Overview

Breast cancer classification is a binary prediction problem where the objective is to distinguish between:

- **Benign tumors**
- **Malignant tumors**

This project does not treat classification as only a machine learning problem. Statistical analysis is first used to investigate the predictor variables, their distributions, differences between diagnostic groups, and multicollinearity.

The resulting workflow combines statistical reasoning with probabilistic machine learning to evaluate both classification performance and the quality of predicted probabilities.

---

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset.

### Dataset characteristics

- Observations: **569**
- Predictor variables: **30**
- Target variable: **Diagnosis**
- Benign cases: **357**
- Malignant cases: **212**

The original diagnosis labels are:

- `B` → Benign
- `M` → Malignant

For modelling, these are encoded as:

- `0` → Benign
- `1` → Malignant

---

## Methodology

The project follows the workflow below:

```text
Raw Dataset
     │
     ▼
Data Preparation
     │
     ▼
Stratified 80/20 Train-Test Split
     │
     ▼
Statistical Analysis
 ┌───┼───────────────┐
 │   │               │
 ▼   ▼               ▼
Shapiro   Welch's    Pearson
Normality t-test     Correlation
 │                     │
 └─────────┬───────────┘
           ▼
          VIF
           │
           ▼
   Feature Reduction
           │
           ▼
      Model Training
      ┌────┴─────┐
      │          │
     GNB         Logistic
                Regression
      │          │
      └────┬─────┘
           ▼
      Model Evaluation
           │
           ▼
  5-Fold Stratified CV
           │
           ▼
 Threshold Optimization
    using Youden's J
           │
           ▼
   Locked Threshold = 0.30
           │
           ▼
 Final Test-Set Evaluation