# Model Tuning Utilities

Reusable functions for experimenting with machine learning model
hyperparameters.

These utilities were created to automate repetitive model training and
evaluation workflows.

---

## Available Functions

## `tune_xgb_hyperparameters()`

A custom sequential hyperparameter tuning function for XGBoost models.

The function tests different values for selected hyperparameters and chooses
the value that produces the highest validation accuracy.

---

## Features

- Supports arbitrary XGBoost hyperparameters
- Automatically trains multiple models
- Tracks training and validation accuracy
- Returns the best performing parameter value
- Stores results for visualization and comparison

---

## Usage

```python
from hyperparameter_tuning import tune_xgb_hyperparameters

best_params, train_scores, validation_scores = tune_xgb_hyperparameters(
    X_train,
    X_validation,
    y_train,
    y_validation,
    max_depth=[2, 4, 6, 8],
    subsample=[0.5, 0.7, 0.9, 1.0]
)
