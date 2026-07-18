# Preprocessing Utilities

Reusable preprocessing functions created during my machine learning journey.

These utilities focus on common data preparation tasks while following
good machine learning practices such as preventing data leakage.

---

## Available Functions

## `scale_numeric_features()`

Scales numerical features using `StandardScaler`.

The scaler is fitted only on the first DataFrame provided (usually the
training set) and then applied to the remaining DataFrames (validation/test).

This ensures that information from validation or test data is not used during
the preprocessing step.

### Features

- Automatically detects numerical columns
- Scales only `int` and `float` features
- Leaves categorical and boolean features unchanged
- Prevents data leakage by fitting only on training data
- Supports transforming multiple datasets using the same scaler

---

## Usage

```python
from preprocessing import scale_numeric_features

scaler = scale_numeric_features(
    X_train,
    X_validation,
    X_test
)
