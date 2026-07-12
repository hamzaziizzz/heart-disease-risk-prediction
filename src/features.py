"""
features.py
============
Feature engineering pipeline for the Heart Disease classifier (Assignment
Task 2: "Feature Engineering & Model Development").

Theory / Design rationale
--------------------------
The Heart Disease UCI dataset has two categories of columns:

    * NUMERIC / continuous features:  age, trestbps, chol, thalach, oldpeak
      -> These need MEAN/MEDIAN imputation for missing values and
         STANDARDIZATION (zero mean, unit variance) because Logistic
         Regression is sensitive to feature scale (its coefficients are
         penalized via L2 regularisation, which is scale-dependent), and
         distance/gradient based optimisers converge faster on scaled data.

    * CATEGORICAL (nominal/ordinal but encoded as integers) features:
      sex, cp, fbs, restecg, exang, slope, ca, thal
      -> These are already integer-encoded in the raw data, but the integer
         codes do NOT represent a true ordinal relationship for several of
         them (e.g. 'cp' chest-pain-type 1-4 are distinct categories, not a
         scale). We therefore treat them as categorical and apply
         MODE (most-frequent) imputation + ONE-HOT encoding so that
         tree-based and linear models alike see them as unordered categories.

We wrap all of this in a single scikit-learn `ColumnTransformer` inside a
`Pipeline`, which is the SAME object used at both TRAINING time and
INFERENCE time (loaded from disk in the FastAPI service). This guarantees
there is no "train/serve skew" -- the exact same imputation statistics
(learned means/modes) and the exact same one-hot category mapping are
applied to new patient records at prediction time as were applied during
training.

Why not do this by hand with pandas .fillna()?
    A hand-written approach would need imputation VALUES to be memorised and
    reapplied manually at inference, which is error-prone and hard to
    version. `ColumnTransformer` inside a `Pipeline` serializes cleanly with
    joblib/pickle and can be shipped as a single artifact -- exactly what the
    assignment's "Model Packaging & Reproducibility" task (Task 4) requires.
"""

from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column groupings. These lists are the single source of truth for which
# raw columns are treated as numeric vs categorical -- reused by the
# notebooks, training script, tests, and the FastAPI request schema.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]

CATEGORICAL_FEATURES: List[str] = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "target"


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Construct the (unfitted) preprocessing ColumnTransformer.

    Numeric branch:
        SimpleImputer(strategy="median") -> StandardScaler()
        Median is used (rather than mean) because clinical measurements like
        cholesterol and resting blood pressure can have skewed distributions
        / outliers, and the median is more robust to those outliers than the
        mean.

    Categorical branch:
        SimpleImputer(strategy="most_frequent") -> OneHotEncoder(handle_unknown="ignore")
        `handle_unknown="ignore"` is critical for production robustness: if
        the live API ever receives a category value not seen during training
        (e.g. a data-entry glitch), the pipeline will encode it as an
        all-zero vector rather than raising an exception and crashing the
        /predict endpoint.

    Returns:
        ColumnTransformer: unfitted transformer; call `.fit()` /
            `.fit_transform()` on training data, or wrap inside a full
            `Pipeline` with an estimator (see `build_full_pipeline`).
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def build_full_pipeline(estimator) -> Pipeline:
    """
    Wrap a scikit-learn compatible estimator (e.g. LogisticRegression,
    RandomForestClassifier, XGBClassifier) together with the preprocessing
    ColumnTransformer into ONE end-to-end `Pipeline` object.

    Why this matters for reproducibility (Task 4):
        Fitting/predicting through a single `Pipeline` object means:
          1. `pipeline.fit(X_train, y_train)` fits BOTH the preprocessor and
             the estimator in one call, on the training data only (no data
             leakage from the test/validation split into the imputer's
             learned medians/modes).
          2. `pipeline.predict(X_new)` and `pipeline.predict_proba(X_new)`
             automatically apply the identical preprocessing before scoring.
          3. `joblib.dump(pipeline, "model.joblib")` serializes the WHOLE
             thing (preprocessing + model) as one reusable artifact -- the
             FastAPI service only needs to load this one file.

    Args:
        estimator: Any unfitted scikit-learn-compatible classifier exposing
            `.fit(X, y)`, `.predict(X)`, and ideally `.predict_proba(X)`.

    Returns:
        sklearn.pipeline.Pipeline: two-step pipeline: ("preprocessor", ...)
            -> ("classifier", estimator).
    """
    preprocessor = build_preprocessing_pipeline()
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])


def split_features_target(df, target_column: str = TARGET_COLUMN) -> Tuple:
    """
    Split a cleaned dataframe into the feature matrix X and target vector y.

    Args:
        df (pd.DataFrame): Cleaned dataframe (output of
            `download_data.clean_and_process`), containing all of
            ALL_FEATURES plus the target column.
        target_column (str): Name of the target column (default "target").

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) where X has exactly the
            columns in ALL_FEATURES (in that order) and y is the binary
            target.
    """
    X = df[ALL_FEATURES].copy()
    y = df[target_column].copy()
    return X, y
