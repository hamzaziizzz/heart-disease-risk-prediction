"""
tests/test_features.py
========================
Pytest unit tests for src/features.py -- the preprocessing pipeline
construction (Assignment Task 5: unit tests for "data processing and model
code"). These tests verify the pipeline correctly imputes, scales, and
encodes a small synthetic dataset, and that it produces a fully numeric
output matrix suitable for any scikit-learn estimator.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import (  # noqa: E402
    ALL_FEATURES,
    NUMERIC_FEATURES,
    build_full_pipeline,
    build_preprocessing_pipeline,
    split_features_target,
)


@pytest.fixture
def synthetic_df():
    """
    Small synthetic dataframe with all required columns, including a couple
    of missing values (NaN) to exercise the imputers, and a binary target.
    """
    data = {
        "age": [63, 67, 41, 56, np.nan],
        "trestbps": [145, 160, 130, 120, 140],
        "chol": [233, 286, 204, 236, 250],
        "thalach": [150, 108, 172, 178, 160],
        "oldpeak": [2.3, 1.5, 1.4, 0.8, np.nan],
        "sex": [1, 1, 0, 1, 0],
        "cp": [1, 4, 2, 3, 1],
        "fbs": [1, 0, 0, 0, 1],
        "restecg": [2, 2, 2, 0, 1],
        "exang": [0, 1, 0, 0, 1],
        "slope": [3, 2, 1, 1, 2],
        "ca": [0, 3, np.nan, 0, 1],
        "thal": [6, 3, 3, np.nan, 7],
        "target": [0, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)


def test_split_features_target_shapes(synthetic_df):
    """X must contain exactly ALL_FEATURES columns; y must be the target series."""
    X, y = split_features_target(synthetic_df)
    assert list(X.columns) == ALL_FEATURES
    assert len(y) == len(synthetic_df)
    assert y.name == "target"


def test_preprocessing_pipeline_handles_missing_values(synthetic_df):
    """
    The fitted ColumnTransformer's output must contain NO NaNs, even though
    the input had missing values in 'age', 'oldpeak', 'ca', and 'thal' --
    proving the SimpleImputer steps are correctly wired for both the
    numeric and categorical branches.
    """
    X, y = split_features_target(synthetic_df)
    preprocessor = build_preprocessing_pipeline()
    X_transformed = preprocessor.fit_transform(X)

    # ColumnTransformer with OneHotEncoder returns a numpy array (or sparse
    # matrix); convert to dense array to check for NaNs uniformly.
    X_dense = (
        np.asarray(X_transformed.todense())
        if hasattr(X_transformed, "todense")
        else np.asarray(X_transformed)
    )
    assert not np.isnan(X_dense).any(), "Transformed feature matrix contains NaNs after imputation"


def test_preprocessing_pipeline_scales_numeric_features(synthetic_df):
    """
    After StandardScaler, each numeric column's transformed values should
    have approximately zero mean (within floating point tolerance) --
    confirms scaling is actually being applied, not silently skipped.
    """
    X, y = split_features_target(synthetic_df)
    preprocessor = build_preprocessing_pipeline()
    X_transformed = preprocessor.fit_transform(X)
    X_dense = (
        np.asarray(X_transformed.todense())
        if hasattr(X_transformed, "todense")
        else np.asarray(X_transformed)
    )

    # The first len(NUMERIC_FEATURES) columns of the ColumnTransformer output
    # correspond to the numeric branch (order defined in build_preprocessing_pipeline).
    numeric_block = X_dense[:, : len(NUMERIC_FEATURES)]
    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-6)


def test_full_pipeline_fits_and_predicts(synthetic_df):
    """
    End-to-end smoke test: build_full_pipeline() wrapping a real estimator
    must fit without error on the synthetic data and produce predictions of
    the correct shape and binary class labels {0, 1}.
    """
    X, y = split_features_target(synthetic_df)
    pipeline = build_full_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})

    probas = pipeline.predict_proba(X)
    assert probas.shape == (len(X), 2)
    # Each row of predicted probabilities must sum to 1 (valid probability distribution).
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_full_pipeline_handles_unseen_category_at_inference(synthetic_df):
    """
    A category value not present during training (e.g. an unusual 'cp'
    code) must NOT crash inference -- OneHotEncoder(handle_unknown="ignore")
    should encode it as all-zeros instead of raising. This directly protects
    the production /predict endpoint from crashing on unexpected input.
    """
    X, y = split_features_target(synthetic_df)
    pipeline = build_full_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)

    X_new = X.iloc[[0]].copy()
    X_new["cp"] = 99  # category never seen during training

    # Should not raise.
    pred = pipeline.predict(X_new)
    assert pred.shape == (1,)
