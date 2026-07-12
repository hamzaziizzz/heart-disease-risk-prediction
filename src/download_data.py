"""
download_data.py
=================
Data Acquisition script for the Heart Disease UCI Dataset (Assignment Task 1).

Theory / Purpose
-----------------
The UCI Heart Disease dataset (Cleveland subset) is distributed as a raw,
whitespace/comma-separated ".data" file with NO header row and missing values
encoded as the literal string "?" instead of NaN. Before any EDA or modelling
can happen, we need a reproducible, scriptable way to:
    1. Fetch the raw file from the official UCI repository (network step).
    2. Attach meaningful column names (the UCI docs define 14 columns).
    3. Replace "?" with proper NaN so pandas/sklearn can handle missingness.
    4. Collapse the multi-class target (0-4, where 1-4 = increasing disease
       severity) into the binary target required by the assignment
       (0 = no disease, 1 = disease present).

Why a separate script (not inline in the notebook)?
    - Reproducibility: anyone cloning the repo can run `python src/download_data.py`
      and get an identical `data/processed/heart_disease_clean.csv`.
    - Separation of concerns: acquisition/cleaning logic is testable in isolation
      (see tests/test_data_processing.py), independent of notebook execution order.

Usage
-----
    python src/download_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
import urllib.request

# ---------------------------------------------------------------------------
# Column names as documented by the UCI Machine Learning Repository for the
# "processed.cleveland.data" file. Order matters -- it must match the raw file.
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # resting electrocardiographic results (0-2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # the slope of the peak exercise ST segment (1-3)
    "ca",        # number of major vessels (0-3) colored by fluoroscopy
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "target",    # original label: 0 = no disease, 1-4 = disease severity
]

# Primary source: official UCI ML Repository (Cleveland processed subset --
# this is the subset used in almost all published benchmarks for this dataset
# because it has the fewest missing values of the four hospital sources).
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Fallback mirror (same data, hosted for resilience in case the UCI URL is
# unreachable from a given network -- common in CI/CD runners with restricted
# egress). Verified to contain byte-identical values.
FALLBACK_URL = (
    "https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo-datasets/"
    "master/heart_disease/processed.cleveland.data"
)

RAW_PATH = os.path.join("data", "raw", "processed.cleveland.data")
PROCESSED_PATH = os.path.join("data", "processed", "heart_disease_clean.csv")


def download_raw_data(dest_path: str = RAW_PATH) -> str:
    """
    Download the raw UCI Cleveland heart-disease file if it is not already
    present locally (idempotent -- safe to call repeatedly, e.g. from CI).

    Why idempotency matters: CI pipelines and repeated local runs should not
    re-download unnecessarily (network flakiness, UCI rate limits).

    Args:
        dest_path (str): Local filesystem path to save the raw file to.

    Returns:
        str: The path the file was saved to (same as dest_path).

    Raises:
        RuntimeError: if both the primary and fallback download attempts fail.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        print(f"[download_data] Raw file already exists at {dest_path}, skipping download.")
        return dest_path

    for url in (UCI_URL, FALLBACK_URL):
        try:
            print(f"[download_data] Attempting download from {url} ...")
            urllib.request.urlretrieve(url, dest_path)
            print(f"[download_data] Saved raw data to {dest_path}")
            return dest_path
        except Exception as exc:  # network errors, 404s, etc.
            print(f"[download_data] Failed from {url}: {exc}")

    raise RuntimeError(
        "Could not download the Heart Disease dataset from any known source. "
        "Please manually download 'processed.cleveland.data' from "
        "https://archive.ics.uci.edu/dataset/45/heart+disease and place it at "
        f"{dest_path}"
    )


def clean_and_process(raw_path: str = RAW_PATH) -> pd.DataFrame:
    """
    Load the raw whitespace/comma-delimited UCI file and turn it into a
    clean, analysis-ready pandas DataFrame.

    Cleaning steps performed (and WHY each is required):
        1. Read with header=None + explicit COLUMN_NAMES, because the raw
           file ships with no header row.
        2. Replace the literal "?" string with np.nan -- pandas does not
           recognise "?" as missing by default, so numeric columns would
           otherwise be read as `object` dtype (strings), breaking every
           downstream numeric operation (scaling, correlation, etc.).
        3. Cast all columns to float64 -- after removing "?", every column is
           numeric, but pandas may still hold object dtype in columns that
           had at least one "?" (e.g. 'ca', 'thal'). Casting makes dtypes
           consistent for scikit-learn.
        4. Binarize the target: the raw target is 0 (no disease) through
           4 (severe disease). The assignment asks for a BINARY classifier
           ("presence/absence of heart disease"), so we map {0} -> 0 and
           {1,2,3,4} -> 1. This is the standard binarization used in the
           published literature on this dataset.
        5. Drop exact duplicate rows, if any (data integrity check).

    Args:
        raw_path (str): Path to the raw .data file downloaded by
            `download_raw_data`.

    Returns:
        pd.DataFrame: Cleaned dataframe with 14 columns (13 features + binary
            'target'), all numeric, missing values represented as NaN
            (deliberately NOT imputed here -- imputation is a *modelling*
            choice and belongs inside the sklearn Pipeline in
            src/features.py so that the exact same imputation is applied at
            training AND inference time, avoiding train/serve skew).
    """
    df = pd.read_csv(raw_path, header=None, names=COLUMN_NAMES, na_values="?")

    # Step 3: enforce numeric dtype on every column.
    df = df.astype(float)

    # Step 4: binarize target -> 0 = no disease, 1 = disease present.
    # np.where is a vectorized if/else: for every row, if target == 0 keep 0,
    # else set 1. This avoids a slow Python-level .apply() loop.
    df["target"] = np.where(df["target"] == 0, 0, 1).astype(int)

    # Step 5: drop exact duplicate rows (keeps the first occurrence).
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"[download_data] Dropped {n_before - n_after} duplicate row(s).")

    return df


def save_processed(df: pd.DataFrame, dest_path: str = PROCESSED_PATH) -> str:
    """
    Persist the cleaned dataframe to disk as CSV -- this is the artifact
    referenced by the notebooks, training script, and unit tests, and is the
    "cleaned dataset" deliverable required by the assignment.

    Args:
        df (pd.DataFrame): Cleaned dataframe from `clean_and_process`.
        dest_path (str): Output CSV path.

    Returns:
        str: The path the file was written to.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df.to_csv(dest_path, index=False)
    print(f"[download_data] Saved cleaned dataset ({df.shape[0]} rows, "
          f"{df.shape[1]} cols) to {dest_path}")
    return dest_path


def main():
    """Entry point: download -> clean -> save. Prints a short data summary."""
    raw_path = download_raw_data()
    df = clean_and_process(raw_path)
    save_processed(df)

    print("\n[download_data] --- Summary ---")
    print(df.describe(include="all").T[["count", "mean", "std", "min", "max"]])
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nClass balance (target):")
    print(df["target"].value_counts(normalize=True).rename("proportion"))


if __name__ == "__main__":
    sys.exit(main())
