"""
tests/test_data_processing.py
==============================
Pytest unit tests for src/download_data.py (Assignment Task 5: "Write unit
tests for data processing ... code").

What we test and why
----------------------
We do NOT hit the network in these tests (no `download_raw_data` call) --
unit tests must be fast, deterministic, and runnable in an isolated CI
runner that may have restricted network egress. Instead we build a small,
in-memory synthetic raw file that mimics the real UCI format (including a
"?" missing value and a multi-class target), and verify that
`clean_and_process` transforms it exactly as documented.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.download_data import COLUMN_NAMES, clean_and_process, save_processed  # noqa: E402


@pytest.fixture
def synthetic_raw_file(tmp_path):
    """
    Create a small synthetic raw ".data" file (5 rows, UCI format: no
    header, comma-separated, "?" for missing values, multi-class target)
    in a pytest-managed temporary directory.

    Returns:
        str: path to the synthetic raw file.
    """
    rows = [
        "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0",   # target 0 -> no disease
        "67,1,4,160,286,0,2,108,1,1.5,2,3,3,2",   # target 2 -> disease (severity 2)
        "41,0,2,130,204,0,2,172,0,1.4,1,?,3,0",   # missing 'ca' ('?')
        "56,1,3,120,236,0,0,178,0,0.8,1,0,?,0",   # missing 'thal' ('?')
        "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0",   # exact duplicate of row 1
    ]
    raw_path = tmp_path / "synthetic.data"
    raw_path.write_text("\n".join(rows))
    return str(raw_path)


def test_clean_and_process_column_names(synthetic_raw_file):
    """The cleaned dataframe must have exactly the documented 14 columns, in order."""
    df = clean_and_process(synthetic_raw_file)
    assert list(df.columns) == COLUMN_NAMES


def test_clean_and_process_missing_values_become_nan(synthetic_raw_file):
    """
    Raw '?' tokens must be converted to real NaN (not left as string '?',
    which would silently break every downstream numeric operation).
    """
    df = clean_and_process(synthetic_raw_file)
    # Row index 2 (third row) had '?' in the 'ca' column.
    assert df["ca"].isna().sum() == 1
    # Row index 3 (fourth row) had '?' in the 'thal' column.
    assert df["thal"].isna().sum() == 1


def test_clean_and_process_target_is_binarized(synthetic_raw_file):
    """
    The multi-class raw target (0-4) must be collapsed to binary {0, 1}:
    0 stays 0 (no disease); any of 1/2/3/4 become 1 (disease present).
    """
    df = clean_and_process(synthetic_raw_file)
    assert set(df["target"].unique()).issubset({0, 1})
    # Second synthetic row had raw target=2 (severity 2) -> must map to 1.
    assert df.loc[df["target"] == 1].shape[0] >= 1


def test_clean_and_process_drops_exact_duplicates(synthetic_raw_file):
    """
    The synthetic fixture deliberately contains one exact duplicate row
    (row 1 and row 5 are identical) -- it must be dropped.
    """
    df = clean_and_process(synthetic_raw_file)
    # 5 raw rows, 1 exact duplicate -> 4 rows remain.
    assert len(df) == 4


def test_clean_and_process_all_columns_numeric(synthetic_raw_file):
    """Every column (post-cleaning) must be numeric -- required for sklearn."""
    df = clean_and_process(synthetic_raw_file)
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"


def test_save_processed_writes_csv_and_is_readable(tmp_path, synthetic_raw_file):
    """
    save_processed() must write a CSV that, when re-read, reproduces the
    same shape and dtypes as the in-memory dataframe (round-trip test).
    """
    df = clean_and_process(synthetic_raw_file)
    dest = tmp_path / "processed" / "clean.csv"
    save_processed(df, str(dest))

    assert dest.exists()
    reloaded = pd.read_csv(dest)
    assert reloaded.shape == df.shape
    assert list(reloaded.columns) == list(df.columns)
