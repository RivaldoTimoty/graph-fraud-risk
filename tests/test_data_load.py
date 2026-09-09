"""Sanity test Fase 0: shape, fraud rate, join, dan dtype reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.data.load import _normalize_identity_columns, load_raw
from src.data.reduce_memory import reduce_memory

CFG = load_config("data")
MERGED_PATH = resolve_path(CFG["paths"]["merged_train"])

requires_merged = pytest.mark.skipif(
    not MERGED_PATH.exists(), reason="run `make data` first to build merged.parquet"
)


@requires_merged
def test_row_count_matches_expected():
    df = pd.read_parquet(MERGED_PATH, columns=["TransactionID"])
    assert len(df) == CFG["expected"]["train_rows"]
    assert df["TransactionID"].is_unique


@requires_merged
def test_fraud_rate_within_tolerance():
    df = pd.read_parquet(MERGED_PATH, columns=["isFraud"])
    rate = df["isFraud"].mean()
    expected = CFG["expected"]
    assert rate == pytest.approx(expected["fraud_rate"], abs=expected["fraud_rate_tol"])


@requires_merged
def test_has_identity_is_binary_and_consistent():
    df = pd.read_parquet(MERGED_PATH, columns=["has_identity", "id_01"])
    assert set(df["has_identity"].unique()) <= {0, 1}
    # Baris tanpa identity record tidak boleh punya nilai id_01.
    assert df.loc[df["has_identity"] == 0, "id_01"].isna().all()
    assert 0 < df["has_identity"].mean() < 1


def test_identity_column_normalization():
    df = pd.DataFrame({"TransactionID": [1], "id-01": [0.0], "DeviceType": ["mobile"]})
    assert list(_normalize_identity_columns(df).columns) == ["TransactionID", "id_01", "DeviceType"]


def test_left_join_preserves_all_transactions(tmp_path):
    transaction = pd.DataFrame(
        {"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0], "TransactionAmt": [10.0, 20.0, 30.0]}
    )
    identity = pd.DataFrame({"TransactionID": [2], "id_01": [-5.0]})
    tx_path, id_path = tmp_path / "tx.csv", tmp_path / "id.csv"
    transaction.to_csv(tx_path, index=False)
    identity.to_csv(id_path, index=False)

    merged = load_raw(tx_path, id_path)

    assert len(merged) == 3
    assert merged["has_identity"].tolist() == [0, 1, 0]
    assert merged.loc[merged["TransactionID"] == 2, "id_01"].item() == -5.0


def test_reduce_memory_preserves_values_and_protects_key_columns():
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(3, dtype="int64"),
            "TransactionDT": np.array([86400, 86401, 86402], dtype="int64"),
            "isFraud": np.array([0, 1, 0], dtype="int64"),
            "count_col": np.array([1, 2, 300], dtype="int64"),
            "amt_col": np.array([1.5, 2.25, 3.75], dtype="float64"),
        }
    )
    out = reduce_memory(df)

    for col in ("TransactionID", "TransactionDT", "isFraud"):
        assert out[col].dtype == np.int64
    assert out["count_col"].dtype.itemsize < 8
    assert out["amt_col"].dtype == np.float32
    pd.testing.assert_frame_equal(out.astype("float64"), df.astype("float64"))
