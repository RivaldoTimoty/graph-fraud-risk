"""Test temporal split - bagian paling kritis Fase 1.

Kalau split bocor, semua hasil di fase berikutnya tidak sah.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.data.split import (
    DAY_ORIGIN,
    SECONDS_PER_DAY,
    add_time_features,
    temporal_split,
    transaction_day,
)

CFG = load_config("split")
PARAMS = CFG["split"]
MASKS_PATH = resolve_path(CFG["paths"]["masks"])

requires_masks = pytest.mark.skipif(
    not MASKS_PATH.exists(), reason="run `make split` first to build split_masks.parquet"
)


def _synthetic(n_days: int = 200, per_day: int = 5) -> pd.DataFrame:
    dt = [
        DAY_ORIGIN + d * SECONDS_PER_DAY + i * 1000 for d in range(n_days) for i in range(per_day)
    ]
    return pd.DataFrame({"TransactionID": range(len(dt)), "TransactionDT": dt})


def test_time_features_are_consistent():
    df = add_time_features(_synthetic(n_days=10))
    assert df["day"].min() == 0 and df["day"].max() == 9
    assert (df["week"] == df["day"] // 7).all()
    assert (df["dayofweek"] == df["day"] % 7).all()
    assert df["hour"].between(0, 23).all()


def test_splits_are_disjoint_and_cover_everything_except_gap():
    df = _synthetic()
    m = temporal_split(df, train_end_day=118, gap_days=7, val_end_day=150)

    stacked = np.vstack([m.train, m.gap, m.val, m.test])
    assert (stacked.sum(axis=0) == 1).all(), "setiap baris harus masuk tepat satu split"
    assert m.train.sum() + m.gap.sum() + m.val.sum() + m.test.sum() == len(df)


def test_temporal_ordering_is_strict():
    df = _synthetic()
    m = temporal_split(df, train_end_day=118, gap_days=7, val_end_day=150)
    dt = df["TransactionDT"]

    assert dt[m.train].max() < dt[m.val].min()
    assert dt[m.val].max() < dt[m.test].min()
    assert dt[m.train].max() < dt[m.gap].min() < dt[m.gap].max() < dt[m.val].min()


def test_no_day_is_split_across_partitions():
    """Batas harus jatuh di batas hari, bukan di tengah hari."""
    df = _synthetic()
    day = transaction_day(df)
    m = temporal_split(df, train_end_day=118, gap_days=7, val_end_day=150)

    for name in ("train", "gap", "val", "test"):
        mask = getattr(m, name)
        days_in = set(day[mask].unique())
        days_out = set(day[~mask].unique())
        assert not (days_in & days_out), f"hari terbelah antara {name} dan sisanya"


def test_gap_boundaries_match_config():
    df = _synthetic()
    day = transaction_day(df)
    m = temporal_split(df, PARAMS["train_end_day"], PARAMS["gap_days"], PARAMS["val_end_day"])

    assert day[m.train].max() == PARAMS["train_end_day"]
    assert day[m.gap].min() == PARAMS["train_end_day"] + 1
    assert day[m.gap].max() == PARAMS["train_end_day"] + PARAMS["gap_days"]
    assert day[m.val].min() == PARAMS["train_end_day"] + PARAMS["gap_days"] + 1
    assert day[m.val].max() == PARAMS["val_end_day"]
    assert day[m.test].min() == PARAMS["val_end_day"] + 1


def test_invalid_boundaries_raise():
    df = _synthetic()
    with pytest.raises(ValueError):
        temporal_split(df, train_end_day=150, gap_days=7, val_end_day=120)


@requires_masks
def test_saved_masks_match_recomputed_split():
    """Mask tersimpan harus identik dengan hasil hitung ulang - reproducibility."""
    data_cfg = load_config("data")
    df = pd.read_parquet(
        resolve_path(data_cfg["paths"]["merged_train"]), columns=["TransactionID", "TransactionDT"]
    )
    saved = pd.read_parquet(MASKS_PATH)
    m = temporal_split(df, PARAMS["train_end_day"], PARAMS["gap_days"], PARAMS["val_end_day"])

    assert saved["TransactionID"].tolist() == df["TransactionID"].tolist()
    np.testing.assert_array_equal(saved["is_train"].to_numpy(), m.train)
    np.testing.assert_array_equal(saved["is_val"].to_numpy(), m.val)
    np.testing.assert_array_equal(saved["is_test"].to_numpy(), m.test)


@requires_masks
def test_saved_masks_are_mutually_exclusive():
    saved = pd.read_parquet(MASKS_PATH)
    cols = saved[["is_train", "is_val", "is_test", "is_gap"]].to_numpy()
    assert (cols.sum(axis=1) == 1).all()
