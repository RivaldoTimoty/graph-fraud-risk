"""Test feature engineering - fokus pada anti-leakage.

Test paling penting di sini adalah `test_*_ignores_non_training_rows`: kalau
mengubah data val/test mengubah nilai fitur pada baris train, berarti statistik
ikut dihitung dari periode yang seharusnya tidak terlihat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate, ks_statistic, lift_at_k
from src.features.tabular import (
    FrequencyEncoder,
    GroupAggregator,
    LabelEncoder,
    amount_features,
)


@pytest.fixture
def toy():
    df = pd.DataFrame(
        {
            "card1": ["a", "a", "b", "a", "c", "c"],
            "cat": ["x", "x", "y", "x", "z", "z"],
            "TransactionAmt": [10.0, 20.0, 100.0, 30.0, 50.0, 70.0],
        }
    )
    train_mask = np.array([True, True, True, False, False, False])
    return df, train_mask


def test_frequency_encoder_counts_only_training_rows(toy):
    df, mask = toy
    out = FrequencyEncoder(["card1"]).fit(df, mask).transform(df)
    # Di train: a muncul 2x, b 1x. c hanya ada di val -> 0, bukan 3.
    assert out["freq_card1"].tolist() == [2.0, 2.0, 1.0, 2.0, 0.0, 0.0]


def test_frequency_encoder_ignores_non_training_rows(toy):
    """Mengubah baris non-training tidak boleh mengubah fitur baris training."""
    df, mask = toy
    baseline = FrequencyEncoder(["card1"]).fit(df, mask).transform(df)

    tampered = df.copy()
    tampered.loc[3:, "card1"] = "a"  # banjiri periode val dengan kategori 'a'
    after = FrequencyEncoder(["card1"]).fit(tampered, mask).transform(tampered)

    np.testing.assert_array_equal(
        baseline.loc[mask, "freq_card1"].to_numpy(), after.loc[mask, "freq_card1"].to_numpy()
    )


def test_group_aggregator_unseen_entity_is_nan_not_zero(toy):
    """Entitas baru harus NaN - 0 akan dibaca model sebagai 'nominal sangat kecil'."""
    df, mask = toy
    out = GroupAggregator("TransactionAmt", ["card1"], ["mean"]).fit(df, mask).transform(df)

    assert out.loc[0, "TransactionAmt_mean_by_card1"] == pytest.approx(15.0)  # (10+20)/2
    assert np.isnan(out.loc[4, "TransactionAmt_mean_by_card1"])  # 'c' tak terlihat
    assert np.isnan(out.loc[4, "TransactionAmt_ratio_to_mean_card1"])


def test_group_aggregator_ratio_is_relative_to_entity(toy):
    df, mask = toy
    out = GroupAggregator("TransactionAmt", ["card1"], ["mean"]).fit(df, mask).transform(df)
    # Baris 3 (val): amt 30 terhadap mean 'a' dari training = 15 -> rasio 2.0
    assert out.loc[3, "TransactionAmt_ratio_to_mean_card1"] == pytest.approx(2.0)


def test_group_aggregator_ignores_non_training_rows(toy):
    df, mask = toy
    baseline = GroupAggregator("TransactionAmt", ["card1"], ["mean"]).fit(df, mask).transform(df)

    tampered = df.copy()
    tampered.loc[3, "TransactionAmt"] = 99999.0
    after = (
        GroupAggregator("TransactionAmt", ["card1"], ["mean"])
        .fit(tampered, mask)
        .transform(tampered)
    )

    col = "TransactionAmt_mean_by_card1"
    np.testing.assert_array_equal(
        baseline.loc[mask, col].to_numpy(), after.loc[mask, col].to_numpy()
    )


def test_label_encoder_marks_unseen_as_negative_one(toy):
    df, mask = toy
    out = LabelEncoder(["cat"]).fit(df, mask).transform(df)
    assert out["le_cat"].tolist()[:4] == [0, 0, 1, 0]  # x->0, y->1
    assert out["le_cat"].tolist()[4:] == [-1, -1]  # z tak terlihat di training


def test_amount_features():
    df = pd.DataFrame({"TransactionAmt": [0.0, 59.99, 100.0]})
    out = amount_features(df)
    assert out["amt_log"].tolist() == pytest.approx(
        [0.0, np.log1p(59.99), np.log1p(100.0)], rel=1e-5
    )
    assert out["amt_decimal"].tolist() == pytest.approx([0.0, 0.99, 0.0], abs=1e-4)


def test_ks_statistic_bounds():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_lift_at_k_perfect_ranking():
    y = np.concatenate([np.ones(10), np.zeros(90)])
    score = np.concatenate([np.linspace(1.0, 0.9, 10), np.zeros(90)])
    m = lift_at_k(y, score, 0.10)

    assert m["n_reviewed"] == 10
    assert m["recall"] == pytest.approx(1.0)
    assert m["lift"] == pytest.approx(10.0)  # base rate 10%, precision 100%


def test_evaluate_reports_all_three_k_levels():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.035, 5000)
    score = rng.random(5000)
    out = evaluate(y, score)

    for tag in ("1pct", "5pct", "10pct"):
        assert f"recall_at_{tag}" in out
        assert f"lift_at_{tag}" in out
    assert 0.0 <= out["auc"] <= 1.0
