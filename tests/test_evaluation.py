"""Test modul evaluasi Fase 5: scorecard, business, stability, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.business import baseline_cost, cost_curve, optimal_threshold
from src.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    fit_isotonic,
    reliability_curve,
)
from src.evaluation.scorecard import (
    gain_table,
    probability_to_score,
    score_bands,
    score_factor,
)
from src.evaluation.stability import (
    characteristic_stability,
    population_stability_index,
    psi_verdict,
)

SCORECARD = dict(pdo=20, anchor_score=600, anchor_odds=50, score_min=300, score_max=850)


def test_score_is_monotonically_decreasing_in_probability():
    """Probabilitas fraud naik -> skor turun. Konvensi credit scoring."""
    probs = np.array([0.001, 0.01, 0.1, 0.5, 0.9])
    scores = probability_to_score(probs, **SCORECARD)
    assert np.all(np.diff(scores) < 0)


def test_pdo_doubles_odds_exactly():
    """Odds non-fraud berlipat dua harus menaikkan skor tepat PDO poin."""
    p1 = 0.02
    odds1 = (1 - p1) / p1
    p2 = 1.0 / (1.0 + 2 * odds1)  # odds_good dua kali lipat -> risiko lebih rendah

    s1, s2 = probability_to_score(np.array([p1, p2]), **SCORECARD)
    assert s2 - s1 == pytest.approx(20.0, abs=0.01)


def test_anchor_point_is_exact():
    """Odds acuan 1:50 harus menghasilkan tepat skor acuan 600."""
    p = 1.0 / 51.0  # odds_good = 50
    score = probability_to_score(np.array([p]), **SCORECARD)[0]
    assert score == pytest.approx(600.0, abs=0.01)


def test_score_is_clipped_to_range():
    extreme = np.array([1e-12, 1.0 - 1e-12])
    scores = probability_to_score(extreme, **SCORECARD)
    assert scores.min() >= 300 and scores.max() <= 850


def test_score_factor_matches_formula():
    assert score_factor(20) == pytest.approx(20 / np.log(2))


def test_score_bands_cumulative_recall_reaches_one():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, 2000)
    scores = rng.normal(600, 50, 2000)
    table = score_bands(scores, y, n_bands=10)

    assert table["cum_pct_fraud"].iloc[-1] == pytest.approx(1.0)
    assert table["cum_pct_population"].iloc[-1] == pytest.approx(1.0)


def test_gain_table_perfect_model():
    y = np.concatenate([np.ones(100), np.zeros(900)])
    score = np.concatenate([np.ones(100), np.zeros(900)])
    gains = gain_table(y, score, n_points=10)

    at_10pct = gains.iloc[0]
    assert at_10pct["review_rate"] == pytest.approx(0.1)
    assert at_10pct["recall"] == pytest.approx(1.0)
    assert at_10pct["lift"] == pytest.approx(10.0)


def test_cost_curve_uses_per_transaction_amount():
    """Biaya FN harus memakai nilai transaksi masing-masing, bukan rata-rata."""
    y = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.1, 0.8, 0.2])
    amounts = np.array([1000.0, 10.0, 50.0, 50.0])

    curve = cost_curve(y, score, amounts, cost_fp=5.0, n_thresholds=5)

    # Threshold terendah: semua ter-flag, tidak ada fraud yang lolos.
    assert curve.iloc[0]["fn_cost"] == pytest.approx(0.0)
    # Threshold tertinggi (0.9): hanya transaksi 0 ter-flag, fraud kedua lolos.
    # Biayanya 10.0 — nilai transaksi ITU, bukan rata-rata fraud (505.0).
    assert curve.iloc[-1]["fn_cost"] == pytest.approx(10.0)

    # Di threshold 0.5: fraud senilai 1000 tertangkap, fraud senilai 10 lolos.
    mid = cost_curve(y, score, amounts, cost_fp=5.0, n_thresholds=5)
    assert mid[mid["threshold"] <= 0.5]["fn_cost"].min() == pytest.approx(0.0)


def test_optimal_threshold_beats_extremes():
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.binomial(1, 0.04, n)
    score = np.clip(y * 0.5 + rng.normal(0.2, 0.15, n), 0, 1)
    amounts = rng.lognormal(4, 1, n)

    curve = cost_curve(y, score, amounts, cost_fp=5.0, n_thresholds=50)
    best = optimal_threshold(curve)
    assert best["total_cost"] <= curve["total_cost"].max()
    assert 0.0 <= best["review_rate"] <= 1.0


def test_baseline_cost_is_total_fraud_value():
    y = np.array([1, 0, 1])
    amounts = np.array([100.0, 999.0, 50.0])
    assert baseline_cost(y, amounts) == pytest.approx(150.0)


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(2)
    values = rng.normal(0, 1, 5000)
    assert population_stability_index(values, values.copy(), n_bins=10) == pytest.approx(
        0.0, abs=1e-6
    )


def test_psi_grows_with_shift():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 10000)
    small = population_stability_index(base, rng.normal(0.1, 1, 10000))
    large = population_stability_index(base, rng.normal(1.5, 1, 10000))
    assert small < large
    assert large > 0.25


def test_psi_verdict_thresholds():
    assert psi_verdict(0.05) == "stabil"
    assert psi_verdict(0.15) == "perlu perhatian"
    assert psi_verdict(0.30) == "bermasalah"


def test_csi_identifies_shifted_feature():
    rng = np.random.default_rng(4)
    expected = pd.DataFrame({"stable": rng.normal(0, 1, 5000), "shifted": rng.normal(0, 1, 5000)})
    actual = pd.DataFrame({"stable": rng.normal(0, 1, 5000), "shifted": rng.normal(2, 1, 5000)})

    csi = characteristic_stability(expected, actual, ["stable", "shifted"])
    assert csi.iloc[0]["feature"] == "shifted"
    assert csi.iloc[0]["csi"] > csi.iloc[1]["csi"]


def test_isotonic_preserves_ranking():
    """Kalibrasi monoton tidak boleh mengubah AUC."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(5)
    n = 4000
    y = rng.binomial(1, 0.05, n)
    prob = np.clip(y * 0.4 + rng.normal(0.15, 0.12, n), 1e-6, 1 - 1e-6)

    calibrator = fit_isotonic(y[: n // 2], prob[: n // 2])
    calibrated = calibrator.predict(prob[n // 2 :])

    before = roc_auc_score(y[n // 2 :], prob[n // 2 :])
    after = roc_auc_score(y[n // 2 :], calibrated)
    assert after == pytest.approx(before, abs=0.02)


def test_isotonic_improves_calibration_of_biased_scores():
    rng = np.random.default_rng(6)
    n = 6000
    y = rng.binomial(1, 0.05, n)
    # Skor sengaja over-confident: dikalikan 3 dari probabilitas sebenarnya.
    prob = np.clip(np.where(y == 1, 0.6, 0.12) + rng.normal(0, 0.05, n), 1e-6, 1 - 1e-6)

    half = n // 2
    calibrator = fit_isotonic(y[:half], prob[:half])
    calibrated = calibrator.predict(prob[half:])

    ece_before = expected_calibration_error(y[half:], prob[half:])
    ece_after = expected_calibration_error(y[half:], calibrated)
    assert ece_after < ece_before


def test_brier_score_perfect_prediction_is_zero():
    y = np.array([0, 1, 1, 0])
    assert brier_score(y, y.astype(float)) == pytest.approx(0.0)


def test_reliability_curve_diagonal_for_calibrated_model():
    rng = np.random.default_rng(7)
    prob = rng.uniform(0, 1, 20000)
    y = rng.binomial(1, prob)
    curve = reliability_curve(y, prob, n_bins=10)

    diff = (curve["mean_predicted"] - curve["actual_rate"]).abs()
    assert diff.max() < 0.05
