"""PSI dan CSI — metrik monitoring wajib di credit scoring.

PSI (Population Stability Index) mengukur pergeseran distribusi SKOR antar periode.
CSI (Characteristic Stability Index) memakai rumus yang sama pada FITUR individual,
untuk menemukan fitur mana yang menyebabkan pergeseran itu.

Ambang industri: < 0,10 stabil | 0,10-0,25 perlu perhatian | > 0,25 bermasalah.

KETERBATASAN PENTING (dibuktikan di Fase 4): PSI hanya melihat NILAI fitur. Fitur
berbasis label di project ini punya nilai yang stabil (drift -2,8%) sementara
kekuatan buktinya runtuh 52% — PSI tidak akan menangkap kegagalan seperti itu.
Untuk fitur berbasis entitas, pantau juga jumlah observasi pendukungnya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-6


def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI = sum (actual% - expected%) * ln(actual% / expected%).

    Bin ditetapkan dari distribusi `expected` (periode acuan/training), lalu
    distribusi `actual` dipetakan ke bin yang sama. Menetapkan bin dari actual
    akan menyembunyikan pergeseran yang justru ingin diukur.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    inner = edges[1:-1]

    expected_pct = np.bincount(np.digitize(expected, inner), minlength=len(inner) + 1) / len(
        expected
    )
    actual_pct = np.bincount(np.digitize(actual, inner), minlength=len(inner) + 1) / len(actual)

    expected_pct = np.clip(expected_pct, EPSILON, None)
    actual_pct = np.clip(actual_pct, EPSILON, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_verdict(psi: float, stable: float = 0.10, warning: float = 0.25) -> str:
    if psi < stable:
        return "stabil"
    if psi < warning:
        return "perlu perhatian"
    return "bermasalah"


def psi_detail(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Rincian PSI per bin — menunjukkan di bagian distribusi mana pergeserannya."""
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    inner = edges[1:-1]

    exp_counts = np.bincount(np.digitize(expected, inner), minlength=len(inner) + 1)
    act_counts = np.bincount(np.digitize(actual, inner), minlength=len(inner) + 1)
    exp_pct = np.clip(exp_counts / len(expected), EPSILON, None)
    act_pct = np.clip(act_counts / len(actual), EPSILON, None)

    return pd.DataFrame(
        {
            "bin": range(len(exp_pct)),
            "bin_min": np.concatenate([[edges[0]], inner]),
            "expected_pct": exp_pct,
            "actual_pct": act_pct,
            "psi_contribution": (act_pct - exp_pct) * np.log(act_pct / exp_pct),
        }
    )


def characteristic_stability(
    expected: pd.DataFrame, actual: pd.DataFrame, features: list[str], n_bins: int = 10
) -> pd.DataFrame:
    """CSI per fitur, terurut dari yang paling bergeser."""
    rows = []
    for feature in features:
        exp_values = expected[feature].to_numpy(dtype=np.float64)
        act_values = actual[feature].to_numpy(dtype=np.float64)
        exp_valid = exp_values[np.isfinite(exp_values)]
        act_valid = act_values[np.isfinite(act_values)]

        if len(exp_valid) < n_bins or len(act_valid) < n_bins:
            continue
        csi = population_stability_index(exp_valid, act_valid, n_bins)
        rows.append(
            {
                "feature": feature,
                "csi": csi,
                "verdict": psi_verdict(csi),
                "mean_expected": exp_valid.mean(),
                "mean_actual": act_valid.mean(),
                "pct_change": (act_valid.mean() - exp_valid.mean())
                / (abs(exp_valid.mean()) + EPSILON),
            }
        )
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)
