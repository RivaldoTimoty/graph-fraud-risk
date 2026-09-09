"""Kalibrasi probabilitas: reliability curve dan isotonic regression.

Model boosting biasanya menghasilkan skor yang baik untuk RANKING tapi buruk
sebagai PROBABILITAS. Ini penting karena keputusan berbasis biaya (Fase 5)
membutuhkan probabilitas yang bermakna: "5% kemungkinan fraud pada transaksi
senilai 1 juta" hanya bisa dihitung kalau 5% itu benar-benar berarti 5%.

ATURAN: isotonic di-fit pada VALIDATION, diterapkan ke test. Fitting pada test
akan membuat kalibrasi menilai dirinya sendiri.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> float:
    """Rata-rata |confidence - accuracy| berbobot ukuran bin."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1])
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum():
            total += mask.sum() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(total / len(y_true))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error probabilitas - menggabungkan kalibrasi dan ketajaman."""
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    """Prediksi rata-rata vs frekuensi aktual per bin.

    Model terkalibrasi sempurna jatuh pada garis diagonal. Di atas diagonal berarti
    under-predict, di bawah berarti over-predict (khas boosting).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1])

    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum():
            rows.append(
                {
                    "bin": b,
                    "n": int(mask.sum()),
                    "mean_predicted": y_prob[mask].mean(),
                    "actual_rate": y_true[mask].mean(),
                }
            )
    return pd.DataFrame(rows)


def fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    """Fit kalibrator isotonic. Monoton, sehingga ranking (AUC/KS) tidak berubah."""
    return IsotonicRegression(out_of_bounds="clip").fit(
        np.asarray(y_prob, dtype=np.float64), np.asarray(y_true, dtype=np.float64)
    )


def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> dict:
    """Ringkasan kalibrasi dalam satu dict."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return {
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "brier": brier_score(y_true, y_prob),
        "mean_predicted": float(y_prob.mean()),
        "actual_rate": float(y_true.mean()),
        "bias": float(y_prob.mean() - y_true.mean()),
    }
