"""Optimasi threshold berbasis biaya - menerjemahkan AUC menjadi bahasa rupiah.

ASUMSI YANG DINYATAKAN EKSPLISIT:

  Biaya false negative = TransactionAmt transaksi itu sendiri. Fraud yang lolos
      berarti kerugian sebesar nilai transaksinya. Dipakai per baris, bukan
      rata-rata, karena fraud bernilai besar dan kecil punya konsekuensi berbeda.

  Biaya false positive = konstanta per transaksi (review manual + friksi
      pelanggan). Ini ASUMSI, bukan angka terukur. Karena itu seluruh analisis
      dijalankan pada beberapa skenario, sehingga stakeholder bisa memilih
      threshold sesuai asumsi biaya mereka sendiri, bukan sesuai asumsi saya.

Yang TIDAK dimodelkan (batas dari analisis ini): biaya reputasi jangka panjang,
churn pelanggan yang salah ditolak, dan biaya tetap operasional tim review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    cost_fp: float,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """Total biaya pada tiap threshold.

    Transaksi dengan skor >= threshold ditandai untuk review (diblokir).
    - Fraud yang lolos (di bawah threshold) -> biaya sebesar nilai transaksinya.
    - Non-fraud yang tertahan (di atas threshold) -> biaya review.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    amounts = np.asarray(amounts, dtype=np.float64)

    thresholds = np.quantile(y_score, np.linspace(0.0, 1.0, n_thresholds))
    thresholds = np.unique(thresholds)

    rows = []
    total_fraud_value = amounts[y_true].sum()
    for threshold in thresholds:
        flagged = y_score >= threshold
        missed_fraud = y_true & ~flagged
        false_positive = ~y_true & flagged

        fn_cost = amounts[missed_fraud].sum()
        fp_cost = false_positive.sum() * cost_fp
        rows.append(
            {
                "threshold": threshold,
                "review_rate": flagged.mean(),
                "n_flagged": int(flagged.sum()),
                "fraud_caught": int((y_true & flagged).sum()),
                "recall": float((y_true & flagged).sum() / y_true.sum()),
                "fn_cost": fn_cost,
                "fp_cost": fp_cost,
                "total_cost": fn_cost + fp_cost,
                "fraud_value_saved": total_fraud_value - fn_cost,
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame) -> pd.Series:
    """Titik dengan total biaya minimum."""
    return curve.loc[curve["total_cost"].idxmin()]


def baseline_cost(y_true: np.ndarray, amounts: np.ndarray) -> float:
    """Biaya bila tidak ada model sama sekali: seluruh fraud lolos."""
    return float(np.asarray(amounts)[np.asarray(y_true).astype(bool)].sum())


def sensitivity_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amounts: np.ndarray,
    scenarios: list[float],
    n_thresholds: int = 200,
    days_in_period: int = 31,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    """Bagaimana threshold optimal bergeser terhadap asumsi biaya FP.

    Mengubah deliverable dari "saya memilih threshold" menjadi "ini alat untuk
    memilih threshold sesuai asumsi bisnis Anda" - jauh lebih berguna karena
    biaya FP berbeda antar organisasi dan tidak pernah benar-benar diketahui.
    """
    no_model = baseline_cost(y_true, amounts)
    curves, rows = {}, []

    for cost_fp in scenarios:
        curve = cost_curve(y_true, y_score, amounts, cost_fp, n_thresholds)
        best = optimal_threshold(curve)
        curves[cost_fp] = curve

        savings = no_model - best["total_cost"]
        rows.append(
            {
                "cost_fp": cost_fp,
                "optimal_threshold": best["threshold"],
                "review_rate": best["review_rate"],
                "recall": best["recall"],
                "total_cost": best["total_cost"],
                "fn_cost": best["fn_cost"],
                "fp_cost": best["fp_cost"],
                "savings_vs_no_model": savings,
                "savings_per_month": savings * 30.0 / days_in_period,
                "pct_cost_reduction": savings / no_model if no_model else np.nan,
            }
        )
    return pd.DataFrame(rows), curves
