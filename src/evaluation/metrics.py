"""Metrik evaluasi: AUC, KS, PR-AUC, lift@k.

KS wajib ada karena ini standar industri credit scoring - mengukur pemisahan
maksimum antara distribusi kumulatif good dan bad.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

DEFAULT_K = (0.01, 0.05, 0.10)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: jarak vertikal maksimum antara TPR dan FPR.

    Interpretasi risk: seberapa jauh model memisahkan populasi fraud dari
    non-fraud. KS 0.60+ dianggap kuat untuk scorecard.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> dict[str, float]:
    """Performa pada k fraksi teratas - pertanyaan operasional tim risk.

    "Kalau kita review 1% transaksi berskor tertinggi, berapa persen fraud yang
    tertangkap?" -> `recall`. `lift` adalah berapa kali lebih baik dari memilih
    acak; lift 20 pada k=1% berarti 20x lebih efisien daripada review random.
    """
    n = len(y_true)
    n_top = max(1, int(np.ceil(n * k)))
    order = np.argsort(-y_score, kind="stable")[:n_top]
    top = y_true[order]

    n_fraud = float(y_true.sum())
    caught = float(top.sum())
    precision = caught / n_top
    base_rate = n_fraud / n

    return {
        "k": k,
        "n_reviewed": n_top,
        "precision": precision,
        "recall": caught / n_fraud if n_fraud else float("nan"),
        "lift": precision / base_rate if base_rate else float("nan"),
    }


def evaluate(y_true, y_score, ks_levels: tuple[float, ...] = DEFAULT_K) -> dict[str, float]:
    """Semua metrik utama dalam satu dict, siap ditulis ke experiments.csv."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    out: dict[str, float] = {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "fraud_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    for k in ks_levels:
        m = lift_at_k(y_true, y_score, k)
        tag = f"{k:.0%}".replace("%", "pct")
        out[f"recall_at_{tag}"] = m["recall"]
        out[f"precision_at_{tag}"] = m["precision"]
        out[f"lift_at_{tag}"] = m["lift"]
    return out


def format_metrics(metrics: dict[str, float]) -> str:
    """Ringkasan satu baris untuk log training."""
    return (
        f"AUC {metrics['auc']:.4f} | KS {metrics['ks']:.4f} | PR-AUC {metrics['pr_auc']:.4f} "
        f"| recall@1% {metrics['recall_at_1pct']:.3f} "
        f"| recall@5% {metrics['recall_at_5pct']:.3f} "
        f"| recall@10% {metrics['recall_at_10pct']:.3f}"
    )
