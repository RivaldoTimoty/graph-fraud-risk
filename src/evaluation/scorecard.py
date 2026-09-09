"""Konversi probabilitas fraud ke skala skor kredit (300-850).

Konvensi industri credit scoring: skor TINGGI = risiko RENDAH. Karena model
memprediksi probabilitas fraud, hubungannya terbalik - probabilitas naik berarti
skor turun.

PDO (points to double the odds) menetapkan seberapa banyak poin yang diperlukan
agar odds non-fraud berlipat dua. Dengan PDO 20, transaksi berskor 620 punya odds
dua kali lebih baik daripada yang berskor 600.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-9


def score_factor(pdo: float) -> float:
    """Faktor skala: berapa poin per satuan log-odds."""
    return pdo / np.log(2.0)


def score_offset(anchor_score: float, anchor_odds: float, factor: float) -> float:
    """Offset supaya odds acuan jatuh tepat di skor acuan."""
    return anchor_score - factor * np.log(anchor_odds)


def probability_to_score(
    probability: np.ndarray,
    pdo: float,
    anchor_score: float,
    anchor_odds: float,
    score_min: float,
    score_max: float,
) -> np.ndarray:
    """Ubah P(fraud) menjadi skor. Probabilitas tinggi -> skor rendah.

    odds_good = (1 - p) / p, lalu score = offset + factor * ln(odds_good).
    Hasil dijepit ke [score_min, score_max] karena probabilitas ekstrem
    menghasilkan skor di luar skala yang bisa ditafsirkan.
    """
    p = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    factor = score_factor(pdo)
    offset = score_offset(anchor_score, anchor_odds, factor)
    raw = offset + factor * np.log((1.0 - p) / p)
    return np.clip(raw, score_min, score_max)


def score_bands(
    scores: np.ndarray, y_true: np.ndarray, n_bands: int = 10, ascending: bool = True
) -> pd.DataFrame:
    """Tabel per band skor: populasi, fraud rate, dan recall kumulatif.

    Ini bentuk yang biasa dibaca tim risk: "band skor terendah berisi berapa persen
    populasi, dan menangkap berapa persen fraud".
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bands + 1)))
    band = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)

    frame = pd.DataFrame({"band": band, "score": scores, "y": y_true})
    table = frame.groupby("band").agg(
        n=("y", "size"),
        n_fraud=("y", "sum"),
        fraud_rate=("y", "mean"),
        score_min=("score", "min"),
        score_max=("score", "max"),
    )
    table["pct_population"] = table["n"] / len(frame)

    order = table.index if ascending else table.index[::-1]
    table = table.loc[order]
    table["cum_pct_population"] = table["pct_population"].cumsum()
    table["cum_pct_fraud"] = table["n_fraud"].cumsum() / table["n_fraud"].sum()
    return table.reset_index()


def gain_table(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 100) -> pd.DataFrame:
    """Kurva gain dan lift untuk seluruh rentang review rate.

    Menjawab pertanyaan operasional tim risk pada setiap kapasitas review:
    "kalau kita sanggup meninjau k% transaksi teratas, berapa fraud yang tertangkap".
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score), kind="stable")
    sorted_y = y_true[order]

    n = len(y_true)
    total_fraud = sorted_y.sum()
    fractions = np.linspace(1.0 / n_points, 1.0, n_points)
    counts = np.maximum(1, (fractions * n).astype(int))

    caught = np.cumsum(sorted_y)[counts - 1]
    recall = caught / total_fraud
    precision = caught / counts
    return pd.DataFrame(
        {
            "review_rate": fractions,
            "n_reviewed": counts,
            "recall": recall,
            "precision": precision,
            "lift": recall / fractions,
        }
    )
