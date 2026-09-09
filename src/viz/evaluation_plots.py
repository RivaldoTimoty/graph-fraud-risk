"""Figur Fase 5: distribusi skor, gain/lift, kurva biaya, reliability, PSI."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import resolve_path

FIGURE_DIR = resolve_path("reports/figures")
FRAUD_COLOR = "#bf616a"
GOOD_COLOR = "#5e81ac"


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_distribution(scores: np.ndarray, y_true: np.ndarray) -> Path:
    """Distribusi skor fraud vs non-fraud — pemisahan yang dilihat tim risk."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.linspace(scores.min(), scores.max(), 60)
    ax.hist(
        scores[y_true == 0], bins=bins, alpha=0.7, label="non-fraud", color=GOOD_COLOR, density=True
    )
    ax.hist(
        scores[y_true == 1], bins=bins, alpha=0.7, label="fraud", color=FRAUD_COLOR, density=True
    )
    ax.set_xlabel("skor (300-850, tinggi = risiko rendah)")
    ax.set_ylabel("densitas")
    ax.set_title("Distribusi skor: fraud vs non-fraud (test out-of-time)")
    ax.legend()
    return _save(fig, "06_score_distribution.png")


def plot_gain_lift(gains: pd.DataFrame) -> Path:
    """Gain chart dan lift chart berdampingan."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(gains["review_rate"], gains["recall"], color=FRAUD_COLOR, lw=2, label="model")
    ax1.plot([0, 1], [0, 1], ls="--", color="#888", label="acak")
    for k in (0.01, 0.05, 0.10):
        row = gains.iloc[(gains["review_rate"] - k).abs().idxmin()]
        ax1.plot(row["review_rate"], row["recall"], "o", color="#2e3440", ms=6)
        ax1.annotate(
            f"{k:.0%} -> {row['recall']:.1%}",
            (row["review_rate"], row["recall"]),
            textcoords="offset points",
            xytext=(8, -10),
            fontsize=8,
        )
    ax1.set_xlabel("review rate (fraksi transaksi ditinjau)")
    ax1.set_ylabel("recall (fraksi fraud tertangkap)")
    ax1.set_title("Gain chart")
    ax1.legend()

    ax2.plot(gains["review_rate"], gains["lift"], color=GOOD_COLOR, lw=2)
    ax2.axhline(1.0, ls="--", color="#888")
    ax2.set_xlabel("review rate")
    ax2.set_ylabel("lift (x lebih baik dari acak)")
    ax2.set_title("Lift chart")
    ax2.set_xlim(0, 0.5)

    fig.suptitle("Performa operasional pada berbagai kapasitas review", y=1.02)
    return _save(fig, "07_gain_lift.png")


def plot_cost_sensitivity(curves: dict[float, pd.DataFrame], summary: pd.DataFrame) -> Path:
    """Kurva biaya per skenario cost_fp, dengan titik optimal ditandai."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#5e81ac", "#a3be8c", "#ebcb8b", "#bf616a"]

    for (cost_fp, curve), color in zip(sorted(curves.items()), colors, strict=False):
        ax.plot(
            curve["review_rate"],
            curve["total_cost"] / 1000,
            label=f"cost_fp = ${cost_fp:g}",
            color=color,
            lw=1.8,
        )
        best = summary[summary["cost_fp"] == cost_fp].iloc[0]
        ax.plot(
            best["review_rate"],
            best["total_cost"] / 1000,
            "o",
            color=color,
            ms=9,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    ax.set_xlabel("review rate")
    ax.set_ylabel("total biaya (ribu $)")
    ax.set_title(
        "Total biaya vs review rate per skenario biaya false positive\n"
        "(titik = threshold optimal)"
    )
    ax.legend()
    ax.set_xlim(0, 0.6)
    return _save(fig, "08_cost_sensitivity.png")


def plot_reliability(
    before: pd.DataFrame, after: pd.DataFrame, ece_before: float, ece_after: float
) -> Path:
    """Reliability curve sebelum dan sesudah kalibrasi isotonic."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="#888", label="terkalibrasi sempurna")
    ax.plot(
        before["mean_predicted"],
        before["actual_rate"],
        "o-",
        color=FRAUD_COLOR,
        label=f"sebelum (ECE {ece_before:.4f})",
    )
    ax.plot(
        after["mean_predicted"],
        after["actual_rate"],
        "s-",
        color=GOOD_COLOR,
        label=f"isotonic (ECE {ece_after:.4f})",
    )

    limit = max(before["mean_predicted"].max(), after["mean_predicted"].max()) * 1.1
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("probabilitas prediksi rata-rata")
    ax.set_ylabel("fraud rate aktual")
    ax.set_title("Reliability curve — test out-of-time")
    ax.legend()
    return _save(fig, "09_reliability_curve.png")


def plot_psi(detail: pd.DataFrame, psi: float, csi: pd.DataFrame, top_n: int = 15) -> Path:
    """Distribusi skor train vs test per bin, dan CSI fitur teratas."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(detail))
    ax1.bar(x - 0.2, detail["expected_pct"], 0.4, label="train", color=GOOD_COLOR)
    ax1.bar(x + 0.2, detail["actual_pct"], 0.4, label="test", color=FRAUD_COLOR)
    ax1.set_xlabel("bin skor (dari kuantil train)")
    ax1.set_ylabel("proporsi populasi")
    ax1.set_title(f"Stabilitas distribusi skor — PSI {psi:.4f}")
    ax1.legend()

    top = csi.head(top_n).iloc[::-1]
    ax2.barh(top["feature"], top["csi"], color=GOOD_COLOR)
    ax2.axvline(0.10, ls="--", color="#ebcb8b", label="0,10 perhatian")
    ax2.axvline(0.25, ls="--", color=FRAUD_COLOR, label="0,25 bermasalah")
    ax2.set_xlabel("CSI")
    ax2.set_title(f"{top_n} fitur dengan pergeseran terbesar")
    ax2.legend()
    ax2.tick_params(axis="y", labelsize=8)

    return _save(fig, "10_stability_psi_csi.png")
