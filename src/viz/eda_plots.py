"""Figur EDA Fase 1. Dipanggil dari notebook maupun CLI supaya hasilnya reproducible."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import resolve_path

FIGURE_DIR = resolve_path("reports/figures")


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_weekly_fraud_rate(weekly: pd.DataFrame, boundaries: dict[str, int]) -> Path:
    """Fraud rate dan volume per minggu, dengan batas split ditandai."""
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.bar(weekly["week"], weekly["n"], color="#d8dee9", label="volume")
    ax1.set_xlabel("minggu (relatif)")
    ax1.set_ylabel("jumlah transaksi")

    ax2 = ax1.twinx()
    ax2.plot(weekly["week"], weekly["fraud_rate"], color="#bf616a", marker="o", label="fraud rate")
    ax2.axhline(weekly["fraud_rate"].mean(), color="#bf616a", ls=":", alpha=0.6)
    ax2.set_ylabel("fraud rate")

    for label, day in boundaries.items():
        ax1.axvline(day / 7, color="#5e81ac", ls="--", alpha=0.8)
        ax1.text(day / 7, ax1.get_ylim()[1] * 0.96, label, rotation=90, fontsize=8, va="top")

    ax1.set_title("Fraud rate mingguan dan volume transaksi (batas split ditandai)")
    return _save(fig, "01_weekly_fraud_rate.png")


def plot_hourly_profile(hourly: pd.DataFrame) -> Path:
    """Volume dan fraud rate per jam. Palung volume menandai offset zona waktu."""
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(hourly["hour"], hourly["n"], color="#d8dee9")
    ax1.set_xlabel("hour (relatif terhadap acuan DT, bukan jam lokal)")
    ax1.set_ylabel("jumlah transaksi")

    ax2 = ax1.twinx()
    ax2.plot(hourly["hour"], hourly["fraud_rate"], color="#bf616a", marker="o")
    ax2.set_ylabel("fraud rate")
    ax1.set_title("Profil per jam — volume rendah di hour 7-10 menandai offset zona waktu")
    return _save(fig, "02_hourly_profile.png")


def plot_cardinality(summary: pd.DataFrame) -> Path:
    """Kardinalitas kandidat node graph, skala log."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    s = summary.sort_values("n_unique")
    ax.barh(s["column"], s["n_unique"], color="#5e81ac")
    ax.set_xscale("log")
    ax.set_xlabel("jumlah nilai unik (log)")
    ax.set_title("Kardinalitas kandidat node graph")
    for i, (n, pct) in enumerate(zip(s["n_unique"], s["pct_missing"], strict=True)):
        ax.text(n * 1.1, i, f"{pct:.0%} missing", va="center", fontsize=8)
    return _save(fig, "03_cardinality.png")


def plot_coverage(coverage: pd.DataFrame) -> Path:
    """Coverage entitas lintas periode: berbobot baris vs berbobot nilai unik."""
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = coverage["column"] + " / " + coverage["period"]
    x = range(len(coverage))
    ax.bar(
        [i - 0.2 for i in x], coverage["pct_rows_covered"], 0.4, label="% baris", color="#5e81ac"
    )
    ax.bar(
        [i + 0.2 for i in x],
        coverage["pct_values_covered"],
        0.4,
        label="% nilai unik",
        color="#a3be8c",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage dari periode training")
    ax.set_title("Entitas OOT yang sudah terlihat saat training")
    ax.legend()
    return _save(fig, "04_entity_coverage.png")
