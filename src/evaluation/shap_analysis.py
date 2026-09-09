"""Analisis SHAP: seberapa besar kontribusi fitur graph dibanding C dan lainnya.

Dihitung pada subsample VALIDATION, bukan test — keputusan menggugurkan fitur
harus diambil sebelum test set dibuka, supaya test tidak berubah menjadi alat
seleksi fitur.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.config import resolve_path

FIGURE_DIR = resolve_path("reports/figures")


def feature_group(name: str) -> str:
    """Kelompokkan fitur untuk agregasi SHAP."""
    if name.startswith("graph_"):
        return "graph"
    if name.startswith("uid_"):
        return "uid"
    if name.startswith("C") and name[1:].isdigit():
        return "C (Vesta counting)"
    if name.startswith("V"):
        return "V (Vesta)"
    if name.startswith(("D",)) and name[1:].isdigit():
        return "D (timedelta)"
    return "lainnya"


def compute_shap(
    booster: xgb.Booster, X: pd.DataFrame, sample_size: int, seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Hitung SHAP pada subsample. Mengembalikan (nilai, baris yang dipakai)."""
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    sample = X.iloc[idx]

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(sample)
    return values, sample


def importance_table(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per fitur, terurut menurun, dengan label kelompoknya."""
    mean_abs = np.abs(values).mean(axis=0)
    table = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
    table["group"] = table["feature"].map(feature_group)
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def group_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Agregasi kontribusi SHAP per kelompok fitur."""
    summary = table.groupby("group").agg(
        n_features=("feature", "size"),
        total_shap=("mean_abs_shap", "sum"),
        mean_shap=("mean_abs_shap", "mean"),
        max_shap=("mean_abs_shap", "max"),
    )
    summary["pct_of_total"] = summary["total_shap"] / summary["total_shap"].sum()
    return summary.sort_values("total_shap", ascending=False)


def plot_summary(values: np.ndarray, sample: pd.DataFrame, name: str, top_n: int = 20):
    """Summary plot SHAP, fitur graph ditandai lewat warna label sumbu-y."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(values, sample, max_display=top_n, show=False)

    ax = plt.gca()
    for label in ax.get_yticklabels():
        group = feature_group(label.get_text())
        if group in ("graph", "uid"):
            label.set_color("#bf616a")
            label.set_fontweight("bold")

    plt.title(f"SHAP top {top_n} — {name} (merah = fitur graph)", fontsize=11)
    plt.tight_layout()
    path = FIGURE_DIR / f"05_shap_summary_{name}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path
