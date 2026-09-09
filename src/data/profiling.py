"""Ringkasan kardinalitas, missingness, dan coverage lintas periode.

Dipakai oleh notebook EDA Fase 1 dan oleh keputusan desain node graph di Fase 3.
Logic di sini, bukan di notebook, karena dipakai lebih dari sekali.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cardinality_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Profil tiap kandidat node graph.

    `pct_singleton` adalah metrik paling menentukan: nilai yang hanya muncul sekali
    menghasilkan node berderajat 1 yang tidak menghubungkan transaksi mana pun,
    sehingga tidak berkontribusi pada struktur graph.
    """
    rows = []
    n = len(df)
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        counts = s.value_counts(dropna=True)
        rows.append(
            {
                "column": col,
                "n_unique": int(s.nunique(dropna=True)),
                "pct_missing": float(s.isna().mean()),
                "largest_group": int(counts.iloc[0]) if len(counts) else 0,
                "pct_in_largest": float(counts.iloc[0] / n) if len(counts) else 0.0,
                "n_singleton": int((counts == 1).sum()),
                "pct_singleton": float((counts == 1).sum() / len(counts)) if len(counts) else 0.0,
                "median_group_size": float(counts.median()) if len(counts) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("n_unique", ascending=False)


def coverage_across_periods(
    df: pd.DataFrame,
    columns: list[str],
    train_mask: np.ndarray,
    target_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Berapa persen nilai di periode lain yang sudah pernah terlihat di training.

    Menentukan ekspektasi realistis untuk fitur berbasis entitas: kalau coverage
    rendah, statistik entitas dari periode training tidak akan berlaku di OOT dan
    fitur graph berbasis label akan lemah di sana.

    Dilaporkan dua cara: `pct_rows_covered` (bobot volume — dampak praktis) dan
    `pct_values_covered` (bobot entitas unik — seberapa banyak entitas baru muncul).
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        seen = set(df.loc[train_mask, col].dropna().unique())
        for period, mask in target_masks.items():
            s = df.loc[mask, col].dropna()
            uniq = s.unique()
            rows.append(
                {
                    "column": col,
                    "period": period,
                    "n_rows": int(len(s)),
                    "n_unique": int(len(uniq)),
                    "pct_rows_covered": float(s.isin(seen).mean()) if len(s) else np.nan,
                    "pct_values_covered": (
                        float(np.isin(uniq, list(seen)).mean()) if len(uniq) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def weekly_fraud_rate(df: pd.DataFrame, week_col: str = "week", target: str = "isFraud"):
    """Volume dan fraud rate per minggu, untuk mendeteksi drift temporal."""
    return df.groupby(week_col).agg(n=(target, "size"), fraud_rate=(target, "mean")).reset_index()


def missing_pattern_groups(df: pd.DataFrame, columns: list[str], min_group: int = 2):
    """Kelompokkan kolom yang punya pola missing identik.

    Banyak kolom Vxxx punya pola missing yang persis sama — petunjuk bahwa mereka
    berasal dari satu sumber/blok fitur Vesta, sehingga bisa diperlakukan sebagai
    satu unit saat seleksi fitur.
    """
    signatures: dict[bytes, list[str]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        sig = np.packbits(df[col].isna().to_numpy()).tobytes()
        signatures.setdefault(sig, []).append(col)
    groups = [cols for cols in signatures.values() if len(cols) >= min_group]
    return sorted(groups, key=len, reverse=True)
