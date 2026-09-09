"""Downcast dtype numerik untuk menekan memori.

Dataset mentah IEEE-CIS ~1.5GB sebagai CSV dan >2GB di memori dengan dtype default
pandas (int64/float64). Downcasting ke tipe terkecil yang masih muat menurunkan
footprint ~60-70% tanpa kehilangan informasi pada kolom integer.

Catatan presisi: float64 -> float32 adalah lossy (≈7 digit signifikan). Untuk fitur
Vesta yang sudah ter-scale dan TransactionAmt (maksimum ~32k, 3 desimal) ini aman,
tapi jangan pakai float32 untuk akumulasi statistik berpresisi tinggi nanti.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Kolom yang tidak boleh di-downcast: identitas dan target harus tetap eksak,
# dan TransactionDT dipakai sebagai basis semua temporal split.
PROTECTED_COLUMNS = ("TransactionID", "TransactionDT", "isFraud")


def _downcast_int(series: pd.Series) -> pd.Series:
    """Pilih tipe integer terkecil yang memuat rentang nilai."""
    return pd.to_numeric(series, downcast="unsigned" if series.min() >= 0 else "integer")


def _downcast_float(series: pd.Series) -> pd.Series:
    """Turunkan ke float32 hanya jika nilainya muat dalam rentang float32."""
    finite = series[np.isfinite(series)]
    if finite.empty:
        return series.astype(np.float32)
    if finite.abs().max() > np.finfo(np.float32).max:
        return series
    return series.astype(np.float32)


def reduce_memory(df: pd.DataFrame, protected: tuple[str, ...] = PROTECTED_COLUMNS) -> pd.DataFrame:
    """Downcast semua kolom numerik non-protected in-place pada salinan.

    Kolom object dibiarkan apa adanya — konversi ke category ditunda sampai tahap
    feature engineering, karena kategori harus di-fit hanya pada periode training.
    """
    out = df.copy()
    for col in out.columns:
        if col in protected:
            continue
        dtype = out[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            out[col] = _downcast_int(out[col])
        elif pd.api.types.is_float_dtype(dtype):
            out[col] = _downcast_float(out[col])
    return out


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Total memori DataFrame dalam MB (termasuk object overhead)."""
    return float(df.memory_usage(deep=True).sum() / 1024**2)
