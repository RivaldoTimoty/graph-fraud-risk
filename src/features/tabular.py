"""Feature engineering tabular untuk baseline Fase 2.

ATURAN LEAKAGE YANG MENGIKAT SELURUH MODUL INI: setiap statistik yang diturunkan
dari data (frekuensi kategori, mean/std per entitas, peta label encoding) di-fit
HANYA pada baris dengan `train_mask == True`, lalu di-transform ke seluruh baris.
Periode gap dan val/test tidak pernah ikut menghitung statistik.

Kategori yang tidak pernah terlihat saat training di-encode sebagai:
- frequency  -> 0   (benar: entitas ini memang tidak pernah muncul di training)
- agregasi   -> NaN (BUKAN 0: 0 akan dibaca model sebagai "nominal jauh di bawah
                     rata-rata", padahal artinya "tidak diketahui")
Ini bukan detail kosmetik — 12,3% nilai card1 di test belum pernah terlihat saat
training (temuan Fase 1 T4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FrequencyEncoder:
    """Petakan tiap kategori ke jumlah kemunculannya di periode training.

    Alasan bisnis: entitas langka (kartu/email/device yang baru muncul) lebih
    berisiko, sementara entitas berfrekuensi sangat tinggi sering merupakan
    merchant atau proxy besar. Frekuensi mentah, bukan proporsi, supaya skalanya
    stabil saat ukuran periode berubah.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.maps_: dict[str, pd.Series] = {}

    def fit(self, df: pd.DataFrame, train_mask: np.ndarray) -> FrequencyEncoder:
        train = df.loc[train_mask]
        self.maps_ = {c: train[c].value_counts(dropna=True) for c in self.columns if c in df}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, counts in self.maps_.items():
            out[f"freq_{col}"] = df[col].map(counts).fillna(0).astype("float32")
        return pd.DataFrame(out, index=df.index)


class GroupAggregator:
    """Statistik `value_col` per entitas, plus rasio nilai terhadap statistik itu.

    Alasan bisnis: pertanyaan analis fraud bukan "apakah nominal ini besar" tapi
    "apakah nominal ini tidak wajar UNTUK kartu ini". Rasio amt/mean_per_card
    menangkap deviasi dari perilaku normal entitas tersebut.

    Entitas yang tidak terlihat saat training menghasilkan NaN — lihat catatan
    leakage di docstring modul.
    """

    def __init__(self, value_col: str, group_cols: list[str], stats: list[str]):
        self.value_col = value_col
        self.group_cols = group_cols
        self.stats = stats
        self.tables_: dict[str, pd.DataFrame] = {}

    def fit(self, df: pd.DataFrame, train_mask: np.ndarray) -> GroupAggregator:
        train = df.loc[train_mask]
        self.tables_ = {
            col: train.groupby(col)[self.value_col].agg(self.stats)
            for col in self.group_cols
            if col in df
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        value = df[self.value_col]
        for col, table in self.tables_.items():
            for stat in self.stats:
                mapped = df[col].map(table[stat]).astype("float32")
                out[f"{self.value_col}_{stat}_by_{col}"] = mapped
                if stat == "mean":
                    out[f"{self.value_col}_ratio_to_mean_{col}"] = (value / mapped).astype(
                        "float32"
                    )
        return pd.DataFrame(out, index=df.index)


class LabelEncoder:
    """Kode integer stabil untuk kategorikal kardinalitas rendah.

    Kategori tak terlihat -> -1, dipisahkan dari kategori valid (0..n-1) supaya
    LightGBM bisa memberinya cabang tersendiri.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.maps_: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame, train_mask: np.ndarray) -> LabelEncoder:
        train = df.loc[train_mask]
        self.maps_ = {
            c: {v: i for i, v in enumerate(sorted(train[c].dropna().unique().tolist()))}
            for c in self.columns
            if c in df
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, mapping in self.maps_.items():
            out[f"le_{col}"] = df[col].map(mapping).fillna(-1).astype("int32")
        return pd.DataFrame(out, index=df.index)


def amount_features(df: pd.DataFrame, col: str = "TransactionAmt") -> pd.DataFrame:
    """Transformasi nominal transaksi. Tidak butuh fitting — bebas leakage.

    `amt_log`     : distribusi TransactionAmt sangat skew; log mendekatkan ke normal.
    `amt_decimal` : bagian desimal. Nominal hasil konversi mata uang punya desimal
                    non-bulat, sehingga ini menjadi proksi transaksi lintas negara.
    """
    amt = df[col]
    return pd.DataFrame(
        {
            "amt_log": np.log1p(amt).astype("float32"),
            "amt_decimal": (amt - np.floor(amt)).astype("float32"),
        },
        index=df.index,
    )
