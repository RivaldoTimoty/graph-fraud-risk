"""Fitur berbasis UID (Pendekatan B - entity resolution klien).

Semua fitur di sini struktural: menghitung intensitas dan keberagaman aktivitas
per klien, TANPA menyentuh label. Statistik berbasis label per UID adalah Level 3
dan memerlukan label_mask.

CATATAN LEAKAGE YANG HARUS DINYATAKAN DI LAPORAN: fitur UID diagregasi atas
seluruh periode, sehingga transaksi test ikut menghitung ukuran klien-nya sendiri.
Ini SAH sebagai simulasi produksi - saat menilai transaksi baru, histori klien
memang tersedia - tapi berbeda dari fitur yang murni backward-looking. Coverage
UID train->test hanya 33,9% baris, jadi dampaknya di OOT terbatas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def uid_activity_features(
    df: pd.DataFrame, uid: pd.Series, amount_col: str, device_col: str = "DeviceInfo"
) -> pd.DataFrame:
    """Intensitas dan keberagaman aktivitas per klien.

    `uid_size`            : berapa transaksi dilakukan klien ini. Klien dengan
        lonjakan transaksi mendadak adalah pola bust-out klasik.
    `uid_amt_mean/std`    : profil nominal klien. Std tinggi berarti perilaku
        belanja tidak konsisten.
    `uid_amt_ratio`       : nominal transaksi ini relatif terhadap rata-rata
        kliennya - menjawab "tidak wajar UNTUK klien ini", bukan sekadar besar.
    `uid_n_unique_device` : satu klien memakai banyak device.
    `uid_span_days`       : rentang hari aktivitas klien.
    `uid_tx_per_day`      : kepadatan transaksi; tinggi berarti aktivitas terburu.

    Baris dengan UID tak lengkap (kode negatif) mendapat NaN, bukan 0 - sama
    seperti perlakuan unseen entity di Fase 2: 0 akan dibaca model sebagai nilai
    nyata yang sangat rendah, padahal artinya "tidak diketahui".
    """
    frame = df.copy()
    frame["_uid"] = uid.to_numpy()
    valid = frame["_uid"] >= 0

    grouped = frame.loc[valid].groupby("_uid")
    stats = grouped.agg(
        uid_size=(amount_col, "size"),
        uid_amt_mean=(amount_col, "mean"),
        uid_amt_std=(amount_col, "std"),
        uid_day_min=("day", "min"),
        uid_day_max=("day", "max"),
    )
    if device_col in frame.columns:
        stats["uid_n_unique_device"] = grouped[device_col].nunique()

    stats["uid_span_days"] = stats["uid_day_max"] - stats["uid_day_min"]
    stats["uid_tx_per_day"] = stats["uid_size"] / (stats["uid_span_days"] + 1.0)
    stats = stats.drop(columns=["uid_day_min", "uid_day_max"])

    out = frame[["_uid"]].join(stats, on="_uid").drop(columns="_uid")
    out["uid_amt_ratio"] = frame[amount_col] / out["uid_amt_mean"]
    out.loc[~valid, :] = np.nan
    return out.astype("float32")
