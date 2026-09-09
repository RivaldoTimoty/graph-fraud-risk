"""Temporal split dan fitur waktu turunan dari TransactionDT.

TransactionDT adalah detik relatif terhadap suatu titik acuan, bukan timestamp
absolut. Nilai minimumnya tepat 86400 (= 1 hari).

PERINGATAN INTERPRETASI:
- `dayofweek` RELATIF, bukan hari kalender sebenarnya. Tanggal absolut dataset
  tidak diketahui, jadi "day 0 = Senin" tidak bisa diklaim. Yang valid hanya
  periodisitas 7-harian.
- `hour` BUKAN jam lokal pengguna. Profil volume menunjukkan palung di hour 7-10
  dan puncak di hour 18-21; kalau acuannya tengah malam lokal, palung seharusnya
  di jam 2-5. Jadi ada offset ~5-7 jam (konsisten dengan DT dalam UTC sementara
  populasi pengguna di zona waktu Amerika). `hour` tetap berguna sebagai fitur
  siklikal, tapi jangan diberi label "dini hari"/"jam kerja" di laporan.

Split dipotong pada batas HARI, bukan persentil baris, sehingga tidak ada hari
yang terbelah antar split — hari terbelah akan membocorkan agregasi harian pada
fitur graph di Fase 3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path

SECONDS_PER_DAY = 86400
DAY_ORIGIN = 86400  # TransactionDT minimum; hari ke-0 dimulai di sini


def add_time_features(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    """Turunkan day/week/hour/dayofweek dari TransactionDT.

    Semua fitur di sini tersedia pada saat transaksi terjadi — tidak ada risiko
    leakage. `dayofweek` relatif terhadap hari ke-0, bukan kalender absolut.
    """
    out = df.copy()
    dt = out[time_col]
    out["day"] = ((dt - DAY_ORIGIN) // SECONDS_PER_DAY).astype("int16")
    out["week"] = (out["day"] // 7).astype("int16")
    out["hour"] = ((dt // 3600) % 24).astype("int8")
    out["dayofweek"] = (out["day"] % 7).astype("int8")
    return out


def transaction_day(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.Series:
    """Indeks hari (0-based) tanpa menyalin seluruh DataFrame."""
    return ((df[time_col] - DAY_ORIGIN) // SECONDS_PER_DAY).astype("int16")


@dataclass(frozen=True)
class SplitMasks:
    """Boolean mask per split. Disimpan sebagai mask, bukan tiga DataFrame terpisah,
    supaya hemat memori dan `train` bisa dipakai ulang sebagai label_mask di Fase 3."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    gap: np.ndarray

    def to_frame(self, index: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "TransactionID": index.to_numpy(),
                "is_train": self.train,
                "is_val": self.val,
                "is_test": self.test,
                "is_gap": self.gap,
            }
        )

    def summary(self, day: pd.Series, target: pd.Series) -> pd.DataFrame:
        rows = []
        for name in ("train", "gap", "val", "test"):
            m = getattr(self, name)
            rows.append(
                {
                    "split": name,
                    "n": int(m.sum()),
                    "pct": float(m.mean()),
                    "day_min": int(day[m].min()) if m.any() else None,
                    "day_max": int(day[m].max()) if m.any() else None,
                    "fraud_rate": float(target[m].mean()) if m.any() else None,
                }
            )
        return pd.DataFrame(rows)


def temporal_split(
    df: pd.DataFrame,
    train_end_day: int,
    gap_days: int,
    val_end_day: int,
    time_col: str = "TransactionDT",
) -> SplitMasks:
    """Bagi data secara temporal pada batas hari.

    train = [0, train_end_day], gap = (train_end_day, train_end_day + gap_days],
    val = (gap_end, val_end_day], test = (val_end_day, akhir].

    Gap meniru delay pelaporan fraud di dunia nyata: pada saat model dilatih,
    label untuk transaksi paling baru belum tersedia.
    """
    day = transaction_day(df, time_col)
    gap_end = train_end_day + gap_days
    if not train_end_day < gap_end < val_end_day < day.max():
        raise ValueError(
            f"batas split tidak valid: train_end={train_end_day} gap_end={gap_end} "
            f"val_end={val_end_day} max_day={day.max()}"
        )

    return SplitMasks(
        train=(day <= train_end_day).to_numpy(),
        gap=((day > train_end_day) & (day <= gap_end)).to_numpy(),
        val=((day > gap_end) & (day <= val_end_day)).to_numpy(),
        test=(day > val_end_day).to_numpy(),
    )


def build_split_masks(config_name: str = "split", data_config: str = "data") -> Path:
    """Hitung split dari parquet dan simpan mask supaya reproducible tanpa re-run."""
    cfg = load_config(config_name)
    data_cfg = load_config(data_config)
    params = cfg["split"]

    merged_path = resolve_path(data_cfg["paths"]["merged_train"])
    df = pd.read_parquet(merged_path, columns=["TransactionID", "TransactionDT", "isFraud"])

    masks = temporal_split(
        df,
        train_end_day=params["train_end_day"],
        gap_days=params["gap_days"],
        val_end_day=params["val_end_day"],
    )

    out_path = resolve_path(cfg["paths"]["masks"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    masks.to_frame(df["TransactionID"]).to_parquet(out_path, index=False)

    print(masks.summary(transaction_day(df), df["isFraud"]).to_string(index=False))
    print(f"\nwritten to {out_path}")
    return out_path


def load_split_masks(config_name: str = "split") -> pd.DataFrame:
    """Baca mask yang sudah tersimpan. Dipakai fase-fase berikutnya."""
    return pd.read_parquet(resolve_path(load_config(config_name)["paths"]["masks"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build temporal split masks")
    parser.add_argument("--config", default="split")
    build_split_masks(parser.parse_args().config)
