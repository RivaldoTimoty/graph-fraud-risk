"""Load dan join transaction + identity IEEE-CIS, simpan sebagai parquet.

Hanya ~24% transaksi punya identity record, sehingga join harus LEFT dari sisi
transaction agar tidak kehilangan baris. Ketiadaan identity itu sendiri informatif
(kanal/device tidak terekam), karena itu ditandai eksplisit lewat `has_identity`
alih-alih dibiarkan sebagai pola missing yang implisit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.reduce_memory import memory_usage_mb, reduce_memory


def _normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan penamaan kolom identity: test memakai 'id-01', train 'id_01'."""
    return df.rename(columns={c: c.replace("-", "_") for c in df.columns if c.startswith("id")})


def load_raw(transaction_path: Path, identity_path: Path, normalize: bool = True) -> pd.DataFrame:
    """Baca dua CSV mentah dan gabungkan via TransactionID.

    Menambahkan `has_identity` (1 jika transaksi punya identity record, 0 jika tidak).
    Tidak ada risiko leakage: flag ini tersedia saat transaksi terjadi.
    """
    transaction = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    if normalize:
        identity = _normalize_identity_columns(identity)

    identity = identity.assign(has_identity=np.uint8(1))
    merged = transaction.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    merged["has_identity"] = merged["has_identity"].fillna(0).astype("uint8")
    return merged


def build_merged_dataset(config_name: str = "data") -> Path:
    """Pipeline `make data`: load -> join -> reduce memory -> tulis parquet."""
    cfg = load_config(config_name)
    paths = cfg["paths"]

    merged = load_raw(
        resolve_path(paths["train_transaction"]),
        resolve_path(paths["train_identity"]),
        normalize=cfg["load"]["normalize_identity_columns"],
    )
    before = memory_usage_mb(merged)
    merged = reduce_memory(merged)
    after = memory_usage_mb(merged)

    out_path = resolve_path(paths["merged_train"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    print(f"rows={len(merged)} cols={merged.shape[1]}")
    print(f"has_identity rate={merged['has_identity'].mean():.4f}")
    print(f"fraud rate={merged['isFraud'].mean():.5f}")
    print(f"memory {before:.0f}MB -> {after:.0f}MB")
    print(f"written to {out_path} ({out_path.stat().st_size / 1024**2:.0f}MB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build merged IEEE-CIS train parquet")
    parser.add_argument("--config", default="data")
    build_merged_dataset(parser.parse_args().config)
