"""Rangkai feature matrix baseline dari parquet mentah + split mask.

Satu matriks dibangun berisi SEMUA blok fitur; pemilihan blok per model terjadi
saat training (`feature_blocks` di config model). Dengan begitu perbandingan
noC vs full benar-benar apple-to-apple: pipeline identik, hanya kolom yang beda.
"""

from __future__ import annotations

import argparse

import pandas as pd
import pyarrow.parquet as pq

from src.config import load_config, resolve_path
from src.data.split import add_time_features, load_split_masks
from src.features.tabular import (
    FrequencyEncoder,
    GroupAggregator,
    LabelEncoder,
    amount_features,
)

BLOCK_PREFIX = {"V": "V", "id": "id_"}


def _resolve_raw_blocks(cfg: dict, available: list[str]) -> dict[str, list[str]]:
    """`null` di config berarti ambil semua kolom dengan prefiks blok tersebut.

    Kolom yang sudah masuk `label_encode` dikeluarkan dari blok mentah - 15 kolom
    id_* bertipe string dan hanya boleh masuk matriks dalam bentuk ter-encode.
    """
    encoded = set(cfg["label_encode"])
    blocks = {}
    for name, cols in cfg["raw_blocks"].items():
        if cols is None:
            prefix = BLOCK_PREFIX[name]
            cols = [c for c in available if c.startswith(prefix)]
        blocks[name] = [c for c in cols if c in available and c not in encoded]
    return blocks


def build_feature_matrix(features_config: str = "features") -> pd.DataFrame:
    cfg = load_config(features_config)
    data_cfg = load_config("data")
    merged_path = resolve_path(data_cfg["paths"]["merged_train"])
    available = pq.ParquetFile(merged_path).schema.names

    raw_blocks = _resolve_raw_blocks(cfg, available)
    agg_cfg = cfg["aggregations"]
    source_cols = sorted(
        set(
            ["TransactionID", "TransactionDT", "isFraud", agg_cfg["value_col"]]
            + cfg["label_encode"]
            + cfg["frequency_encode"]
            + agg_cfg["group_cols"]
            + [c for cols in raw_blocks.values() for c in cols]
        )
        & set(available)
    )

    df = pd.read_parquet(merged_path, columns=source_cols)
    df = add_time_features(df)
    train_mask = load_split_masks()["is_train"].to_numpy()

    parts = [
        df[["TransactionID", "TransactionDT", "isFraud"]],
        df[cfg["derived"]["time"]],
        amount_features(df, agg_cfg["value_col"]),
        LabelEncoder(cfg["label_encode"]).fit(df, train_mask).transform(df),
        FrequencyEncoder(cfg["frequency_encode"]).fit(df, train_mask).transform(df),
        GroupAggregator(agg_cfg["value_col"], agg_cfg["group_cols"], agg_cfg["stats"])
        .fit(df, train_mask)
        .transform(df),
    ]
    parts.extend(df[cols] for cols in raw_blocks.values() if cols)
    return pd.concat(parts, axis=1)


def main(features_config: str = "features"):
    cfg = load_config(features_config)
    matrix = build_feature_matrix(features_config)

    out_path = resolve_path(cfg["paths"]["baseline_matrix"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out_path, index=False)

    print(f"rows={len(matrix)} cols={matrix.shape[1]}")
    print(f"written to {out_path} ({out_path.stat().st_size / 1024**2:.0f}MB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build baseline feature matrix")
    parser.add_argument("--config", default="features")
    main(parser.parse_args().config)
