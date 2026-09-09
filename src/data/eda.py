"""Jalankan analisis EDA Fase 1 dan simpan figur + tabel ringkasan.

Notebook memanggil fungsi-fungsi ini, bukan mengulang logic-nya.
"""

from __future__ import annotations

import pandas as pd

from src.config import load_config, resolve_path
from src.data.profiling import (
    cardinality_summary,
    coverage_across_periods,
    weekly_fraud_rate,
)
from src.data.split import add_time_features, load_split_masks
from src.viz.eda_plots import (
    plot_cardinality,
    plot_coverage,
    plot_hourly_profile,
    plot_weekly_fraud_rate,
)


def _hourly_profile(df: pd.DataFrame, target: str = "isFraud") -> pd.DataFrame:
    return df.groupby("hour").agg(n=(target, "size"), fraud_rate=(target, "mean")).reset_index()


def run_eda(split_config: str = "split", data_config: str = "data") -> dict[str, pd.DataFrame]:
    cfg = load_config(split_config)
    data_cfg = load_config(data_config)
    params = cfg["split"]
    candidates = [c for group in cfg["cardinality_candidates"].values() for c in group]

    merged_path = resolve_path(data_cfg["paths"]["merged_train"])
    base_cols = ["TransactionID", "TransactionDT", "isFraud"]
    df = pd.read_parquet(merged_path, columns=base_cols + candidates)
    df = add_time_features(df)

    weekly = weekly_fraud_rate(df)
    hourly = _hourly_profile(df)
    cardinality = cardinality_summary(df, candidates)

    masks = load_split_masks(split_config)
    coverage = coverage_across_periods(
        df,
        cfg["coverage_columns"],
        masks["is_train"].to_numpy(),
        {"val": masks["is_val"].to_numpy(), "test": masks["is_test"].to_numpy()},
    )

    gap_end = params["train_end_day"] + params["gap_days"]
    plot_weekly_fraud_rate(
        weekly,
        {
            "train end": params["train_end_day"],
            "val start": gap_end,
            "test start": params["val_end_day"],
        },
    )
    plot_hourly_profile(hourly)
    plot_cardinality(cardinality)
    plot_coverage(coverage)

    return {
        "weekly": weekly,
        "hourly": hourly,
        "cardinality": cardinality,
        "coverage": coverage,
    }


if __name__ == "__main__":
    results = run_eda()
    for name, table in results.items():
        print(f"\n=== {name.upper()} ===")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
