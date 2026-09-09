"""Bangun fitur Level 3 (berbasis label) dan simpan ke parquet.

label_mask SELALU = is_train, untuk seluruh baris. Fitur pada baris test dihitung
dari label training, bukan train+val — pilihan konservatif yang disengaja.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_config, resolve_path
from src.data.split import add_time_features, load_split_masks
from src.graph.build import build_incidence, build_uid, load_source_frame
from src.graph.community import transaction_community_ids
from src.graph.features import cap_high_degree_columns
from src.graph.label_features import build_level3_features, training_prior


def build(config_name: str = "graph") -> pd.DataFrame:
    cfg = load_config(config_name)
    l3_cfg = cfg["level3"]

    df = add_time_features(load_source_frame(cfg))
    y = pd.read_parquet(
        resolve_path(load_config("data")["paths"]["merged_train"]), columns=["isFraud"]
    )["isFraud"].to_numpy()

    label_mask = load_split_masks()["is_train"].to_numpy().astype(bool)
    prior = training_prior(y, label_mask)
    print(f"label_mask: {label_mask.sum():,} baris training | prior fraud rate {prior:.5f}")

    graph = build_incidence(df, cfg["node_columns"])
    capped, n_capped = cap_high_degree_columns(graph.incidence, cfg["max_degree_for_aggregation"])
    print(f"hub cap: {n_capped} node dikeluarkan dari propagasi label")

    community = transaction_community_ids(capped, cfg["community"]["resolution"], cfg["seed"])
    uid = build_uid(df, cfg).to_numpy()

    features = build_level3_features(
        capped,
        y,
        label_mask,
        community,
        uid,
        alpha=l3_cfg["alpha"],
        include_2hop=l3_cfg["include_2hop"],
    )
    return pd.concat([df[["TransactionID"]], features], axis=1)


def correlation_with_c_features(features: pd.DataFrame) -> pd.DataFrame:
    """Seberapa mirip fitur Level 3 dengan C-features Vesta?

    Fase 2 menemukan C13 berkorelasi 0,46 dengan degree UID. Kalau
    graph_nb_fraud_rate_1hop juga berkorelasi tinggi dengan C13, itu bukti
    tambahan bahwa Vesta sudah meng-encode sinyal bertipe jaringan — temuan yang
    menentukan interpretasi ablation Fase 4.

    Korelasi dihitung pada PERIODE TRAINING saja, agar konsisten dengan
    label_mask yang membentuk fitur Level 3.
    """
    merged_path = resolve_path(load_config("data")["paths"]["merged_train"])
    c_cols = [f"C{i}" for i in range(1, 15)]
    c_df = pd.read_parquet(merged_path, columns=c_cols)
    train = load_split_masks()["is_train"].to_numpy()

    targets = [c for c in features.columns if "fraud_rate" in c]
    rows = []
    for target in targets:
        left = features.loc[train, target]
        for c in c_cols:
            rows.append(
                {
                    "level3_feature": target,
                    "c_feature": c,
                    "spearman": left.corr(c_df.loc[train, c], method="spearman"),
                }
            )
    return pd.DataFrame(rows)


def main(config_name: str = "graph"):
    cfg = load_config(config_name)
    features = build(config_name)

    out_path = resolve_path(cfg["level3"]["features_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    print(f"\nrows={len(features)} cols={features.shape[1] - 1}")
    print(f"written to {out_path}")

    corr = correlation_with_c_features(features)
    pivot = corr.pivot(index="c_feature", columns="level3_feature", values="spearman")
    pivot = pivot.reindex([f"C{i}" for i in range(1, 15)])
    print("\n=== Korelasi Spearman: fitur Level 3 vs C-features (training) ===")
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Level 3 label-based features")
    parser.add_argument("--config", default="graph")
    main(parser.parse_args().config)
