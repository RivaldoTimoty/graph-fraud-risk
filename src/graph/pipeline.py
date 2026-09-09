"""Rangkai seluruh fitur graph Level 1-2 menjadi satu matriks.

Prefiks nama kolom menandai asal fitur, supaya Fase 4 bisa menyusun feature set
selektif untuk ablation:
  graph_*  -> bipartite graph transaksi-atribut (Pendekatan A)
  uid_*    -> entity resolution klien (Pendekatan B)
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_config, resolve_path
from src.data.split import add_time_features
from src.graph.build import build_incidence, build_uid, load_source_frame
from src.graph.community import community_and_core
from src.graph.features import (
    attribute_entropy,
    cap_high_degree_columns,
    component_features,
    cross_cardinality_features,
    degree_features,
    neighbor_count_features,
    pagerank,
)
from src.graph.uid_features import uid_activity_features


def build_graph_features(config_name: str = "graph") -> pd.DataFrame:
    cfg = load_config(config_name)
    df = add_time_features(load_source_frame(cfg))
    graph = build_incidence(df, cfg["node_columns"])

    capped, n_capped = cap_high_degree_columns(graph.incidence, cfg["max_degree_for_aggregation"])
    print(f"hub cap: {n_capped} node atribut dikeluarkan dari agregasi tetangga")

    pairs = [tuple(p) for p in cfg["cross_cardinality"]]
    pr_cfg, com_cfg = cfg["pagerank"], cfg["community"]

    parts = [
        df[["TransactionID"]],
        degree_features(graph),
        cross_cardinality_features(df, pairs),
        neighbor_count_features(capped),
        attribute_entropy(graph),
        pd.DataFrame({"graph_pagerank": pagerank(capped, **pr_cfg).astype("float32")}),
        component_features(capped),
        community_and_core(capped, com_cfg["resolution"], cfg["seed"]),
        uid_activity_features(df, build_uid(df, cfg), cfg["uid"]["aggregate_col"]),
    ]
    return pd.concat(parts, axis=1)


def main(config_name: str = "graph"):
    cfg = load_config(config_name)
    features = build_graph_features(config_name)

    out_path = resolve_path(cfg["paths"]["features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)

    graph_cols = [c for c in features.columns if c.startswith("graph_")]
    uid_cols = [c for c in features.columns if c.startswith("uid_")]
    print(f"\nrows={len(features)} | graph_*={len(graph_cols)} | uid_*={len(uid_cols)}")
    print(f"written to {out_path} ({out_path.stat().st_size / 1024**2:.1f}MB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Level 1-2 graph features")
    parser.add_argument("--config", default="graph")
    main(parser.parse_args().config)
