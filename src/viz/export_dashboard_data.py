"""Precompute artefak ringan untuk dashboard Next.js.

Model, parquet, dan CSV mentah semuanya ter-.gitignore — benar untuk repo, tapi
artinya dashboard tidak akan punya data setelah `git clone`. Script ini
menghasilkan JSON kecil yang ter-commit, supaya dashboard jalan dari clone bersih
tanpa perlu menjalankan ulang seluruh pipeline.

Semua perhitungan berat (training, layout graph) dilakukan di sini; browser hanya
membaca hasilnya.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.split import load_split_masks
from src.evaluation.business import baseline_cost, sensitivity_analysis
from src.evaluation.phase5 import predict, train_final_model
from src.evaluation.scorecard import gain_table, probability_to_score, score_bands
from src.models.experiment import EXPERIMENTS_PATH
from src.models.gbdt import load_matrix

OUTPUT_DIR = resolve_path("dashboard/public/data")
SCORE_SAMPLE_SIZE = 20_000
N_SUBGRAPHS = 24
SUBGRAPH_MIN_SIZE = 6
SUBGRAPH_MAX_SIZE = 60
SEED = 42


def _write(name: str, payload) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  {name:22} {path.stat().st_size / 1024:8.1f} KB")


def export_metrics() -> None:
    """Hasil keempat model di test out-of-time, plus lift ablation."""
    experiments = pd.read_csv(EXPERIMENTS_PATH)
    tested = experiments[experiments["test_auc"].notna()].drop_duplicates("name", keep="last")

    metric_keys = ["auc", "ks", "pr_auc", "recall_at_1pct", "recall_at_5pct", "recall_at_10pct"]
    models = []
    for _, row in tested.iterrows():
        models.append(
            {
                "name": row["name"],
                "n_features": int(row["n_features"]),
                "has_graph": "graph" in row["name"],
                "has_c": "full" in row["name"],
                "val": {k: float(row[f"val_{k}"]) for k in metric_keys},
                "test": {k: float(row[f"test_{k}"]) for k in metric_keys},
            }
        )

    by_name = {m["name"]: m["test"] for m in models}
    lifts = {
        "lift_b1": {
            k: by_name["M2_graph_noC"][k] - by_name["M1_baseline_noC"][k] for k in metric_keys
        },
        "lift_b2": {
            k: by_name["M4_graph_full"][k] - by_name["M3_baseline_full"][k] for k in metric_keys
        },
        "value_of_c": {
            k: by_name["M3_baseline_full"][k] - by_name["M1_baseline_noC"][k] for k in metric_keys
        },
    }
    lifts["overlap"] = {k: lifts["lift_b1"][k] - lifts["lift_b2"][k] for k in metric_keys}

    _write(
        "metrics.json",
        {
            "models": models,
            "lifts": lifts,
            "decomposition": {
                "note": "validation saja — test sudah dikunci untuk 4 model di atas",
                "rows": [
                    {"name": "B2 (baseline)", "n_features": 447, "auc": 0.9031, "ks": 0.6435},
                    {"name": "B2 + Level 1-2", "n_features": 473, "auc": 0.9261, "ks": 0.7096},
                    {"name": "B2 + Level 3", "n_features": 453, "auc": 0.9067, "ks": 0.6557},
                    {"name": "B2 + semua graph", "n_features": 479, "auc": 0.9081, "ks": 0.6630},
                ],
            },
        },
    )


def export_scoring_artifacts(matrix: pd.DataFrame, masks: pd.DataFrame) -> None:
    """Gain curve, band skor, sampel skor, dan sensitivitas biaya dari M3."""
    cfg = load_config("evaluation")
    booster, features = train_final_model(cfg, matrix, masks)
    test_mask = masks["is_test"].to_numpy()
    y_test = matrix.loc[test_mask, "isFraud"].to_numpy()
    amounts = matrix.loc[test_mask, "TransactionAmt"].to_numpy()
    pred = predict(booster, matrix, features, test_mask)

    sc = cfg["scorecard"]
    params = {k: sc[k] for k in ("pdo", "anchor_score", "anchor_odds", "score_min", "score_max")}
    scores = probability_to_score(pred, **params)

    gains = gain_table(y_test, pred, n_points=100)
    _write(
        "gain_curve.json",
        {
            "review_rate": gains["review_rate"].round(4).tolist(),
            "recall": gains["recall"].round(4).tolist(),
            "lift": gains["lift"].round(3).tolist(),
            "precision": gains["precision"].round(4).tolist(),
        },
    )

    bands = score_bands(scores, y_test, sc["n_bands"])
    _write(
        "score_bands.json",
        [
            {
                "band": int(r["band"]),
                "n": int(r["n"]),
                "n_fraud": int(r["n_fraud"]),
                "fraud_rate": round(float(r["fraud_rate"]), 5),
                "score_min": round(float(r["score_min"]), 1),
                "score_max": round(float(r["score_max"]), 1),
                "cum_pct_fraud": round(float(r["cum_pct_fraud"]), 4),
            }
            for _, r in bands.iterrows()
        ],
    )

    # Sampel untuk slider threshold interaktif. Stratified supaya fraud yang jarang
    # tetap terwakili cukup untuk menghitung recall secara stabil.
    rng = np.random.default_rng(SEED)
    fraud_idx = np.flatnonzero(y_test == 1)
    good_idx = np.flatnonzero(y_test == 0)
    n_good = min(len(good_idx), SCORE_SAMPLE_SIZE - len(fraud_idx))
    keep = np.concatenate([fraud_idx, rng.choice(good_idx, n_good, replace=False)])
    keep.sort()

    _write(
        "score_sample.json",
        {
            "sampling": {
                "n_total_test": int(len(y_test)),
                "n_sampled": int(len(keep)),
                "n_fraud_sampled": int(len(fraud_idx)),
                "good_weight": round(float(len(good_idx) / n_good), 4),
                "note": "fraud diambil seluruhnya, non-fraud disubsample; "
                "pakai good_weight untuk ekstrapolasi",
            },
            "score": np.round(scores[keep], 1).tolist(),
            "prob": np.round(pred[keep], 5).tolist(),
            "label": y_test[keep].astype(int).tolist(),
            "amount": np.round(amounts[keep], 2).tolist(),
        },
    )

    bus = cfg["business"]
    summary, _ = sensitivity_analysis(
        y_test, pred, amounts, bus["cost_fp_scenarios"], bus["n_thresholds"], bus["days_in_period"]
    )
    _write(
        "cost_sensitivity.json",
        {
            "baseline_cost": round(float(baseline_cost(y_test, amounts)), 2),
            "days_in_period": bus["days_in_period"],
            "scenarios": [
                {
                    "cost_fp": float(r["cost_fp"]),
                    "optimal_threshold": round(float(r["optimal_threshold"]), 5),
                    "review_rate": round(float(r["review_rate"]), 4),
                    "recall": round(float(r["recall"]), 4),
                    "total_cost": round(float(r["total_cost"]), 2),
                    "savings_per_month": round(float(r["savings_per_month"]), 2),
                    "pct_cost_reduction": round(float(r["pct_cost_reduction"]), 4),
                }
                for _, r in summary.iterrows()
            ],
        },
    )


def _build_subgraph(frame: pd.DataFrame, column: str, value, rng) -> dict | None:
    """Subgraph transaksi-atribut di sekitar satu nilai atribut.

    NetworkX dipakai HANYA untuk layout subgraph kecil (<60 node), sesuai aturan
    project — komputasi graph skala penuh memakai scipy.sparse.
    """
    rows = frame[frame[column] == value]
    if not SUBGRAPH_MIN_SIZE <= len(rows) <= SUBGRAPH_MAX_SIZE:
        return None

    graph = nx.Graph()
    hub = f"{column}={value}"
    graph.add_node(hub, kind="attribute")

    attribute_columns = [c for c in ("card1", "addr1", "DeviceInfo", "P_emaildomain") if c in frame]
    for _, row in rows.iterrows():
        tx = f"tx{int(row['TransactionID'])}"
        graph.add_node(
            tx, kind="transaction", fraud=int(row["isFraud"]), amount=float(row["TransactionAmt"])
        )
        graph.add_edge(tx, hub)
        for col in attribute_columns:
            if col == column or pd.isna(row[col]):
                continue
            neighbour = f"{col}={row[col]}"
            if graph.degree(neighbour) if neighbour in graph else 0:
                graph.add_edge(tx, neighbour)
            elif rows[col].eq(row[col]).sum() > 1:  # hanya atribut yang dibagi
                graph.add_node(neighbour, kind="attribute")
                graph.add_edge(tx, neighbour)

    layout = nx.spring_layout(graph, seed=SEED, k=0.9, iterations=60)
    nodes = []
    for name, data in graph.nodes(data=True):
        x, y = layout[name]
        node = {
            "id": name,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "kind": data["kind"],
            "degree": graph.degree(name),
        }
        if data["kind"] == "transaction":
            node["fraud"] = data["fraud"]
            node["amount"] = round(data["amount"], 2)
        else:
            node["label"] = name.split("=", 1)[1][:24]
        nodes.append(node)

    return {
        "id": hub,
        "anchor_column": column,
        "n_transactions": int(len(rows)),
        "n_fraud": int(rows["isFraud"].sum()),
        "fraud_rate": round(float(rows["isFraud"].mean()), 4),
        "nodes": nodes,
        "edges": [{"source": u, "target": v} for u, v in graph.edges()],
    }


def export_subgraphs() -> None:
    """Subgraph pra-layout untuk network explorer, dipilih beragam fraud rate."""
    data_cfg = load_config("data")
    columns = [
        "TransactionID",
        "isFraud",
        "TransactionAmt",
        "card1",
        "addr1",
        "DeviceInfo",
        "P_emaildomain",
    ]
    frame = pd.read_parquet(resolve_path(data_cfg["paths"]["merged_train"]), columns=columns)
    rng = np.random.default_rng(SEED)

    subgraphs = []
    for column in ("card1", "DeviceInfo", "addr1"):
        stats = frame.groupby(column)["isFraud"].agg(["size", "mean"])
        eligible = stats[
            (stats["size"] >= SUBGRAPH_MIN_SIZE) & (stats["size"] <= SUBGRAPH_MAX_SIZE)
        ]
        if eligible.empty:
            continue
        # Ambil campuran: fraud rate tinggi (ring) dan rendah (kontrol).
        high = eligible.nlargest(N_SUBGRAPHS // 4, "mean").index.tolist()
        low = eligible[eligible["mean"] == 0].head(N_SUBGRAPHS // 8).index.tolist()
        for value in high + low:
            built = _build_subgraph(frame, column, value, rng)
            if built:
                subgraphs.append(built)

    subgraphs.sort(key=lambda s: -s["fraud_rate"])
    _write("subgraphs.json", subgraphs)


def main() -> None:
    print(f"menulis artefak dashboard ke {OUTPUT_DIR}")
    export_metrics()
    matrix = load_matrix()
    masks = load_split_masks()
    export_scoring_artifacts(matrix, masks)
    export_subgraphs()
    print("selesai")


if __name__ == "__main__":
    main()
