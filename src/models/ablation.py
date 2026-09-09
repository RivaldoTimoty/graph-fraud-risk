"""Ablation study Fase 4a: mengukur nilai tambah graph features.

URUTAN YANG MENGIKAT — dirancang supaya test set tidak menjadi alat seleksi:

  1. Latih 4 model, evaluasi HANYA di validation
  2. SHAP pada M4 di validation -> putuskan nasib 2-hop
  3. Kunci konfigurasi final (latih ulang bila 2-hop digugurkan)
  4. Buka test set SEKALI untuk keempat model
  5. Laporkan apa adanya, tanpa tuning setelahnya

Membalik urutan ini (menggugurkan fitur setelah melihat test) adalah bentuk halus
tuning-on-test dan membuat angka test tidak sah.
"""

from __future__ import annotations

import argparse

import pandas as pd
import xgboost as xgb

from src.config import load_config
from src.data.split import load_split_masks
from src.evaluation.metrics import evaluate, format_metrics
from src.evaluation.shap_analysis import (
    compute_shap,
    group_summary,
    importance_table,
    plot_summary,
)
from src.models.experiment import log_experiment
from src.models.gbdt import load_matrix, select_features

MODELS = {
    "M1_baseline_noC": "lgbm_baseline_noC",
    "M2_graph_noC": "xgb_graph_noC",
    "M3_baseline_full": "lgbm_baseline_full",
    "M4_graph_full": "xgb_graph_full",
}

# Ambang gugur 2-hop, DITETAPKAN DI MUKA sebelum melihat angka apa pun.
TWO_HOP_FEATURE = "graph_nb_fraud_rate_2hop"
TWO_HOP_MIN_RATIO = 0.20  # mean|SHAP| 2hop harus >= 20% dari 1hop
TWO_HOP_MAX_RANK = 50
SHAP_SAMPLE = 50_000


def train_one(model_config: str, matrix: pd.DataFrame, masks: pd.DataFrame, excluded: tuple):
    """Latih satu model. Test set TIDAK disentuh di fungsi ini."""
    cfg = load_config(f"model/{model_config}")
    features = select_features(list(matrix.columns), cfg["feature_blocks"], excluded)

    tr, va = masks["is_train"].to_numpy(), masks["is_val"].to_numpy()
    y = matrix["isFraud"].to_numpy()

    dtrain = xgb.DMatrix(matrix.loc[tr, features], label=y[tr])
    dval = xgb.DMatrix(matrix.loc[va, features], label=y[va])
    params = {**cfg["params_xgb"], "seed": cfg["seed"], "nthread": 0}

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg["train"]["num_boost_round"],
        evals=[(dval, "val")],
        early_stopping_rounds=cfg["train"]["early_stopping_rounds"],
        verbose_eval=False,
    )
    val_pred = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    return booster, features, evaluate(y[va], val_pred), params


def decide_two_hop(table: pd.DataFrame) -> tuple[bool, str]:
    """Putuskan apakah 2-hop digugurkan, berdasarkan SHAP di validation.

    Kriteria ditetapkan sebelum angka dilihat, supaya tidak ada ruang untuk
    merasionalisasi hasil apa pun yang keluar.
    """
    ranked = table.reset_index(drop=True)
    row = ranked.index[ranked["feature"] == TWO_HOP_FEATURE]
    if len(row) == 0:
        return False, "2-hop tidak ada di feature set"

    rank = int(row[0]) + 1
    shap_2hop = float(ranked.loc[row[0], "mean_abs_shap"])
    one_hop = ranked.loc[ranked["feature"] == "graph_nb_fraud_rate_1hop", "mean_abs_shap"]
    shap_1hop = float(one_hop.iloc[0]) if len(one_hop) else 0.0
    ratio = shap_2hop / shap_1hop if shap_1hop > 0 else 0.0

    reason = f"rank {rank}, mean|SHAP| {shap_2hop:.5f} " f"({ratio:.1%} dari 1-hop {shap_1hop:.5f})"
    drop = ratio < TWO_HOP_MIN_RATIO or rank > TWO_HOP_MAX_RANK
    return drop, reason


def run(shap_sample: int = SHAP_SAMPLE):
    matrix = load_matrix(with_graph=True)
    masks = load_split_masks()
    y = matrix["isFraud"].to_numpy()
    excluded: tuple[str, ...] = ()

    print("=" * 78)
    print("TAHAP 1-2: training + SHAP di VALIDATION. Test set belum disentuh.")
    print("=" * 78)

    results = {}
    for tag, config_name in MODELS.items():
        booster, features, metrics, params = train_one(config_name, matrix, masks, excluded)
        results[tag] = (booster, features, metrics, params, config_name)
        print(f"{tag:20} n_feat={len(features):4} VAL {format_metrics(metrics)}")

    booster, features, _, _, _ = results["M4_graph_full"]
    values, sample = compute_shap(
        booster, matrix.loc[masks["is_val"].to_numpy(), features], shap_sample, seed=42
    )
    table = importance_table(values, features)
    drop_2hop, reason = decide_two_hop(table)

    print(f"\n2-hop: {reason}")
    print(
        f"keputusan: {'GUGUR' if drop_2hop else 'DIPERTAHANKAN'} "
        f"(ambang: ratio >= {TWO_HOP_MIN_RATIO:.0%} dan rank <= {TWO_HOP_MAX_RANK})"
    )

    if drop_2hop:
        excluded = (TWO_HOP_FEATURE,)
        print("\nMelatih ulang 4 model tanpa 2-hop...")
        results = {}
        for tag, config_name in MODELS.items():
            booster, features, metrics, params = train_one(config_name, matrix, masks, excluded)
            results[tag] = (booster, features, metrics, params, config_name)
            print(f"{tag:20} n_feat={len(features):4} VAL {format_metrics(metrics)}")
        booster, features, _, _, _ = results["M4_graph_full"]
        values, sample = compute_shap(
            booster, matrix.loc[masks["is_val"].to_numpy(), features], shap_sample, seed=42
        )
        table = importance_table(values, features)

    plot_path = plot_summary(values, sample, "M4_graph_full")
    print(f"\nSHAP summary -> {plot_path}")
    return matrix, masks, y, results, table, excluded, drop_2hop, reason


def open_test_set(matrix, masks, y, results, excluded, notes: str):
    """TAHAP 4: buka test set — SEKALI, untuk keempat model sekaligus."""
    print("\n" + "=" * 78)
    print("TAHAP 4: MEMBUKA TEST SET OUT-OF-TIME. Config sudah dikunci.")
    print("=" * 78)

    te = masks["is_test"].to_numpy()
    rows = []
    for tag, (booster, features, val_metrics, params, config_name) in results.items():
        dtest = xgb.DMatrix(matrix.loc[te, features])
        pred = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
        test_metrics = evaluate(y[te], pred)

        log_experiment(
            name=tag,
            feature_set="+".join(load_config(f"model/{config_name}")["feature_blocks"]),
            params=params,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            n_features=len(features),
            best_iteration=booster.best_iteration,
            notes=notes,
        )
        rows.append({"model": tag, "n_features": len(features), **test_metrics})
        print(f"{tag:20} TEST {format_metrics(test_metrics)}")
    return pd.DataFrame(rows)


def comparison_table(test_df: pd.DataFrame) -> pd.DataFrame:
    """Tabel lift: graph vs non-graph, plus tumpang-tindih C dengan graph."""
    idx = test_df.set_index("model")
    metrics = ["auc", "ks", "pr_auc", "recall_at_1pct", "recall_at_5pct", "recall_at_10pct"]

    lift_b1 = idx.loc["M2_graph_noC", metrics] - idx.loc["M1_baseline_noC", metrics]
    lift_b2 = idx.loc["M4_graph_full", metrics] - idx.loc["M3_baseline_full", metrics]
    c_value = idx.loc["M3_baseline_full", metrics] - idx.loc["M1_baseline_noC", metrics]

    return pd.DataFrame(
        {
            "M1_baseline_noC": idx.loc["M1_baseline_noC", metrics],
            "M2_graph_noC": idx.loc["M2_graph_noC", metrics],
            "M3_baseline_full": idx.loc["M3_baseline_full", metrics],
            "M4_graph_full": idx.loc["M4_graph_full", metrics],
            "Lift_B1 (M2-M1)": lift_b1,
            "Lift_B2 (M4-M3)": lift_b2,
            "Nilai C (M3-M1)": c_value,
            # Berapa banyak sinyal graph yang ternyata SUDAH ada di C-features.
            "Tumpang-tindih C<->graph": lift_b1 - lift_b2,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Fase 4a ablation study")
    parser.add_argument("--shap-sample", type=int, default=SHAP_SAMPLE)
    args = parser.parse_args()

    matrix, masks, y, results, table, excluded, dropped, reason = run(args.shap_sample)

    print("\n=== SHAP: top 20 fitur (M4, validation) ===")
    print(table.head(20).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    n_graph_top20 = table.head(20)["group"].isin(["graph", "uid"]).sum()
    print(f"\nFitur graph/uid di top 20: {n_graph_top20}")

    print("\n=== SHAP per kelompok fitur ===")
    print(group_summary(table).to_string(float_format=lambda x: f"{x:.5f}"))

    notes = f"fase4 ablation; 2hop {'digugurkan' if dropped else 'dipertahankan'} ({reason})"
    test_df = open_test_set(matrix, masks, y, results, excluded, notes)

    print("\n=== TABEL PERBANDINGAN (TEST SET OUT-OF-TIME) ===")
    print(comparison_table(test_df).to_string(float_format=lambda x: f"{x:+.4f}"))
    return table


if __name__ == "__main__":
    main()
