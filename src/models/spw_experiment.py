"""Eksperimen scale_pos_weight: {1, 5, 27.6} pada baseline full.

Tujuan bukan mencari nilai terbaik, tapi mendokumentasikan bahwa reweighting
hampir tidak mengubah metrik berbasis ranking (AUC/KS) sementara merusak
kalibrasi - yang dibutuhkan Fase 5 untuk cost-based threshold.

Baseline resmi tetap scale_pos_weight=1.0 dan dipakai konsisten di semua fase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import load_config
from src.data.split import load_split_masks
from src.evaluation.metrics import evaluate
from src.models.experiment import log_experiment
from src.models.gbdt import load_matrix, select_features

WEIGHTS = (1.0, 5.0, 27.6)


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> float:
    """Expected Calibration Error: rata-rata |confidence - accuracy| berbobot bin.

    Model yang overconfident (khas setelah reweighting agresif) punya ECE tinggi
    meski AUC-nya tidak berubah.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            total += m.sum() * abs(y_prob[m].mean() - y_true[m].mean())
    return float(total / len(y_true))


def run(model_config: str = "lgbm_baseline_full") -> pd.DataFrame:
    cfg = load_config(f"model/{model_config}")
    matrix = load_matrix()
    masks = load_split_masks()

    features = select_features(list(matrix.columns), cfg["feature_blocks"])
    tr, va = masks["is_train"].to_numpy(), masks["is_val"].to_numpy()
    y = matrix["isFraud"].to_numpy()

    dtrain = xgb.DMatrix(matrix.loc[tr, features], label=y[tr])
    dval = xgb.DMatrix(matrix.loc[va, features], label=y[va])

    rows = []
    for spw in WEIGHTS:
        params = {**cfg["params_xgb"], "seed": cfg["seed"], "scale_pos_weight": spw}
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=cfg["train"]["num_boost_round"],
            evals=[(dval, "val")],
            early_stopping_rounds=cfg["train"]["early_stopping_rounds"],
            verbose_eval=False,
        )
        pred = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        metrics = evaluate(y[va], pred)
        metrics["ece"] = calibration_error(y[va], pred)
        metrics["mean_pred"] = float(pred.mean())

        log_experiment(
            name=f"spw_{spw}",
            feature_set="+".join(cfg["feature_blocks"]),
            params=params,
            val_metrics=metrics,
            n_features=len(features),
            best_iteration=booster.best_iteration,
            notes="eksperimen scale_pos_weight, bukan baseline resmi",
        )
        rows.append({"scale_pos_weight": spw, **metrics})
        print(
            f"spw={spw:5} AUC {metrics['auc']:.4f} KS {metrics['ks']:.4f} "
            f"PR-AUC {metrics['pr_auc']:.4f} ECE {metrics['ece']:.4f} "
            f"mean_pred {metrics['mean_pred']:.4f}"
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    val_fraud_rate = load_matrix()["isFraud"][load_split_masks()["is_val"]].mean()
    print(f"actual fraud rate (val): {val_fraud_rate:.4f}\n")
    run()
