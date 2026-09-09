"""Training LightGBM dengan early stopping pada validation set temporal.

Pemilihan blok fitur terjadi di sini (bukan saat membangun matriks) supaya
perbandingan antar model apple-to-apple: pipeline preprocessing identik, yang
berbeda hanya kolom yang dipakai.
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import pandas as pd
import xgboost as xgb

from src.config import load_config, resolve_path
from src.data.split import load_split_masks
from src.evaluation.metrics import evaluate, format_metrics
from src.models.experiment import log_experiment

META_COLS = ("TransactionID", "TransactionDT", "isFraud")

# Prefiks kolom yang dihasilkan tiap blok fitur di src/features/build.py.
BLOCK_PATTERNS: dict[str, tuple[str, ...]] = {
    "base": ("TransactionAmt", "dist1", "dist2", "has_identity"),
    "derived": ("hour", "dayofweek", "amt_log", "amt_decimal"),
    "label_encode": ("le_",),
    "frequency_encode": ("freq_",),
    "aggregations": ("TransactionAmt_",),
    "D": tuple(f"D{i}" for i in range(1, 16)),
    "C": tuple(f"C{i}" for i in range(1, 15)),
    "V": ("V",),
    "id": ("id_",),
}


def select_features(columns: list[str], blocks: list[str]) -> list[str]:
    """Pilih kolom sesuai daftar blok. Kolom meta selalu dikecualikan."""
    exact = {"base", "derived", "D", "C"}
    selected = []
    for col in columns:
        if col in META_COLS:
            continue
        for block in blocks:
            patterns = BLOCK_PATTERNS[block]
            hit = col in patterns if block in exact else col.startswith(patterns)
            if hit:
                selected.append(col)
                break
    return selected


def load_matrix(features_config: str = "features") -> pd.DataFrame:
    path = resolve_path(load_config(features_config)["paths"]["baseline_matrix"])
    return pd.read_parquet(path)


def _train_lightgbm(X_tr, y_tr, X_va, y_va, cfg):
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    params = {**cfg["params"], "seed": cfg["seed"], "num_threads": 0}

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg["train"]["num_boost_round"],
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(cfg["train"]["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(cfg["train"]["log_every"]),
        ],
    )
    pred = booster.predict(X_va, num_iteration=booster.best_iteration)
    return booster, pred, params


def _train_xgboost(X_tr, y_tr, X_va, y_va, cfg):
    """Backend XGBoost.

    Dipakai sebagai backend utama karena binary LightGBM crash di mesin ini
    (access violation pada LGBM_DatasetSetField, terjadi juga pada data acak di
    venv bersih). XGBoost menangani NaN secara native seperti LightGBM, sehingga
    keputusan "unseen entity -> NaN" tetap berlaku sama.
    """
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    params = {**cfg["params_xgb"], "seed": cfg["seed"], "nthread": 0}

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg["train"]["num_boost_round"],
        evals=[(dval, "val")],
        early_stopping_rounds=cfg["train"]["early_stopping_rounds"],
        verbose_eval=cfg["train"]["log_every"],
    )
    pred = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    return booster, pred, params


BACKENDS = {"lightgbm": _train_lightgbm, "xgboost": _train_xgboost}
DEFAULT_BACKEND = "xgboost"


def train_model(model_config: str, features_config: str = "features", log: bool = True):
    """Latih satu model dan catat metrik validation. Test set TIDAK disentuh."""
    cfg = load_config(f"model/{model_config}")
    matrix = load_matrix(features_config)
    masks = load_split_masks()

    features = select_features(list(matrix.columns), cfg["feature_blocks"])
    tr, va = masks["is_train"].to_numpy(), masks["is_val"].to_numpy()
    y = matrix["isFraud"].to_numpy()

    backend = BACKENDS[cfg.get("backend", DEFAULT_BACKEND)]
    booster, val_pred, params = backend(
        matrix.loc[tr, features], y[tr], matrix.loc[va, features], y[va], cfg
    )
    val_metrics = evaluate(y[va], val_pred)

    print(f"\n=== {cfg['name']} ===")
    print(f"backend: {cfg.get('backend', DEFAULT_BACKEND)} | features: {len(features)}")
    print(f"best_iteration: {booster.best_iteration}")
    print(f"VAL  {format_metrics(val_metrics)}")

    if log:
        log_experiment(
            name=cfg["name"],
            feature_set="+".join(cfg["feature_blocks"]),
            params=params,
            val_metrics=val_metrics,
            n_features=len(features),
            best_iteration=booster.best_iteration,
            notes=cfg.get("notes", ""),
        )
    return booster, val_metrics, features


def save_model(booster, name: str):
    path = resolve_path(f"artifacts/{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(booster, lgb.Booster):
        booster.save_model(str(path), num_iteration=booster.best_iteration)
    else:
        booster.save_model(str(path))
    return path


def main():
    parser = argparse.ArgumentParser(description="Train a LightGBM model")
    parser.add_argument("--model", required=True, help="nama config di configs/model/")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    booster, _, _ = train_model(args.model)
    if args.save:
        print(f"model saved to {save_model(booster, args.model)}")


if __name__ == "__main__":
    main()
