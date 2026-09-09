"""Pipeline evaluasi risk-style Fase 5 pada M3 (baseline terbaik OOT).

M3 dilatih ulang dengan konfigurasi identik Fase 2/4 - bukan model baru. Prediksi
test yang dihasilkan sama dengan yang sudah dievaluasi di Fase 4, sehingga
menghitung scorecard/PSI/biaya di atasnya tidak membuka test set untuk kedua kali.

Peninjauan spw=5 dijalankan di VALIDATION saja. Kalau menang, hasilnya dilaporkan
sebagai temuan - bukan dipakai lalu dievaluasi di test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import load_config
from src.data.split import load_split_masks
from src.evaluation.business import baseline_cost, sensitivity_analysis
from src.evaluation.calibration import (
    calibration_summary,
    fit_isotonic,
    reliability_curve,
)
from src.evaluation.metrics import evaluate
from src.evaluation.scorecard import gain_table, probability_to_score, score_bands
from src.evaluation.stability import (
    characteristic_stability,
    population_stability_index,
    psi_detail,
    psi_verdict,
)
from src.models.gbdt import load_matrix, select_features
from src.viz.evaluation_plots import (
    plot_cost_sensitivity,
    plot_gain_lift,
    plot_psi,
    plot_reliability,
    plot_score_distribution,
)


def train_final_model(cfg: dict, matrix: pd.DataFrame, masks: pd.DataFrame, spw: float = 1.0):
    """Latih M3 dengan konfigurasi Fase 2/4. spw dapat di-override untuk peninjauan."""
    model_cfg = load_config(f"model/{cfg['model']}")
    features = select_features(list(matrix.columns), model_cfg["feature_blocks"])
    tr, va = masks["is_train"].to_numpy(), masks["is_val"].to_numpy()
    y = matrix["isFraud"].to_numpy()

    params = {**model_cfg["params_xgb"], "seed": model_cfg["seed"], "scale_pos_weight": spw}
    booster = xgb.train(
        params,
        xgb.DMatrix(matrix.loc[tr, features], label=y[tr]),
        num_boost_round=model_cfg["train"]["num_boost_round"],
        evals=[(xgb.DMatrix(matrix.loc[va, features], label=y[va]), "val")],
        early_stopping_rounds=model_cfg["train"]["early_stopping_rounds"],
        verbose_eval=False,
    )
    return booster, features


def predict(booster, matrix: pd.DataFrame, features: list[str], mask: np.ndarray) -> np.ndarray:
    dmatrix = xgb.DMatrix(matrix.loc[mask, features])
    return booster.predict(dmatrix, iteration_range=(0, booster.best_iteration + 1))


def review_scale_pos_weight(cfg: dict, matrix, masks) -> pd.DataFrame:
    """Apakah spw=5 + isotonic mengalahkan spw=1? VALIDATION SAJA.

    Isotonic di-fit pada paruh pertama validation dan dinilai pada paruh kedua,
    supaya kalibrasi tidak menilai dirinya sendiri.
    """
    va = masks["is_val"].to_numpy()
    y = matrix["isFraud"].to_numpy()
    y_val = y[va]
    half = len(y_val) // 2
    rows = []

    for spw in cfg["spw_review"]["weights"]:
        booster, features = train_final_model(cfg, matrix, masks, spw=spw)
        pred = predict(booster, matrix, features, va)

        calibrator = fit_isotonic(y_val[:half], pred[:half])
        holdout_y, holdout_raw = y_val[half:], pred[half:]
        holdout_cal = calibrator.predict(holdout_raw)

        for label, scores in (("raw", holdout_raw), ("isotonic", holdout_cal)):
            metrics = evaluate(holdout_y, scores)
            calib = calibration_summary(holdout_y, scores, cfg["calibration"]["n_bins"])
            rows.append(
                {
                    "scale_pos_weight": spw,
                    "calibration": label,
                    "auc": metrics["auc"],
                    "ks": metrics["ks"],
                    "pr_auc": metrics["pr_auc"],
                    **calib,
                }
            )
    return pd.DataFrame(rows)


def run():
    cfg = load_config("evaluation")
    matrix = load_matrix()
    masks = load_split_masks()
    y = matrix["isFraud"].to_numpy()
    tr, va, te = (masks[c].to_numpy() for c in ("is_train", "is_val", "is_test"))

    booster, features = train_final_model(cfg, matrix, masks)
    pred_train = predict(booster, matrix, features, tr)
    pred_val = predict(booster, matrix, features, va)
    pred_test = predict(booster, matrix, features, te)

    print(f"M3 direproduksi: TEST AUC {evaluate(y[te], pred_test)['auc']:.4f}")

    results = {"booster": booster, "features": features}
    results |= _scorecard_section(cfg, y, te, pred_train, pred_test)
    results |= _business_section(cfg, matrix, y, te, pred_test)
    results |= _stability_section(cfg, matrix, features, tr, te, pred_train, pred_test)
    results |= _calibration_section(cfg, y, va, te, pred_val, pred_test)

    print("\n=== PENINJAUAN scale_pos_weight (VALIDATION SAJA) ===")
    review = review_scale_pos_weight(cfg, matrix, masks)
    print(review.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    results["spw_review"] = review
    return results


def _scorecard_section(cfg, y, te, pred_train, pred_test) -> dict:
    sc = cfg["scorecard"]
    params = {k: sc[k] for k in ("pdo", "anchor_score", "anchor_odds", "score_min", "score_max")}
    scores_test = probability_to_score(pred_test, **params)
    scores_train = probability_to_score(pred_train, **params)

    bands = score_bands(scores_test, y[te], sc["n_bands"])
    gains = gain_table(y[te], pred_test)
    plot_score_distribution(scores_test, y[te])
    plot_gain_lift(gains)

    print("\n=== SCORECARD (test out-of-time) ===")
    print(f"skor fraud    : mean {scores_test[y[te] == 1].mean():.0f}")
    print(f"skor non-fraud: mean {scores_test[y[te] == 0].mean():.0f}")
    print(bands.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return {
        "scores_test": scores_test,
        "scores_train": scores_train,
        "bands": bands,
        "gains": gains,
    }


def _business_section(cfg, matrix, y, te, pred_test) -> dict:
    bus = cfg["business"]
    amounts = matrix.loc[te, "TransactionAmt"].to_numpy()
    summary, curves = sensitivity_analysis(
        y[te],
        pred_test,
        amounts,
        bus["cost_fp_scenarios"],
        bus["n_thresholds"],
        bus["days_in_period"],
    )
    plot_cost_sensitivity(curves, summary)

    print("\n=== SENSITIVITAS BIAYA FALSE POSITIVE ===")
    print(f"biaya tanpa model (semua fraud lolos): ${baseline_cost(y[te], amounts):,.0f}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    return {"cost_summary": summary, "cost_curves": curves}


def _stability_section(cfg, matrix, features, tr, te, pred_train, pred_test) -> dict:
    st = cfg["stability"]
    psi = population_stability_index(pred_train, pred_test, st["n_bins"])
    detail = psi_detail(pred_train, pred_test, st["n_bins"])

    top_features = features[: st["top_features_for_csi"]]
    csi = characteristic_stability(matrix.loc[tr], matrix.loc[te], top_features, st["n_bins"])
    plot_psi(detail, psi, csi)

    print(f"\n=== STABILITAS ===\nPSI skor train->test: {psi:.4f} ({psi_verdict(psi)})")
    print(csi.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return {"psi": psi, "psi_detail": detail, "csi": csi}


def _calibration_section(cfg, y, va, te, pred_val, pred_test) -> dict:
    n_bins = cfg["calibration"]["n_bins"]
    calibrator = fit_isotonic(y[va], pred_val)
    calibrated = calibrator.predict(pred_test)

    before = calibration_summary(y[te], pred_test, n_bins)
    after = calibration_summary(y[te], calibrated, n_bins)
    plot_reliability(
        reliability_curve(y[te], pred_test, n_bins),
        reliability_curve(y[te], calibrated, n_bins),
        before["ece"],
        after["ece"],
    )

    print("\n=== KALIBRASI (isotonic di-fit di validation) ===")
    print(
        pd.DataFrame([before, after], index=["sebelum", "isotonic"]).to_string(
            float_format=lambda x: f"{x:.5f}"
        )
    )
    print(
        f"AUC sebelum {evaluate(y[te], pred_test)['auc']:.4f} | "
        f"sesudah {evaluate(y[te], calibrated)['auc']:.4f}"
    )
    return {"calib_before": before, "calib_after": after, "pred_test_calibrated": calibrated}


if __name__ == "__main__":
    run()
