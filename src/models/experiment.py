"""Pencatat eksperimen ke artifacts/experiments.csv.

Setiap baris adalah satu training run. Hash hyperparameter membuat konfigurasi
yang identik dapat dikenali tanpa membandingkan seluruh dict.

Kolom `test_*` sengaja dibiarkan kosong sampai Fase 5 — test set out-of-time
hanya dibuka sekali. Jumlah baris dengan test_auc terisi adalah audit trail
berapa kali test set benar-benar disentuh.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from src.config import resolve_path

EXPERIMENTS_PATH = resolve_path("artifacts/experiments.csv")


def hyperparameter_hash(params: dict) -> str:
    """Hash stabil 8 karakter dari dict hyperparameter."""
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def log_experiment(
    name: str,
    feature_set: str,
    params: dict,
    val_metrics: dict,
    test_metrics: dict | None = None,
    n_features: int | None = None,
    best_iteration: int | None = None,
    notes: str = "",
) -> pd.DataFrame:
    """Tambahkan satu baris ke experiments.csv, buat file kalau belum ada."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "feature_set": feature_set,
        "param_hash": hyperparameter_hash(params),
        "n_features": n_features,
        "best_iteration": best_iteration,
        "scale_pos_weight": params.get("scale_pos_weight"),
        "notes": notes,
    }
    row.update({f"val_{k}": v for k, v in val_metrics.items()})
    if test_metrics:
        row.update({f"test_{k}": v for k, v in test_metrics.items()})

    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if EXPERIMENTS_PATH.exists():
        frame = pd.concat([pd.read_csv(EXPERIMENTS_PATH), frame], ignore_index=True)
    frame.to_csv(EXPERIMENTS_PATH, index=False)
    return frame
