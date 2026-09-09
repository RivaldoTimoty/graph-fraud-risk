"""Config loading. Semua path dan parameter berasal dari YAML di configs/, tidak ada hardcode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str = "data") -> dict[str, Any]:
    """Baca configs/<name>.yaml sebagai dict."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Ubah path relatif di config menjadi absolut terhadap root repo."""
    return PROJECT_ROOT / relative
