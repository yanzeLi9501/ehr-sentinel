"""Model serialization: joblib payload + JSON sidecar metadata."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib

from ehr_sentinel import __version__


class ModelPersistence:
    """Save and load trained models with metadata."""

    @staticmethod
    def save(
        model: Any,
        path: str | Path,
        feature_names: list[str],
        target: str,
        metrics: Optional[dict[str, float]] = None,
        config_dict: Optional[dict] = None,
        extra_metadata: Optional[dict] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        meta = {
            "ehr_sentinel_version": __version__,
            "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "target": target,
            "feature_names": list(feature_names),
            "metrics": dict(metrics or {}),
            "config": dict(config_dict or {}),
            "extra": dict(extra_metadata or {}),
        }
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        return path

    @staticmethod
    def load(path: str | Path) -> tuple[Any, dict]:
        path = Path(path)
        model = joblib.load(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        return model, meta
