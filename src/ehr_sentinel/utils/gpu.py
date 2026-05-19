"""Local GPU detection with graceful CPU fallback."""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class DeviceInfo:
    """Device descriptor for XGBoost."""

    device: str            # "cuda" or "cpu"
    tree_method: str = "hist"
    n_gpus: int = 0
    reason: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"


@lru_cache(maxsize=1)
def detect_device(prefer: Optional[str] = None) -> DeviceInfo:
    """Detect best available device for XGBoost.

    Tries CUDA when ``prefer`` is None or ``"cuda"``; falls back to CPU on
    any failure. Result is cached for the process. Set the environment
    variable ``EHR_SENTINEL_FORCE_CPU=1`` to force CPU.
    """
    if os.environ.get("EHR_SENTINEL_FORCE_CPU", "").strip() == "1":
        return DeviceInfo(device="cpu", reason="forced via EHR_SENTINEL_FORCE_CPU")
    if prefer == "cpu":
        return DeviceInfo(device="cpu", reason="explicit prefer='cpu'")

    try:
        import xgboost as xgb  # noqa: F401
    except ImportError:
        warnings.warn("xgboost not installed; defaulting to CPU device.", RuntimeWarning, stacklevel=2)
        return DeviceInfo(device="cpu", reason="xgboost not installed")

    # Try a tiny CUDA training step to validate the GPU build
    try:
        import numpy as np
        import xgboost as xgb
        rng = np.random.default_rng(0)
        X = rng.normal(size=(32, 4)).astype("float32")
        y = rng.normal(size=32).astype("float32")
        dtrain = xgb.DMatrix(X, label=y)
        xgb.train(
            {"tree_method": "hist", "device": "cuda", "verbosity": 0},
            dtrain,
            num_boost_round=1,
        )
        n_gpus = 1
        try:
            import subprocess
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=2)
            n_gpus = max(1, len([l for l in out.stdout.splitlines() if l.startswith("GPU ")]))
        except Exception:
            pass
        return DeviceInfo(device="cuda", tree_method="hist", n_gpus=n_gpus, reason="cuda probe succeeded")
    except Exception as e:
        return DeviceInfo(device="cpu", tree_method="hist", reason=f"cuda unavailable: {type(e).__name__}")
