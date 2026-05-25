"""GPU-aware XGBoost trainer for gap / LOS rhythm prediction.

Defaults reflect the published GPU-tuned configuration:
n_estimators=1200, max_depth=5, lr=0.01, subsample=0.7,
colsample_bytree=0.6, min_child_weight=20, reg_alpha=2.0, reg_lambda=5.0,
gamma=0.1.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ehr_sentinel.utils.gpu import DeviceInfo, detect_device


DEFAULT_PARAMS: dict = {
    "n_estimators": 1200,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 20,
    "reg_alpha": 2.0,
    "reg_lambda": 5.0,
    "gamma": 0.1,
    "objective": "reg:squarederror",
    "tree_method": "hist",
}


@dataclass
class TrainResult:
    model: object
    feature_names: list[str]
    target: str
    device: str
    n_train: int
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class XGBTrainer:
    """Train XGBoost regressors for gap / LOS prediction with GPU autodetect."""

    def __init__(
        self,
        params: Optional[dict] = None,
        device_info: Optional[DeviceInfo] = None,
        random_state: int = 20260513,
    ) -> None:
        try:
            import xgboost as xgb  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "XGBTrainer requires xgboost. Install with: pip install 'ehr-sentinel[lgdi]'"
            ) from e
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.device_info = device_info or detect_device()
        self.random_state = random_state

    # ── Public API ──────────────────────────────────────────────────────
    def train_gap_model(self, X: pd.DataFrame, y: pd.Series, **fit_kwargs) -> TrainResult:
        return self._train(X, y, target="gap", **fit_kwargs)

    def train_los_model(self, X: pd.DataFrame, y: pd.Series, **fit_kwargs) -> TrainResult:
        return self._train(X, y, target="los", **fit_kwargs)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, list[float]]:
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_absolute_error, r2_score

        n_splits = max(2, min(n_splits, groups.nunique()))
        gkf = GroupKFold(n_splits=n_splits)
        r2s: list[float] = []
        maes: list[float] = []
        train_r2s: list[float] = []
        for tr, te in gkf.split(X, y, groups=groups):
            model = self._make_estimator()
            model.fit(X.iloc[tr], y.iloc[tr])
            pred_te = model.predict(X.iloc[te])
            pred_tr = model.predict(X.iloc[tr])
            r2s.append(float(r2_score(y.iloc[te], pred_te)))
            maes.append(float(mean_absolute_error(y.iloc[te], pred_te)))
            train_r2s.append(float(r2_score(y.iloc[tr], pred_tr)))
        return {"r2_test": r2s, "mae_test": maes, "r2_train": train_r2s}

    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict(X))

    # ── Internal ────────────────────────────────────────────────────────
    def _make_estimator(self):
        import xgboost as xgb
        params = dict(self.params)
        params["device"] = self.device_info.device
        params["random_state"] = self.random_state
        params["verbosity"] = 0
        return xgb.XGBRegressor(**params)

    def _train(self, X: pd.DataFrame, y: pd.Series, target: str, **fit_kwargs) -> TrainResult:
        from sklearn.metrics import mean_absolute_error, r2_score

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X) < 10:
            warnings.warn(f"Very small training set: n={len(X)}", RuntimeWarning, stacklevel=2)

        model = self._make_estimator()
        try:
            model.fit(X, y, **fit_kwargs)
        except Exception as e:
            if self.device_info.is_gpu:
                warnings.warn(
                    f"GPU training failed ({type(e).__name__}); retrying on CPU.",
                    RuntimeWarning, stacklevel=2,
                )
                self.device_info = DeviceInfo(device="cpu", reason="gpu retry fallback")
                model = self._make_estimator()
                model.fit(X, y, **fit_kwargs)
            else:
                raise

        pred = model.predict(X)
        return TrainResult(
            model=model,
            feature_names=list(X.columns),
            target=target,
            device=self.device_info.device,
            n_train=len(X),
            metrics={
                "train_r2": float(r2_score(y, pred)),
                "train_mae": float(mean_absolute_error(y, pred)),
            },
        )
