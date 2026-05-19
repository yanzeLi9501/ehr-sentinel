"""Optuna hyperparameter tuning for XGBoost rhythm models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ehr_sentinel.models.trainer import XGBTrainer
from ehr_sentinel.utils.gpu import DeviceInfo, detect_device


@dataclass
class TuneResult:
    best_params: dict
    best_value: float
    n_trials: int
    objective: str
    history: list[dict]


class XGBTuner:
    """Optuna TPE tuner. Default objective: maximize CV R²."""

    def __init__(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        device_info: Optional[DeviceInfo] = None,
        random_state: int = 42,
    ) -> None:
        try:
            import optuna  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "XGBTuner requires optuna. Install with: pip install 'ehr-sentinel[lgdi]'"
            ) from e
        self.n_trials = n_trials
        self.timeout = timeout
        self.device_info = device_info or detect_device()
        self.random_state = random_state

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_splits: int = 5,
    ) -> TuneResult:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.trial.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            }
            trainer = XGBTrainer(params=params, device_info=self.device_info,
                                 random_state=self.random_state)
            cv = trainer.cross_validate(X, y, groups=groups, n_splits=n_splits)
            return float(np.mean(cv["r2_test"]))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=False)

        return TuneResult(
            best_params=dict(study.best_params),
            best_value=float(study.best_value),
            n_trials=len(study.trials),
            objective="cv_r2_mean",
            history=[{"number": t.number, "value": t.value, "params": t.params} for t in study.trials],
        )

    def tune_ppv_targeted(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        ppv_at: int = 10,
        n_splits: int = 5,
        custom_metric: Optional[Callable] = None,
    ) -> TuneResult:
        """Tune to maximize PPV at the top-k% gap residuals (a proxy for
        outbreak detection precision)."""
        import optuna
        from sklearn.model_selection import GroupKFold
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def ppv_top_k(y_true: np.ndarray, y_pred: np.ndarray, k_pct: int = 10) -> float:
            n = len(y_true)
            k = max(1, int(n * k_pct / 100))
            resid = y_pred - y_true
            top_idx = np.argsort(-resid)[:k]
            # "positive" = real residual above median absolute deviation
            mad = np.median(np.abs(y_true - np.median(y_true))) or 1.0
            positives = np.abs(y_true - np.median(y_true)) > 1.5 * mad
            return float(positives[top_idx].mean())

        def objective(trial: "optuna.trial.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            }
            trainer = XGBTrainer(params=params, device_info=self.device_info,
                                 random_state=self.random_state)
            gkf = GroupKFold(n_splits=max(2, min(n_splits, groups.nunique())))
            scores: list[float] = []
            for tr, te in gkf.split(X, y, groups=groups):
                m = trainer._make_estimator()
                m.fit(X.iloc[tr], y.iloc[tr])
                pred = m.predict(X.iloc[te])
                metric = custom_metric or ppv_top_k
                scores.append(metric(y.iloc[te].to_numpy(), pred, ppv_at))
            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=False)
        return TuneResult(
            best_params=dict(study.best_params),
            best_value=float(study.best_value),
            n_trials=len(study.trials),
            objective=f"ppv_top_{ppv_at}",
            history=[{"number": t.number, "value": t.value, "params": t.params} for t in study.trials],
        )
