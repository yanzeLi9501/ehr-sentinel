"""Alert-evaluation metrics: PPV, sensitivity, specificity, threshold sweep."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvalResult:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    ppv: float
    sensitivity: float
    specificity: float
    f1: float
    false_alarm_rate: float


class AlertEvaluator:
    """Evaluate alert series against a known ground-truth outbreak label."""

    def __init__(self, positive_label: int = 1) -> None:
        self.positive_label = positive_label

    def evaluate(self, alerts: np.ndarray, truth: np.ndarray) -> EvalResult:
        alerts = np.asarray(alerts).astype(int)
        truth = np.asarray(truth).astype(int)
        tp = int(((alerts == 1) & (truth == 1)).sum())
        fp = int(((alerts == 1) & (truth == 0)).sum())
        tn = int(((alerts == 0) & (truth == 0)).sum())
        fn = int(((alerts == 0) & (truth == 1)).sum())
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return EvalResult(
            threshold=float("nan"), tp=tp, fp=fp, tn=tn, fn=fn,
            ppv=float(ppv), sensitivity=float(sens), specificity=float(spec),
            f1=float(f1), false_alarm_rate=float(far),
        )

    def threshold_sweep(
        self,
        scores: pd.Series | np.ndarray,
        truth: pd.Series | np.ndarray,
        n_thresholds: int = 50,
    ) -> pd.DataFrame:
        scores = np.asarray(scores, dtype=float)
        truth = np.asarray(truth).astype(int)
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds))
        rows: list[dict] = []
        for t in thresholds:
            alerts = (scores >= t).astype(int)
            r = self.evaluate(alerts, truth)
            d = r.__dict__ | {"threshold": float(t)}
            rows.append(d)
        return pd.DataFrame(rows)

    @staticmethod
    def ppv_at_top_k(scores: np.ndarray, truth: np.ndarray, k_pct: int = 10) -> float:
        scores = np.asarray(scores, dtype=float)
        truth = np.asarray(truth).astype(int)
        n = len(scores)
        k = max(1, int(n * k_pct / 100))
        top = np.argsort(-scores)[:k]
        return float(truth[top].mean()) if k > 0 else 0.0
