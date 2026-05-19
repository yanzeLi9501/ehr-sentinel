"""LOS–Gap Deviation Index (LGDI).

LGDI(t) = S_target(t) − mean(S_other_groups(t))

where S_group(t) is a per-group MASE-style residual scoring of the
admission rhythm in week t, computed against a baseline window stored at
``fit()`` time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LGDIResult:
    timeline: pd.DataFrame                 # week, group, S
    lgdi: pd.DataFrame                     # week, lgdi
    baseline_stats: dict[str, dict[str, float]] = field(default_factory=dict)


class LGDIComputer:
    """Compute LGDI per group / per week.

    The "rhythm" residual for each admission is the squared error of an
    XGBoost gap-predictor's residual (or a plain z-score of the gap when
    no model is provided). Per-group, per-week mean residuals are then
    scaled by the baseline-window mean absolute residual (MASE-style)
    before differencing.
    """

    def __init__(
        self,
        target_group: str = "Respiratory",
        baseline_start: Optional[str] = None,
        baseline_end: Optional[str] = None,
        min_admissions: int = 5,
    ) -> None:
        self.target_group = target_group
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end
        self.min_admissions = int(min_admissions)
        self._baseline_mean_abs: dict[str, float] = {}

    # ── Residuals ──────────────────────────────────────────────────────
    @staticmethod
    def _finite(values: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    @staticmethod
    def compute_residuals(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=float)
        if y_pred is None:
            finite = y_true[np.isfinite(y_true)]
            if len(finite) == 0:
                return np.zeros_like(y_true, dtype=float)
            mu = float(finite.mean())
            sd = float(finite.std()) or 1.0
            return (y_true - mu) / sd
        return y_true - np.asarray(y_pred, dtype=float)

    # ── Baseline fit ────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        residual_col: str,
        group_col: str = "comorbidity_group",
        date_col: str = "admission_date",
    ) -> "LGDIComputer":
        d = pd.to_datetime(df[date_col], errors="coerce")
        mask = pd.Series(True, index=df.index)
        if self.baseline_start:
            mask &= d >= pd.Timestamp(self.baseline_start)
        if self.baseline_end:
            mask &= d <= pd.Timestamp(self.baseline_end)
        base = df.loc[mask]
        for group, sub in base.groupby(group_col):
            if pd.isna(group) or len(sub) < self.min_admissions:
                continue
            finite = self._finite(sub[residual_col])
            if len(finite) == 0:
                continue
            self._baseline_mean_abs[group] = float(np.abs(finite).mean()) or 1.0
        return self

    # ── Group MASE ──────────────────────────────────────────────────────
    def group_mase(
        self,
        df: pd.DataFrame,
        residual_col: str,
        group_col: str = "comorbidity_group",
        date_col: str = "admission_date",
    ) -> pd.DataFrame:
        d = pd.to_datetime(df[date_col], errors="coerce")
        weeks = d.dt.to_period("W")
        rows: list[dict] = []
        for (week, group), sub in df.groupby([weeks, group_col]):
            if pd.isna(group) or len(sub) < self.min_admissions:
                continue
            finite = self._finite(sub[residual_col])
            mae = float(np.abs(finite).mean()) if len(finite) else 0.0
            denom = self._baseline_mean_abs.get(group, 1.0) or 1.0
            rows.append({"week": week.start_time, "group": group, "S": mae / denom, "n": int(len(sub))})
        return pd.DataFrame(rows)

    # ── LGDI ────────────────────────────────────────────────────────────
    def compute_lgdi(self, mase_df: pd.DataFrame) -> pd.DataFrame:
        if mase_df.empty:
            return pd.DataFrame(columns=["week", "lgdi"])
        pivot = mase_df.pivot_table(index="week", columns="group", values="S")
        if self.target_group not in pivot.columns:
            return pd.DataFrame(columns=["week", "lgdi"])
        others = pivot.drop(columns=[self.target_group])
        lgdi = pivot[self.target_group] - others.mean(axis=1, skipna=True)
        return pd.DataFrame({"week": lgdi.index, "lgdi": lgdi.values}).reset_index(drop=True)

    # ── Convenience: one-shot ───────────────────────────────────────────
    def run(
        self,
        df: pd.DataFrame,
        y_col: str = "gap",
        pred_col: Optional[str] = None,
        group_col: str = "comorbidity_group",
        date_col: str = "admission_date",
    ) -> LGDIResult:
        df = df.copy()
        y_true = df[y_col].to_numpy()
        y_pred = df[pred_col].to_numpy() if pred_col and pred_col in df.columns else None
        df["_resid"] = self.compute_residuals(y_true, y_pred)
        if not self._baseline_mean_abs:
            self.fit(df, residual_col="_resid", group_col=group_col, date_col=date_col)
        mase = self.group_mase(df, residual_col="_resid", group_col=group_col, date_col=date_col)
        lgdi = self.compute_lgdi(mase)
        return LGDIResult(timeline=mase, lgdi=lgdi, baseline_stats={
            g: {"mean_abs_residual": v} for g, v in self._baseline_mean_abs.items()
        })

    # ── Rolling windows ─────────────────────────────────────────────────
    @staticmethod
    def rolling_windows(timeline: pd.DataFrame, window: int = 4, value_col: str = "lgdi") -> pd.DataFrame:
        if timeline.empty:
            return timeline
        out = timeline.sort_values("week").copy()
        out[f"{value_col}_rmean{window}"] = out[value_col].rolling(window, min_periods=1).mean()
        out[f"{value_col}_rstd{window}"] = out[value_col].rolling(window, min_periods=2).std()
        return out
