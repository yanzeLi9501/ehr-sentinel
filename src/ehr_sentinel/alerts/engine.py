"""Configurable consensus / season / sustained / CUSUM / EWMA alert rules.

Simple rules
------------
- ``ConsensusRule``: ≥k comorbidity groups exceed 1.5 SD in the same week.
- ``SeasonFilter``: mask alerts outside epidemic season months.
- ``SustainedRule``: require alert to persist ≥n_weeks consecutive weeks.

Advanced rules (matching NC_revision/run_lgdi_optimized_strategies.py)
-----------------------------------------------------------------------
- ``CUSUMRule``: one-sided CUSUM, k=0.5σ, h=4σ on baseline LGDI.
- ``EWMARule``: EWMA control chart, λ=0.2, L=3.0 σ_z.
- ``SeasonalAdjustedRule``: per-ISO-month mean+threshold_sd·SD threshold.
- ``MultiScaleRule``: OR of 4-week LGDI ≥ threshold AND 2-week rolling ≥ threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class ConsensusRule:
    """Trigger an alert when at least ``k`` of the monitored group MASE
    scores in a given week exceed ``threshold_sd`` above their baseline
    mean.
    """

    k: int = 2
    threshold_sd: float = 1.5
    baseline_stats: dict[str, tuple[float, float]] = field(default_factory=dict)  # group → (mean, sd)

    def fit(self, mase_df: pd.DataFrame, group_col: str = "group", value_col: str = "S") -> "ConsensusRule":
        for group, sub in mase_df.groupby(group_col):
            mu = float(sub[value_col].mean())
            sd = float(sub[value_col].std()) or 1.0
            self.baseline_stats[group] = (mu, sd)
        return self

    def evaluate(self, mase_df: pd.DataFrame, group_col: str = "group", value_col: str = "S",
                 date_col: str = "week") -> pd.DataFrame:
        if mase_df.empty or not self.baseline_stats:
            return pd.DataFrame(columns=[date_col, "alert", "n_exceeding"])
        df = mase_df.copy()
        df["exceeds"] = df.apply(
            lambda row: (
                (row[value_col] - self.baseline_stats.get(row[group_col], (0.0, 1.0))[0])
                / (self.baseline_stats.get(row[group_col], (0.0, 1.0))[1] or 1.0)
                >= self.threshold_sd
            ),
            axis=1,
        ).astype(int)
        out = df.groupby(date_col)["exceeds"].sum().reset_index(name="n_exceeding")
        out["alert"] = (out["n_exceeding"] >= self.k).astype(int)
        return out[[date_col, "alert", "n_exceeding"]]


@dataclass
class SeasonFilter:
    """Mask alerts outside the configured epidemic season months."""

    months: list[int] = field(default_factory=lambda: list(range(1, 13)))

    def apply(self, alerts: pd.DataFrame, date_col: str = "week") -> pd.DataFrame:
        out = alerts.copy()
        d = pd.to_datetime(out[date_col], errors="coerce")
        in_season = d.dt.month.isin(self.months)
        out["alert"] = (out["alert"].astype(int) & in_season.astype(int))
        return out


@dataclass
class SustainedRule:
    """Require an alert to persist for ``n_weeks`` consecutive weeks."""

    n_weeks: int = 2

    def apply(self, alerts: pd.DataFrame, date_col: str = "week") -> pd.DataFrame:
        out = alerts.sort_values(date_col).reset_index(drop=True).copy()
        run = (out["alert"].astype(int)
               .groupby((out["alert"] != out["alert"].shift()).cumsum())
               .cumcount() + 1) * out["alert"].astype(int)
        out["alert_sustained"] = (run >= self.n_weeks).astype(int)
        return out


# ════════════════════════════════════════════════════════════════════════
# Advanced alert strategies (NC_revision run_lgdi_optimized_strategies.py)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CUSUMRule:
    """One-sided CUSUM control chart on LGDI.

    Alert when cumulative upper-sum C_t ≥ h·σ, where:
        C_t = max(0, C_{t−1} + (x_t − μ) − k·σ)
        k = k_sigma · σ   (reference value / allowance)
        h = h_sigma · σ   (decision interval / threshold)

    Default k_sigma=0.5, h_sigma=4.0 matches NC_revision optimised strategy.
    Fit on the baseline window; apply to the monitoring series.
    """

    k_sigma: float = 0.5   # allowance as fraction of baseline σ
    h_sigma: float = 4.0   # decision threshold as fraction of baseline σ
    # Set by fit()
    mu_: float = field(default=math.nan, init=False, repr=False)
    sd_: float = field(default=math.nan, init=False, repr=False)
    k_: float = field(default=math.nan, init=False, repr=False)
    h_: float = field(default=math.nan, init=False, repr=False)

    def fit(
        self,
        lgdi_series: pd.Series,
        baseline_start: str | None = None,
        baseline_end: str | None = None,
        date_index: pd.Series | None = None,
    ) -> "CUSUMRule":
        vals = lgdi_series
        if date_index is not None and (baseline_start or baseline_end):
            d = pd.to_datetime(date_index, errors="coerce")
            mask = pd.Series(True, index=lgdi_series.index)
            if baseline_start:
                mask &= d >= pd.Timestamp(baseline_start)
            if baseline_end:
                mask &= d <= pd.Timestamp(baseline_end)
            vals = lgdi_series[mask]
        vals = vals.dropna()
        self.mu_ = float(vals.mean()) if len(vals) else 0.0
        self.sd_ = float(vals.std()) if len(vals) > 1 else 1.0
        if self.sd_ <= 1e-9:
            self.sd_ = 1.0
        self.k_ = self.k_sigma * self.sd_
        self.h_ = self.h_sigma * self.sd_
        return self

    def apply(
        self, lgdi_series: pd.Series, date_col: str = "week"
    ) -> pd.DataFrame:
        """Return DataFrame with columns [week, lgdi, cusum, alert]."""
        if not math.isfinite(self.mu_):
            raise RuntimeError("Call fit() before apply()")
        cusum = 0.0
        rows: list[dict] = []
        for week, value in lgdi_series.items():
            v = float(value) if math.isfinite(float(value)) else self.mu_
            cusum = max(0.0, cusum + (v - self.mu_) - self.k_)
            rows.append({date_col: week, "lgdi": v, "cusum": cusum,
                         "alert": int(cusum >= self.h_)})
        return pd.DataFrame(rows)


@dataclass
class EWMARule:
    """EWMA (Exponentially Weighted Moving Average) control chart on LGDI.

    Alert when Z_t = λ·x_t + (1−λ)·Z_{t−1} ≥ μ + L·σ_z
    where σ_z = σ·√(λ/(2−λ)).

    Default λ=0.2, L=3.0 matches NC_revision optimised strategy.
    """

    lam: float = 0.2    # smoothing parameter λ ∈ (0, 1]
    L: float = 3.0      # control limit multiplier
    mu_: float = field(default=math.nan, init=False, repr=False)
    sd_: float = field(default=math.nan, init=False, repr=False)
    threshold_: float = field(default=math.nan, init=False, repr=False)

    def fit(
        self,
        lgdi_series: pd.Series,
        baseline_start: str | None = None,
        baseline_end: str | None = None,
        date_index: pd.Series | None = None,
    ) -> "EWMARule":
        vals = lgdi_series
        if date_index is not None and (baseline_start or baseline_end):
            d = pd.to_datetime(date_index, errors="coerce")
            mask = pd.Series(True, index=lgdi_series.index)
            if baseline_start:
                mask &= d >= pd.Timestamp(baseline_start)
            if baseline_end:
                mask &= d <= pd.Timestamp(baseline_end)
            vals = lgdi_series[mask]
        vals = vals.dropna()
        self.mu_ = float(vals.mean()) if len(vals) else 0.0
        sd = float(vals.std()) if len(vals) > 1 else 1.0
        if sd <= 1e-9:
            sd = 1.0
        self.sd_ = sd
        sigma_z = sd * math.sqrt(self.lam / (2.0 - self.lam))
        self.threshold_ = self.mu_ + self.L * sigma_z
        return self

    def apply(
        self, lgdi_series: pd.Series, date_col: str = "week"
    ) -> pd.DataFrame:
        """Return DataFrame with columns [week, lgdi, ewma, alert]."""
        if not math.isfinite(self.mu_):
            raise RuntimeError("Call fit() before apply()")
        z = self.mu_
        rows: list[dict] = []
        for week, value in lgdi_series.items():
            v = float(value) if math.isfinite(float(value)) else self.mu_
            z = self.lam * v + (1.0 - self.lam) * z
            rows.append({date_col: week, "lgdi": v, "ewma": z,
                         "alert": int(z >= self.threshold_)})
        return pd.DataFrame(rows)


@dataclass
class SeasonalAdjustedRule:
    """Per-ISO-month LGDI threshold (mean + threshold_sd·SD per month-of-year).

    For months with fewer than ``min_month_n`` baseline observations, falls
    back to the global mean+threshold_sd·SD.

    Default threshold_sd=1.5 matches NC_revision optimised strategy.
    """

    threshold_sd: float = 1.5
    min_month_n: int = 3
    _month_thresholds: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _global_threshold: float = field(default=math.nan, init=False, repr=False)

    def fit(
        self,
        lgdi_series: pd.Series,
        date_index: pd.Series,
        baseline_start: str | None = None,
        baseline_end: str | None = None,
    ) -> "SeasonalAdjustedRule":
        d = pd.to_datetime(date_index, errors="coerce")
        mask = pd.Series(True, index=lgdi_series.index)
        if baseline_start:
            mask &= d >= pd.Timestamp(baseline_start)
        if baseline_end:
            mask &= d <= pd.Timestamp(baseline_end)
        base_vals = lgdi_series[mask].dropna()
        base_months = d[mask][lgdi_series[mask].notna()].dt.month

        global_mu = float(base_vals.mean()) if len(base_vals) else 0.0
        global_sd = float(base_vals.std()) if len(base_vals) > 1 else 1.0
        self._global_threshold = global_mu + self.threshold_sd * global_sd

        for m in range(1, 13):
            month_mask = base_months == m
            s = base_vals[month_mask.values]
            if len(s) >= self.min_month_n:
                self._month_thresholds[m] = float(s.mean() + self.threshold_sd * s.std())
            else:
                self._month_thresholds[m] = self._global_threshold
        return self

    def apply(
        self, lgdi_series: pd.Series, date_index: pd.Series, date_col: str = "week"
    ) -> pd.DataFrame:
        """Return DataFrame with columns [week, lgdi, threshold, alert]."""
        if not self._month_thresholds:
            raise RuntimeError("Call fit() before apply()")
        d = pd.to_datetime(date_index, errors="coerce")
        rows: list[dict] = []
        for (week, month), value in zip(zip(lgdi_series.index, d.dt.month), lgdi_series):
            thr = self._month_thresholds.get(int(month), self._global_threshold)
            v = float(value)
            rows.append({date_col: week, "lgdi": v, "threshold": thr,
                         "alert": int(math.isfinite(v) and v >= thr)})
        return pd.DataFrame(rows)


@dataclass
class MultiScaleRule:
    """Dual-window OR rule: alert when 4-week OR 2-week rolling LGDI ≥ threshold.

    Thresholds are computed as mean + threshold_sd·SD on the baseline window.
    Default threshold_sd=1.5 matches NC_revision optimised strategy.
    """

    threshold_sd: float = 1.5
    threshold_4w_: float = field(default=math.nan, init=False, repr=False)
    threshold_2w_: float = field(default=math.nan, init=False, repr=False)

    def fit(
        self,
        lgdi_series: pd.Series,
        baseline_start: str | None = None,
        baseline_end: str | None = None,
        date_index: pd.Series | None = None,
    ) -> "MultiScaleRule":
        vals = lgdi_series
        if date_index is not None and (baseline_start or baseline_end):
            d = pd.to_datetime(date_index, errors="coerce")
            mask = pd.Series(True, index=lgdi_series.index)
            if baseline_start:
                mask &= d >= pd.Timestamp(baseline_start)
            if baseline_end:
                mask &= d <= pd.Timestamp(baseline_end)
            vals = lgdi_series[mask]
        vals_4w = vals.dropna()
        vals_2w = vals.rolling(window=2, min_periods=1).mean().dropna()
        for src, attr in ((vals_4w, "threshold_4w_"), (vals_2w, "threshold_2w_")):
            if len(src) < 2:
                setattr(self, attr, 0.0)
                continue
            setattr(self, attr, float(src.mean() + self.threshold_sd * src.std()))
        return self

    def apply(
        self, lgdi_series: pd.Series, date_col: str = "week"
    ) -> pd.DataFrame:
        """Return DataFrame with columns [week, lgdi_4w, lgdi_2w, alert]."""
        lgdi_2w = lgdi_series.rolling(window=2, min_periods=1).mean()
        rows: list[dict] = []
        for week, (v4, v2) in zip(lgdi_series.index, zip(lgdi_series, lgdi_2w)):
            alert = int(
                (math.isfinite(float(v4)) and float(v4) >= self.threshold_4w_)
                or (math.isfinite(float(v2)) and float(v2) >= self.threshold_2w_)
            )
            rows.append({date_col: week, "lgdi_4w": float(v4), "lgdi_2w": float(v2), "alert": alert})
        return pd.DataFrame(rows)
