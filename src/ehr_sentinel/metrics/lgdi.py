"""LOS–Gap Deviation Index (LGDI).

Two computation modes
---------------------
1. **Simple mode** (``LGDIComputer``) — backward-compatible, single residual
   column, weekly calendar period grouping.  Suitable for quick validation
   when no trained LOS/gap models are available.

   LGDI(t) = S_target(t) − mean(S_other_groups(t))

   where S_group(t) = MAE_group(t) / baseline_MAE_group (unsigned MASE).

2. **Sliding-window dual-target mode** (``SlidingWindowLGDI``) — authoritative
   publication-grade algorithm matching NC_revision/run_lgdi_surveillance.py.

   For each 4-week window [anchor−21d, anchor+6d] stepped weekly (W-MON):
   - COVID-positive admissions are excluded from signal computation.
   - Two XGBoost models predict next_los_days and next_gap_days.
   - Signed MASE per target:
       signed_los  = +1 × mean(resid_los)  / baseline_MAE_los
       signed_gap  = −1 × mean(resid_gap)  / baseline_MAE_gap
   - Group score = mean(signed_los, signed_gap) when both finite.
   - LGDI(window) = score_Respiratory − mean(score_other_groups)

   Reference: Li et al. (manuscript in preparation), §2 Eq. 1–4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Direction signs per target (LOS longer = worse → +1; gap shorter = worse → −1)
_TARGET_DIRECTION: dict[str, float] = {
    "next_los_days": 1.0,
    "next_gap_days": -1.0,
}


@dataclass
class LGDIResult:
    timeline: pd.DataFrame                 # week/window_anchor, group, S
    lgdi: pd.DataFrame                     # week/window_anchor, lgdi
    baseline_stats: dict[str, dict[str, float]] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════
# Simple mode (backward-compatible)
# ════════════════════════════════════════════════════════════════════════

class LGDIComputer:
    """Compute LGDI per group / per calendar week (simple, single-residual mode).

    The "rhythm" residual for each admission is the XGBoost gap-predictor
    residual (or a plain z-score when no model is provided).  Per-group,
    per-week mean residuals are MASE-scaled then differenced.

    For the publication-grade 4-week sliding window algorithm with dual
    signed MASE, use ``SlidingWindowLGDI`` instead.
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


# ════════════════════════════════════════════════════════════════════════
# Sliding-window dual-target mode (publication-grade)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class WindowRow:
    """Per-window summary returned by ``SlidingWindowLGDI``."""
    window_anchor: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_admissions: int
    n_admissions_excl_covid: int
    valid: bool                  # False when window has < min_total_n admissions
    resp_score: float
    mean_other_score: float
    lgdi: float
    group_scores: dict[str, float] = field(default_factory=dict)
    group_n: dict[str, int] = field(default_factory=dict)


class SlidingWindowLGDI:
    """Publication-grade 4-week sliding window LGDI computation.

    Matches NC_revision/run_lgdi_surveillance.py exactly:

    1. Fit XGBoost LOS and gap models on the baseline window.
    2. For each W-MON anchor from ``first_anchor`` to the last admission:
       a. Collect admissions in [anchor−21d, anchor+6d].
       b. Exclude COVID-positive admissions (``covid_col`` flag).
       c. For each comorbidity group with ≥ ``min_group_n`` admissions:
          - signed_los  = +1 × mean(resid_los)  / baseline_MAE_los_group
          - signed_gap  = −1 × mean(resid_gap)  / baseline_MAE_gap_group
          - group_score = mean of the two signed MASEs (finite only)
       d. LGDI = score_target_group − mean(score_other_groups)

    Parameters
    ----------
    target_group : str
        Comorbidity group used as the epidemic signal (default "Respiratory").
    comorbidity_groups : list[str]
        All comorbidity group names. Must include ``target_group``.
    baseline_start, baseline_end : str
        ISO date strings for the baseline window used to fit models and
        compute per-group MASE denominators (e.g. "2016-01-01", "2018-12-31").
    min_total_n : int
        Minimum admissions per window to produce a valid row (default 50).
    min_group_n : int
        Minimum admissions per comorbidity group per window (default 10).
    """

    DEFAULT_GROUPS = [
        "Cardiovascular", "Hypertension", "Diabetes",
        "Cerebrovascular", "Renal", "Respiratory",
    ]

    def __init__(
        self,
        target_group: str = "Respiratory",
        comorbidity_groups: Optional[list[str]] = None,
        baseline_start: Optional[str] = None,
        baseline_end: Optional[str] = None,
        min_total_n: int = 50,
        min_group_n: int = 10,
    ) -> None:
        self.target_group = target_group
        self.comorbidity_groups = comorbidity_groups or self.DEFAULT_GROUPS
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end
        self.min_total_n = int(min_total_n)
        self.min_group_n = int(min_group_n)
        # Per-group per-target baseline MAE scale  {group: {target: float}}
        self._baseline_scale: dict[str, dict[str, float]] = {}

    # ── Baseline scale fit ───────────────────────────────────────────────
    def fit_baseline_scale(
        self,
        df: pd.DataFrame,
        group_col: str = "comorbidity_group",
    ) -> "SlidingWindowLGDI":
        """Compute per-group per-target baseline mean absolute residual."""
        base = df
        if self.baseline_start:
            base = base[pd.to_datetime(base["admission_date"], errors="coerce")
                        >= pd.Timestamp(self.baseline_start)]
        if self.baseline_end:
            base = base[pd.to_datetime(base["admission_date"], errors="coerce")
                        <= pd.Timestamp(self.baseline_end)]
        for group in self.comorbidity_groups:
            self._baseline_scale[group] = {}
            sub = base[base[group_col] == group] if group_col in base.columns else base[base.get(group, False)]
            for target in ("resid_next_los_days", "resid_next_gap_days"):
                if target not in sub.columns:
                    self._baseline_scale[group][target] = 1.0
                    continue
                vals = pd.to_numeric(sub[target], errors="coerce").dropna().to_numpy()
                mae = float(np.mean(np.abs(vals))) if len(vals) > 0 else 1.0
                self._baseline_scale[group][target] = max(mae, 1e-9)
        return self

    # ── Per-window group score ───────────────────────────────────────────
    def _group_score(
        self,
        window_df: pd.DataFrame,
        group: str,
        group_col: str = "comorbidity_group",
    ) -> tuple[float, int]:
        """Return (group_score, n) for one group in one window."""
        sub = window_df[window_df[group_col] == group] if group_col in window_df.columns else pd.DataFrame()
        n = len(sub)
        if n < self.min_group_n:
            return math.nan, n
        scores: list[float] = []
        for target, direction in (("resid_next_los_days", 1.0), ("resid_next_gap_days", -1.0)):
            if target not in sub.columns:
                continue
            vals = pd.to_numeric(sub[target], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                continue
            mae_scale = self._baseline_scale.get(group, {}).get(target, 1.0)
            signed = direction * float(np.mean(vals)) / mae_scale
            if math.isfinite(signed):
                scores.append(signed)
        score = float(np.mean(scores)) if scores else math.nan
        return score, n

    # ── Single window ────────────────────────────────────────────────────
    def _window_row(
        self,
        anchor: pd.Timestamp,
        df: pd.DataFrame,
        group_col: str = "comorbidity_group",
        covid_col: Optional[str] = "is_covid_positive",
        date_col: str = "admission_date",
    ) -> WindowRow:
        start = anchor - pd.Timedelta(days=21)
        end = anchor + pd.Timedelta(days=6)
        d = pd.to_datetime(df[date_col], errors="coerce")
        frame = df[(d >= start) & (d <= end)]
        n_total = len(frame)
        # Exclude COVID-positive admissions from the signal
        if covid_col and covid_col in frame.columns:
            monitoring = frame[~frame[covid_col].astype(bool)]
        else:
            monitoring = frame
        n_excl = len(monitoring)
        invalid = WindowRow(
            window_anchor=anchor, window_start=start, window_end=end,
            n_admissions=n_total, n_admissions_excl_covid=n_excl,
            valid=False, resp_score=math.nan, mean_other_score=math.nan, lgdi=math.nan,
        )
        if n_total < self.min_total_n:
            return invalid
        group_scores: dict[str, float] = {}
        group_n: dict[str, int] = {}
        for group in self.comorbidity_groups:
            score, n = self._group_score(monitoring, group, group_col=group_col)
            group_scores[group] = score
            group_n[group] = n
        resp_score = group_scores.get(self.target_group, math.nan)
        other_scores = [
            v for g, v in group_scores.items()
            if g != self.target_group and math.isfinite(v)
        ]
        mean_other = float(np.mean(other_scores)) if other_scores else math.nan
        if not math.isfinite(resp_score) or not math.isfinite(mean_other):
            return invalid
        lgdi = resp_score - mean_other
        return WindowRow(
            window_anchor=anchor, window_start=start, window_end=end,
            n_admissions=n_total, n_admissions_excl_covid=n_excl,
            valid=True, resp_score=resp_score, mean_other_score=mean_other, lgdi=lgdi,
            group_scores=group_scores, group_n=group_n,
        )

    # ── Full timeline ────────────────────────────────────────────────────
    def compute(
        self,
        df: pd.DataFrame,
        group_col: str = "comorbidity_group",
        covid_col: Optional[str] = "is_covid_positive",
        date_col: str = "admission_date",
        first_anchor: Optional[str] = None,
    ) -> LGDIResult:
        """Compute LGDI over all W-MON anchors from ``first_anchor`` to last admission.

        If ``_baseline_scale`` is empty, ``fit_baseline_scale()`` is called
        automatically on the full DataFrame (use ``fit_baseline_scale()``
        explicitly to restrict to a baseline window before calling this).
        """
        if not self._baseline_scale:
            self.fit_baseline_scale(df, group_col=group_col)

        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if dates.empty:
            empty = pd.DataFrame(columns=["window_anchor", "lgdi"])
            return LGDIResult(timeline=empty, lgdi=empty)

        fa = pd.Timestamp(first_anchor) if first_anchor else dates.min()
        last = dates.max()
        anchors = pd.date_range(fa, last, freq="W-MON")

        rows: list[dict] = []
        for anchor in anchors:
            wr = self._window_row(anchor, df, group_col=group_col,
                                  covid_col=covid_col, date_col=date_col)
            row: dict = {
                "window_anchor": wr.window_anchor,
                "window_start": wr.window_start,
                "window_end": wr.window_end,
                "n_admissions": wr.n_admissions,
                "n_admissions_excl_covid": wr.n_admissions_excl_covid,
                "valid": wr.valid,
                "resp_score": wr.resp_score,
                "mean_other_score": wr.mean_other_score,
                "lgdi": wr.lgdi,
            }
            for g, v in wr.group_scores.items():
                row[f"score_{g}"] = v
            for g, n in wr.group_n.items():
                row[f"n_{g}"] = n
            rows.append(row)

        timeline = pd.DataFrame(rows)
        if timeline.empty:
            empty = pd.DataFrame(columns=["window_anchor", "lgdi"])
            return LGDIResult(timeline=empty, lgdi=empty)

        lgdi_df = (
            timeline[timeline["valid"]][["window_anchor", "lgdi"]]
            .rename(columns={"window_anchor": "week"})
            .reset_index(drop=True)
        )
        return LGDIResult(
            timeline=timeline,
            lgdi=lgdi_df,
            baseline_stats={
                g: {t: v for t, v in tv.items()}
                for g, tv in self._baseline_scale.items()
            },
        )

    # ── Baseline thresholds ──────────────────────────────────────────────
    def thresholds(
        self,
        timeline: pd.DataFrame,
        lgdi_col: str = "lgdi",
    ) -> dict[str, float]:
        """Return mean+1.5SD, mean+2SD, and P97.5 thresholds from baseline window."""
        base = timeline
        if self.baseline_start:
            base = base[pd.to_datetime(base["window_anchor"], errors="coerce")
                        >= pd.Timestamp(self.baseline_start)]
        if self.baseline_end:
            base = base[pd.to_datetime(base["window_anchor"], errors="coerce")
                        <= pd.Timestamp(self.baseline_end)]
        vals = base[base.get("valid", True).astype(bool)][lgdi_col].dropna() \
               if "valid" in base.columns else base[lgdi_col].dropna()
        if vals.empty:
            return {}
        mu, sd = float(vals.mean()), float(vals.std())
        return {
            "mean_plus_1_5sd": mu + 1.5 * sd,
            "mean_plus_2sd": mu + 2.0 * sd,
            "p97_5": float(np.percentile(vals, 97.5)),
        }

    # ── Rolling windows helper (shared with LGDIComputer) ───────────────
    @staticmethod
    def rolling_windows(timeline: pd.DataFrame, window: int = 4, value_col: str = "lgdi") -> pd.DataFrame:
        if timeline.empty:
            return timeline
        date_col = "window_anchor" if "window_anchor" in timeline.columns else "week"
        out = timeline.sort_values(date_col).copy()
        out[f"{value_col}_rmean{window}"] = out[value_col].rolling(window, min_periods=1).mean()
        out[f"{value_col}_rstd{window}"] = out[value_col].rolling(window, min_periods=2).std()
        return out
