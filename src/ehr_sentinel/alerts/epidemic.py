"""General-purpose epidemic cycle predictor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EpidemicWarning:
    target_disease: str
    onset_week: Optional[pd.Timestamp]
    lead_time_days: Optional[int]
    peak_week_estimate: Optional[pd.Timestamp]
    peak_value_estimate: Optional[float]
    at_risk_groups: list[tuple[str, float]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "target_disease": self.target_disease,
            "onset_week": str(self.onset_week) if self.onset_week is not None else None,
            "lead_time_days": self.lead_time_days,
            "peak_week_estimate": str(self.peak_week_estimate) if self.peak_week_estimate is not None else None,
            "peak_value_estimate": self.peak_value_estimate,
            "at_risk_groups": [(g, float(v)) for g, v in self.at_risk_groups],
            "notes": self.notes,
        }


class EpidemicPredictor:
    """Detect onset, estimate peak timing, rank at-risk groups."""

    def __init__(self, target_disease: str, alert_threshold_sd: float = 1.5) -> None:
        self.target_disease = target_disease
        self.alert_threshold_sd = float(alert_threshold_sd)

    def detect_cycle_onset(
        self,
        lgdi_timeline: pd.DataFrame,
        value_col: str = "lgdi",
        date_col: str = "week",
        baseline_frac: float = 0.3,
    ) -> Optional[pd.Timestamp]:
        if lgdi_timeline.empty:
            return None
        s = lgdi_timeline.sort_values(date_col).reset_index(drop=True)
        n = len(s)
        n_base = max(3, int(round(n * baseline_frac)))
        base = s[value_col].iloc[:n_base]
        mu = float(base.mean())
        sd = float(base.std()) or 1.0
        z = (s[value_col] - mu) / sd
        crossed = s[z >= self.alert_threshold_sd]
        if crossed.empty:
            return None
        return pd.Timestamp(crossed.iloc[0][date_col])

    def estimate_lead_time(
        self,
        onset_week: Optional[pd.Timestamp],
        true_outbreak_start: Optional[pd.Timestamp],
    ) -> Optional[int]:
        if onset_week is None or true_outbreak_start is None:
            return None
        return int((pd.Timestamp(true_outbreak_start) - pd.Timestamp(onset_week)).days)

    def predict_peak_timing(
        self,
        lgdi_timeline: pd.DataFrame,
        value_col: str = "lgdi",
        date_col: str = "week",
        lookahead_weeks: int = 12,
    ) -> tuple[Optional[pd.Timestamp], Optional[float]]:
        if lgdi_timeline.empty:
            return None, None
        s = lgdi_timeline.sort_values(date_col).copy()
        s["rmean"] = s[value_col].rolling(4, min_periods=1).mean()
        peak_idx = int(np.argmax(s["rmean"].values))
        return pd.Timestamp(s.iloc[peak_idx][date_col]), float(s.iloc[peak_idx]["rmean"])

    def identify_at_risk_groups(
        self,
        mase_df: pd.DataFrame,
        top_k: int = 3,
        group_col: str = "group",
        value_col: str = "S",
    ) -> list[tuple[str, float]]:
        if mase_df.empty:
            return []
        agg = mase_df.groupby(group_col)[value_col].mean().sort_values(ascending=False)
        return [(str(g), float(v)) for g, v in agg.head(top_k).items()]

    def generate_warning(
        self,
        lgdi_timeline: pd.DataFrame,
        mase_df: pd.DataFrame,
        true_outbreak_start: Optional[str] = None,
        top_k: int = 3,
    ) -> EpidemicWarning:
        onset = self.detect_cycle_onset(lgdi_timeline)
        peak_week, peak_val = self.predict_peak_timing(lgdi_timeline)
        groups = self.identify_at_risk_groups(mase_df, top_k=top_k)
        lead = self.estimate_lead_time(onset, pd.Timestamp(true_outbreak_start) if true_outbreak_start else None)
        notes = "no onset detected" if onset is None else "onset triggered by LGDI z-score threshold"
        return EpidemicWarning(
            target_disease=self.target_disease,
            onset_week=onset,
            lead_time_days=lead,
            peak_week_estimate=peak_week,
            peak_value_estimate=peak_val,
            at_risk_groups=groups,
            notes=notes,
        )
