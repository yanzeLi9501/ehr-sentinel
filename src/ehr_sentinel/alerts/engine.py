"""Configurable consensus / season / sustained alert rules."""
from __future__ import annotations

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
