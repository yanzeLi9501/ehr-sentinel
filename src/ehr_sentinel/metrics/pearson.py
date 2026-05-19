"""Pearson profile correlation + RDI (configurable target group)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PearsonResult:
    week: pd.Timestamp
    group: str
    r: float
    p: float
    n: int


class PearsonProfileCorrelation:
    """Compute weekly Pearson correlation between a per-group lab-profile
    vector and a configurable reference epidemic profile, then aggregate
    into a relative deviation index (RDI).
    """

    def __init__(
        self,
        lab_panel: list[str],
        target_group: str = "Respiratory",
        min_admissions: int = 5,
    ) -> None:
        self.lab_panel = list(lab_panel)
        self.target_group = target_group
        self.min_admissions = int(min_admissions)

    # ── Profile construction ────────────────────────────────────────────
    def build_profile_vector(
        self,
        df: pd.DataFrame,
        baseline_mean: pd.Series,
        baseline_std: pd.Series,
    ) -> pd.Series:
        """Return z-scored mean of each lab in the panel."""
        panel = [l for l in self.lab_panel if l in df.columns]
        if not panel or df.empty:
            return pd.Series(dtype=float)
        means = df[panel].mean()
        z = (means - baseline_mean[panel]) / baseline_std[panel].replace(0, np.nan)
        return z.fillna(0.0)

    def build_reference_profile(
        self,
        df: pd.DataFrame,
        reference_icd10_codes: list[str],
        reference_years: list[int],
        reference_months: list[int],
        date_col: str = "admission_date",
        icd10_col: str = "icd10",
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return (reference_z_profile, baseline_mean, baseline_std)."""
        panel = [l for l in self.lab_panel if l in df.columns]
        d = pd.to_datetime(df[date_col], errors="coerce")
        in_year = d.dt.year.isin(reference_years)
        in_month = d.dt.month.isin(reference_months)
        icd_ok = df[icd10_col].astype(str).str.upper().isin([c.upper() for c in reference_icd10_codes])
        ref_mask = in_year & in_month & icd_ok
        other_mask = ~ref_mask
        baseline_mean = df.loc[other_mask, panel].mean()
        baseline_std = df.loc[other_mask, panel].std().replace(0, np.nan).fillna(1.0)
        if ref_mask.sum() == 0:
            ref_profile = pd.Series(0.0, index=panel)
        else:
            ref_means = df.loc[ref_mask, panel].mean()
            ref_profile = (ref_means - baseline_mean) / baseline_std
            ref_profile = ref_profile.fillna(0.0)
        return ref_profile, baseline_mean, baseline_std

    # ── Weekly correlation ──────────────────────────────────────────────
    def weekly_correlation(
        self,
        df: pd.DataFrame,
        reference_profile: pd.Series,
        baseline_mean: pd.Series,
        baseline_std: pd.Series,
        group_col: str = "comorbidity_group",
        date_col: str = "admission_date",
    ) -> pd.DataFrame:
        out: list[PearsonResult] = []
        d = pd.to_datetime(df[date_col], errors="coerce")
        weeks = d.dt.to_period("W")
        for (week, group), sub in df.groupby([weeks, group_col]):
            if len(sub) < self.min_admissions or pd.isna(group):
                continue
            vec = self.build_profile_vector(sub, baseline_mean, baseline_std)
            if vec.empty or vec.std() == 0 or reference_profile.std() == 0:
                continue
            r, p = stats.pearsonr(vec.values, reference_profile.values)
            out.append(PearsonResult(
                week=week.start_time, group=group, r=float(r), p=float(p), n=int(len(sub))
            ))
        return pd.DataFrame([r.__dict__ for r in out])

    # ── RDI ─────────────────────────────────────────────────────────────
    def compute_rdi(self, weekly_r: pd.DataFrame) -> pd.DataFrame:
        """RDI(t) = r_target_group(t) − mean(r_other_groups(t))."""
        if weekly_r.empty:
            return weekly_r.assign(rdi=pd.Series(dtype=float))
        pivot = weekly_r.pivot_table(index="week", columns="group", values="r")
        if self.target_group not in pivot.columns:
            return pd.DataFrame(columns=["week", "rdi"])
        others = pivot.drop(columns=[self.target_group])
        rdi = pivot[self.target_group] - others.mean(axis=1, skipna=True)
        return rdi.reset_index().rename(columns={self.target_group: "rdi"}).assign(rdi=rdi.values)

    # ── Bootstrap CI for a single (group, week) ─────────────────────────
    def bootstrap_ci(
        self,
        df_week: pd.DataFrame,
        reference_profile: pd.Series,
        baseline_mean: pd.Series,
        baseline_std: pd.Series,
        n_boot: int = 500,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        rs: list[float] = []
        for _ in range(n_boot):
            sub = df_week.sample(len(df_week), replace=True, random_state=int(rng.integers(0, 2**31)))
            vec = self.build_profile_vector(sub, baseline_mean, baseline_std)
            if vec.empty or vec.std() == 0:
                continue
            r, _ = stats.pearsonr(vec.values, reference_profile.values)
            rs.append(r)
        if not rs:
            return float("nan"), float("nan")
        lo = float(np.quantile(rs, alpha / 2))
        hi = float(np.quantile(rs, 1 - alpha / 2))
        return lo, hi
