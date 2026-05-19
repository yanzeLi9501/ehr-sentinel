"""Dataset-adaptive lab, disease, and feature configuration helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ehr_sentinel.utils.config import DEFAULT_COMORBIDITY_GROUPS, EpidemicConfig


DEFAULT_LAB_CANDIDATES = ["WBC", "CRP", "HGB", "ALB", "CREA", "GLU", "K", "Na"]

ADAPTIVE_COMORBIDITY_GROUPS: dict[str, str] = {
    "Cardiovascular": DEFAULT_COMORBIDITY_GROUPS["Cardiovascular"] + r"|^39[0-8]|^4[0-5][0-9]",
    "Diabetes": DEFAULT_COMORBIDITY_GROUPS["Diabetes"] + r"|^250",
    "Respiratory": DEFAULT_COMORBIDITY_GROUPS["Respiratory"] + r"|^4[6-9][0-9]|^5[0-1][0-9]",
    "Renal": DEFAULT_COMORBIDITY_GROUPS["Renal"] + r"|^58[0-9]|^59[0-9]",
    "Cancer": DEFAULT_COMORBIDITY_GROUPS["Cancer"] + r"|^1[4-9][0-9]|^2[0-3][0-9]",
    "Neurological": DEFAULT_COMORBIDITY_GROUPS["Neurological"] + r"|^29[0-9]|^3[0-8][0-9]",
}


@dataclass(frozen=True)
class LabPanelSpec:
    selected: list[str]
    detected: list[str]
    coverage: dict[str, float]
    reason: str


class LabPanelAdapter:
    """Select a per-dataset lab panel from available numeric lab columns."""

    def __init__(
        self,
        *,
        min_coverage: float = 0.01,
        min_non_null: int = 30,
        min_variance: float = 1e-12,
        preferred_order: Iterable[str] = DEFAULT_LAB_CANDIDATES,
    ) -> None:
        self.min_coverage = float(min_coverage)
        self.min_non_null = int(min_non_null)
        self.min_variance = float(min_variance)
        self.preferred_order = list(preferred_order)

    def select(self, df: pd.DataFrame) -> LabPanelSpec:
        detected = [c for c in self.preferred_order if c in df.columns]
        coverage: dict[str, float] = {}
        selected: list[str] = []
        n = max(len(df), 1)
        for col in detected:
            s = pd.to_numeric(df[col], errors="coerce")
            non_null = int(s.notna().sum())
            coverage[col] = float(non_null / n)
            if non_null >= self.min_non_null and coverage[col] >= self.min_coverage and float(s.var(skipna=True) or 0.0) > self.min_variance:
                selected.append(col)
        reason = (
            f"selected {len(selected)}/{len(detected)} detected labs with "
            f"coverage>={self.min_coverage:g}, n>={self.min_non_null}, non-zero variance"
        )
        return LabPanelSpec(selected=selected, detected=detected, coverage=coverage, reason=reason)


@dataclass(frozen=True)
class DiseaseSignal:
    name: str
    label: str
    reference_codes: list[str]
    count: int
    selection_reason: str = ""
    reference_months: list[int] = field(default_factory=lambda: list(range(1, 13)))
    season_months: list[int] = field(default_factory=lambda: list(range(1, 13)))


class DiseaseDetector:
    """Detect COVID, influenza, and other viral records from ICD-9/ICD-10/text."""

    COVID_CODES = re.compile(r"^(?:U07|B342)", re.I)
    FLU_CODES = re.compile(r"^(?:J09|J10|J11|487)", re.I)
    OTHER_VIRAL_CODES = re.compile(r"^(?:B0[0-9]|B1[5-9]|B2[5-7]|B3[0-4]|B97|J12|079|480|008)", re.I)

    COVID_TEXT = re.compile(r"COVID|SARS[- ]?COV|新型冠状病毒|冠状病毒肺炎|冠状病毒感染", re.I)
    FLU_TEXT = re.compile(r"INFLUENZA|FLU\b|流感", re.I)
    OTHER_VIRAL_TEXT = re.compile(r"VIRAL|VIRUS|病毒感染|病毒性", re.I)

    SIGNALS = {
        "covid19": ("COVID-19", ["U07.1", "U07.2", "B34.2"], "COVID records present"),
        "influenza": ("Influenza", ["J09", "J10", "J11", "487"], "No COVID records; influenza records present"),
        "other_viral": ("Other viral infection", ["B34", "B97", "J12", "079", "480"], "No COVID/influenza records; other viral records present"),
    }

    def detect(self, df: pd.DataFrame, *, code_col: str = "icd10", text_col: str = "diagnosis_text") -> dict[str, int]:
        codes = df[code_col].fillna("").astype(str).str.upper().str.replace(".", "", regex=False) if code_col in df else pd.Series("", index=df.index)
        text = df[text_col].fillna("").astype(str) if text_col in df else pd.Series("", index=df.index)
        covid = codes.str.contains(self.COVID_CODES) | text.str.contains(self.COVID_TEXT, regex=True, na=False)
        flu = codes.str.contains(self.FLU_CODES) | text.str.contains(self.FLU_TEXT, regex=True, na=False)
        other = (codes.str.contains(self.OTHER_VIRAL_CODES) | text.str.contains(self.OTHER_VIRAL_TEXT, regex=True, na=False)) & ~covid & ~flu
        return {"covid19": int(covid.sum()), "influenza": int(flu.sum()), "other_viral": int(other.sum())}

    #: Minimum disease record count to consider a target "present".
    #: Datasets like MIMIC-IV 2.2 contain only 2 SARS (B34.2) rows — which are
    #: not clinically meaningful as a COVID surveillance target — so we require
    #: at least this many records before promoting that disease to the analysis.
    MIN_SIGNAL_COUNT: int = 10

    def select(self, counts: dict[str, int]) -> list[DiseaseSignal]:
        selected: list[DiseaseSignal] = []
        covid_n = counts.get("covid19", 0)
        if covid_n >= self.MIN_SIGNAL_COUNT:
            label, codes, reason = self.SIGNALS["covid19"]
            selected.append(DiseaseSignal("covid19", label, codes, covid_n, reason))
            return selected
        if counts.get("influenza", 0) > 0:
            label, codes, reason = self.SIGNALS["influenza"]
            selected.append(DiseaseSignal("influenza", label, codes, counts["influenza"], reason, [1, 2, 12], [11, 12, 1, 2, 3]))
        if counts.get("other_viral", 0) > 0:
            label, codes, reason = self.SIGNALS["other_viral"]
            if selected:
                reason = "Influenza and other viral records both present; run parallel analyses"
            selected.append(DiseaseSignal("other_viral", label, codes, counts["other_viral"], reason))
        if not selected:
            label, codes, _ = self.SIGNALS["influenza"]
            # Carry through near-zero COVID count in reason so caller can see it
            covid_note = f" (detected {covid_n} COVID records, below min_signal_count={self.MIN_SIGNAL_COUNT})" if covid_n else ""
            selected.append(DiseaseSignal("influenza", label, codes, 0, f"No sufficient COVID/flu/other viral records detected{covid_note}; influenza fallback config only", [1, 2, 12], [11, 12, 1, 2, 3]))
        return selected
        if counts.get("influenza", 0) > 0:
            label, codes, reason = self.SIGNALS["influenza"]
            selected.append(DiseaseSignal("influenza", label, codes, counts["influenza"], reason, [1, 2, 12], [11, 12, 1, 2, 3]))
        if counts.get("other_viral", 0) > 0:
            label, codes, reason = self.SIGNALS["other_viral"]
            if selected:
                reason = "Influenza and other viral records both present; run parallel analyses"
            selected.append(DiseaseSignal("other_viral", label, codes, counts["other_viral"], reason))
        if not selected:
            label, codes, _ = self.SIGNALS["influenza"]
            selected.append(DiseaseSignal("influenza", label, codes, 0, "No COVID/flu/other viral records detected; influenza fallback config only", [1, 2, 12], [11, 12, 1, 2, 3]))
        return selected


@dataclass(frozen=True)
class FeaturePlan:
    lab_panel: list[str]
    min_visit_order: int
    gap_cap_days: int
    los_cap_days: int
    enhanced_features: bool
    reason: str


class AutoFeatureEngineer:
    """Choose feature-engineering knobs from source shape and lab availability."""

    def plan(self, df: pd.DataFrame, labs: LabPanelSpec) -> FeaturePlan:
        repeat_ratio = 0.0
        if "mrn" in df.columns and len(df):
            repeat_ratio = float((df.groupby("mrn").size() > 1).mean())
        gap_cap = self._cap_from_quantile(df.get("gap"), default=30, lower=7, upper=90)
        los_cap = self._cap_from_quantile(df.get("los"), default=60, lower=7, upper=180)
        # Enhanced rolling/EMA features are useful for model training, but can be
        # prohibitively expensive for aggregate validation on very large cohorts.
        enhanced = 5000 <= len(df) <= 100_000 and repeat_ratio >= 0.05 and len(labs.selected) >= 2
        min_visit_order = 2 if repeat_ratio >= 0.10 else 1
        return FeaturePlan(
            lab_panel=list(labs.selected),
            min_visit_order=min_visit_order,
            gap_cap_days=gap_cap,
            los_cap_days=los_cap,
            enhanced_features=enhanced,
            reason=(
                f"repeat_ratio={repeat_ratio:.3f}; labs={len(labs.selected)}; "
                f"gap_cap={gap_cap}; los_cap={los_cap}; enhanced={enhanced}"
            ),
        )

    @staticmethod
    def _cap_from_quantile(values: pd.Series | None, *, default: int, lower: int, upper: int) -> int:
        if values is None:
            return default
        s = pd.to_numeric(values, errors="coerce")
        s = s[np.isfinite(s) & (s >= 0)]
        if len(s) < 20:
            return default
        return int(max(lower, min(upper, round(float(s.quantile(0.95))))))


def build_adaptive_config(
    df: pd.DataFrame,
    signal: DiseaseSignal,
    feature_plan: FeaturePlan,
    *,
    target_group: str = "Respiratory",
) -> EpidemicConfig:
    dates = pd.to_datetime(df["admission_date"], errors="coerce").dropna()
    min_year = int(dates.dt.year.min()) if len(dates) else 2016
    max_year = int(dates.dt.year.max()) if len(dates) else min_year + 2
    span = max_year - min_year
    if span >= 2:
        # Normal case: first 2 years as baseline, rest as monitoring.
        baseline_end_year = min_year + 1
        monitoring_start_year = min_year + 2
    elif span == 1:
        # 2-year dataset: use first year as baseline, second as monitoring.
        baseline_end_year = min_year
        monitoring_start_year = max_year
    else:
        # Single-year dataset (e.g. CDSL 2020-only): use first half of year
        # as baseline so the second half can be monitored.
        baseline_end_year = min_year
        monitoring_start_year = min_year
    return EpidemicConfig(
        target_disease=signal.label,
        reference_icd10_codes=signal.reference_codes,
        reference_years=list(range(min_year, min(max_year, min_year + 4) + 1)),
        reference_months=signal.reference_months,
        baseline_start=f"{min_year}-01-01",
        baseline_end=f"{baseline_end_year}-06-30" if span == 0 else f"{baseline_end_year}-12-31",
        monitoring_start=f"{monitoring_start_year}-07-01" if span == 0 else f"{monitoring_start_year}-01-01",
        comorbidity_groups=dict(ADAPTIVE_COMORBIDITY_GROUPS),
        target_group=target_group,
        lab_panel=list(feature_plan.lab_panel),
        epidemic_season_months=signal.season_months,
        min_visit_order=feature_plan.min_visit_order,
        gap_cap_days=feature_plan.gap_cap_days,
        los_cap_days=feature_plan.los_cap_days,
        enhanced_features=feature_plan.enhanced_features,
    )
