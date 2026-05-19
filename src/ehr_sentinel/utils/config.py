"""Disease-agnostic configuration model for epidemic surveillance pipelines.

`EpidemicConfig` is a Pydantic v2 model. Every disease-specific parameter is
exposed here; the rest of the package never hardcodes a disease name, ICD-10
code, comorbidity group, or season.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# Sensible defaults — purely conventional, not disease-specific
DEFAULT_COMORBIDITY_GROUPS: dict[str, str] = {
    "Cardiovascular": r"^I0[0-9]|^I1[0-5]|^I2[0-5]|^I3[0-9]|^I4[0-9]|^I5[0-1]|^I6[0-9]|^I7[0-9]",
    "Diabetes": r"^E1[0-4]",
    "Respiratory": r"^J0[0-9]|^J1[0-9]|^J2[0-2]|^J3[0-9]|^J4[0-79]|^J6[0-9]|^J8[0-69]|^J9[0-69]",
    "Renal": r"^N1[7-9]|^N0[0-8]",
    "Cancer": r"^C[0-9]{2}|^D0[0-9]|^D3[7-9]|^D4[0-8]",
    "Neurological": r"^G[0-9]{2}|^F0[0-3]",
}

DEFAULT_LAB_PANEL: list[str] = ["WBC", "CRP", "HGB", "ALB", "CREA", "GLU", "K", "Na"]


class EpidemicConfig(BaseModel):
    """Configuration for a single surveillance run.

    The pipeline is fully driven by this object. Different epidemics are
    handled by instantiating different configs; nothing in the code is
    hardcoded for a specific disease.
    """

    model_config = {"arbitrary_types_allowed": False, "frozen": False}

    # ── Target disease & reference profile ──────────────────────────────
    target_disease: str = Field(..., description="Disease label (e.g., 'COVID-19', 'Influenza').")
    reference_icd10_codes: list[str] = Field(
        ..., min_length=1,
        description="ICD-10 codes that define the epidemic reference profile.",
    )
    reference_years: list[int] = Field(..., min_length=1)
    reference_months: list[int] = Field(..., min_length=1)

    # ── Windows ─────────────────────────────────────────────────────────
    baseline_start: str = Field(..., description="ISO date 'YYYY-MM-DD' — baseline start.")
    baseline_end: str = Field(..., description="ISO date 'YYYY-MM-DD' — baseline end.")
    monitoring_start: Optional[str] = Field(
        default=None, description="When surveillance starts; defaults to baseline_end + 1 day."
    )

    # ── Groups & lab panel ──────────────────────────────────────────────
    comorbidity_groups: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_COMORBIDITY_GROUPS))
    target_group: str = Field(
        default="Respiratory",
        description="Name of the group that should serve as the RDI/LGDI target.",
    )
    lab_panel: list[str] = Field(default_factory=lambda: list(DEFAULT_LAB_PANEL))

    # ── Alerting & seasonality ──────────────────────────────────────────
    alert_threshold_sd: float = Field(default=1.5, ge=0.0)
    consensus_k: int = Field(default=2, ge=1, description="k-of-n consensus rule threshold.")
    sustained_weeks: int = Field(default=2, ge=1)
    epidemic_season_months: list[int] = Field(
        default_factory=lambda: list(range(1, 13)),
        description="Months when alerts are active (e.g., [11,12,1,2,3] for flu).",
    )

    # ── Feature engineering knobs ───────────────────────────────────────
    min_visit_order: int = Field(default=5, ge=1)
    gap_cap_days: int = Field(default=30, ge=1)
    los_cap_days: int = Field(default=60, ge=1)
    enhanced_features: bool = Field(
        default=True, description="If True, build the 140-feature set; else the 119 base feature set.",
    )

    # ── Validators ──────────────────────────────────────────────────────
    @field_validator("reference_months", "epidemic_season_months")
    @classmethod
    def _months_range(cls, v: list[int]) -> list[int]:
        for m in v:
            if not 1 <= m <= 12:
                raise ValueError(f"month {m} out of range [1, 12]")
        return v

    @field_validator("baseline_start", "baseline_end", "monitoring_start")
    @classmethod
    def _iso_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"expected 'YYYY-MM-DD', got {v!r}") from e
        return v

    @model_validator(mode="after")
    def _window_order(self) -> "EpidemicConfig":
        b0 = datetime.strptime(self.baseline_start, "%Y-%m-%d")
        b1 = datetime.strptime(self.baseline_end, "%Y-%m-%d")
        if b1 <= b0:
            raise ValueError("baseline_end must be after baseline_start")
        if self.target_group not in self.comorbidity_groups:
            raise ValueError(
                f"target_group {self.target_group!r} must be one of {list(self.comorbidity_groups)}"
            )
        return self

    # ── Helpers ─────────────────────────────────────────────────────────
    def monitoring_start_date(self) -> str:
        if self.monitoring_start:
            return self.monitoring_start
        dt = datetime.strptime(self.baseline_end, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")


class PresetConfigs:
    """Convenience presets — these are example configurations, not hardcoded
    assumptions. They simply construct an `EpidemicConfig` with sensible
    defaults for a few well-known epidemics.
    """

    @staticmethod
    def covid_19() -> EpidemicConfig:
        return EpidemicConfig(
            target_disease="COVID-19",
            reference_icd10_codes=["U07.1", "U07.2", "B34.2"],
            reference_years=[2020],
            reference_months=[1, 2, 3, 4],
            baseline_start="2016-01-01",
            baseline_end="2018-12-31",
            monitoring_start="2019-01-01",
            target_group="Respiratory",
            epidemic_season_months=list(range(1, 13)),
        )

    @staticmethod
    def influenza_seasonal() -> EpidemicConfig:
        return EpidemicConfig(
            target_disease="Influenza",
            reference_icd10_codes=["J09", "J10", "J11"],
            reference_years=[2016, 2017, 2018],
            reference_months=[1, 2, 12],
            baseline_start="2013-01-01",
            baseline_end="2015-12-31",
            monitoring_start="2016-01-01",
            target_group="Respiratory",
            epidemic_season_months=[11, 12, 1, 2, 3],
        )

    @staticmethod
    def custom(
        *,
        target_disease: str,
        icd10_codes: list[str],
        baseline_start: str,
        baseline_end: str,
        reference_years: list[int],
        reference_months: list[int],
        **kwargs,
    ) -> EpidemicConfig:
        return EpidemicConfig(
            target_disease=target_disease,
            reference_icd10_codes=icd10_codes,
            reference_years=reference_years,
            reference_months=reference_months,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            **kwargs,
        )
