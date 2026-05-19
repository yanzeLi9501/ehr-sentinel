"""High-level surveillance pipeline orchestration.

`run_surveillance_pipeline(df, config)` performs:
  1. Load / normalize  (caller-supplied DataFrame)
  2. Feature engineering
  3. Optional XGBoost gap-model training (skipped if xgboost missing)
  4. Pearson RDI
  5. LGDI
  6. Consensus / season / sustained alerts
  7. Epidemic warning
  8. Optional report writing
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ehr_sentinel.alerts.engine import ConsensusRule, SeasonFilter, SustainedRule
from ehr_sentinel.alerts.epidemic import EpidemicPredictor, EpidemicWarning
from ehr_sentinel.data.loader import EHRLoader
from ehr_sentinel.features.builder import FeatureBuilder
from ehr_sentinel.metrics.lgdi import LGDIComputer, LGDIResult
from ehr_sentinel.metrics.pearson import PearsonProfileCorrelation
from ehr_sentinel.reporting.tables import ReportGenerator
from ehr_sentinel.utils.config import EpidemicConfig


@dataclass
class SurveillanceResult:
    config: EpidemicConfig
    rdi_timeline: pd.DataFrame
    lgdi_result: LGDIResult
    alerts: pd.DataFrame
    warning: EpidemicWarning
    model_metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        n_lgdi = len(self.lgdi_result.lgdi)
        n_rdi = len(self.rdi_timeline)
        n_alerts = int(self.alerts["alert_sustained"].sum()) if "alert_sustained" in self.alerts.columns else 0
        return (
            f"[{self.config.target_disease}] "
            f"LGDI weeks={n_lgdi}, RDI weeks={n_rdi}, sustained alerts={n_alerts}, "
            f"onset={self.warning.onset_week}, peak≈{self.warning.peak_week_estimate}"
        )


def run_surveillance_pipeline(
    data: pd.DataFrame,
    config: EpidemicConfig,
    *,
    train_xgb: bool = True,
    output_dir: Optional[str | Path] = None,
) -> SurveillanceResult:
    """Run the full surveillance pipeline on an admissions DataFrame."""
    df = EHRLoader().from_dataframe(data)

    # Features
    builder = FeatureBuilder(config)
    fm = builder.build(df)

    # Optional model
    model_metrics: dict = {}
    gap_pred: Optional[np.ndarray] = None
    if train_xgb and fm.y_gap is not None and fm.y_gap.notna().sum() >= 50:
        try:
            from ehr_sentinel.models.trainer import XGBTrainer
            mask = fm.y_gap.notna() & (fm.meta.get("vo_pass", 1).astype(int) == 1)
            if mask.sum() >= 30:
                trainer = XGBTrainer()
                cv = trainer.cross_validate(fm.X.loc[mask], fm.y_gap.loc[mask], groups=fm.groups.loc[mask],
                                            n_splits=min(5, max(2, fm.groups.loc[mask].nunique())))
                model_metrics["cv_r2_mean"] = float(np.mean(cv["r2_test"]))
                model_metrics["cv_mae_mean"] = float(np.mean(cv["mae_test"]))
                tr = trainer.train_gap_model(fm.X.loc[mask], fm.y_gap.loc[mask])
                gap_pred_partial = trainer.predict(tr.model, fm.X)
                gap_pred = gap_pred_partial
                model_metrics["device"] = tr.device
                model_metrics["n_train"] = tr.n_train
        except ImportError as e:
            warnings.warn(f"XGBoost unavailable; skipping rhythm model ({e}).", RuntimeWarning, stacklevel=2)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"Rhythm model failed: {e}; continuing without it.", RuntimeWarning, stacklevel=2)

    # Attach predictions / group for downstream metrics
    df_metrics = df.copy()
    if "comorbidity_group" not in df_metrics.columns:
        df_metrics["comorbidity_group"] = fm.meta["comorbidity_group"].values
    else:
        df_metrics["comorbidity_group"] = fm.meta["comorbidity_group"].values
    if gap_pred is not None:
        df_metrics["gap_pred"] = gap_pred

    # Pearson RDI
    pearson = PearsonProfileCorrelation(
        lab_panel=config.lab_panel, target_group=config.target_group
    )
    ref_profile, base_mean, base_std = pearson.build_reference_profile(
        df_metrics,
        reference_icd10_codes=config.reference_icd10_codes,
        reference_years=config.reference_years,
        reference_months=config.reference_months,
    )
    weekly_r = pearson.weekly_correlation(df_metrics, ref_profile, base_mean, base_std)
    rdi_timeline = pearson.compute_rdi(weekly_r)

    # LGDI
    lgdi_comp = LGDIComputer(
        target_group=config.target_group,
        baseline_start=config.baseline_start,
        baseline_end=config.baseline_end,
    )
    lgdi_res = lgdi_comp.run(
        df_metrics,
        y_col="gap",
        pred_col="gap_pred" if "gap_pred" in df_metrics.columns else None,
    )

    # Alerts
    consensus = ConsensusRule(k=config.consensus_k, threshold_sd=config.alert_threshold_sd)
    timeline = lgdi_res.timeline
    if timeline.empty or "group" not in timeline.columns:
        import pandas as _pd
        sustained = _pd.DataFrame(columns=["week", "group", "S", "alert", "alert_sustained"])
    else:
        consensus.fit(timeline, group_col="group", value_col="S")
        raw_alerts = consensus.evaluate(timeline)
        season = SeasonFilter(months=config.epidemic_season_months).apply(raw_alerts)
        sustained = SustainedRule(n_weeks=config.sustained_weeks).apply(season)

    # Warning
    predictor = EpidemicPredictor(config.target_disease, alert_threshold_sd=config.alert_threshold_sd)
    warning = predictor.generate_warning(lgdi_res.lgdi, lgdi_res.timeline)

    # Optional report writing
    if output_dir is not None:
        rg = ReportGenerator(output_dir, config.target_disease)
        if not lgdi_res.lgdi.empty:
            rolled = LGDIComputer.rolling_windows(lgdi_res.lgdi, window=4, value_col="lgdi")
            rg.write_rolling4_weekly(rolled)
        rg.write_alerts(sustained)
        rg.write_performance_summary(model_metrics or {"note": "no model trained"})

    return SurveillanceResult(
        config=config,
        rdi_timeline=rdi_timeline,
        lgdi_result=lgdi_res,
        alerts=sustained,
        warning=warning,
        model_metrics=model_metrics,
    )
