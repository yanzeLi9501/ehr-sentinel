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
from ehr_sentinel.features.temporal import compute_next_targets
from ehr_sentinel.metrics.lgdi import LGDIComputer, LGDIResult, SlidingWindowLGDI
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
    """Run the full surveillance pipeline on an admissions DataFrame.

    When *train_xgb* is ``True`` and XGBoost is installed the pipeline trains
    **two** models — one predicting ``next_los_days`` and one predicting
    ``next_gap_days`` — then uses ``SlidingWindowLGDI`` (4-week sliding window,
    dual signed-MASE, COVID-positive exclusion) to compute the publication-grade
    LGDI.  When XGBoost is unavailable or training fails the pipeline falls back
    to the simple ``LGDIComputer`` (unsigned single-outcome MASE).
    """
    df = EHRLoader().from_dataframe(data)

    # Compute shift(−1) next-admission targets BEFORE feature building so
    # both share the same mrn + admission_date sort order (both reset index).
    df = compute_next_targets(df)

    # Features
    builder = FeatureBuilder(config)
    fm = builder.build(df)

    # Align next-admission target series to the feature-matrix index
    # (compute_next_targets and compute_visit_order both sort by mrn+date,
    # so the positional order matches after both reset the index).
    y_next_gap = pd.Series(df["next_gap_days"].values, index=fm.X.index, name="next_gap_days")
    y_next_los = pd.Series(df["next_los_days"].values, index=fm.X.index, name="next_los_days")

    vo_pass = (
        fm.meta["vo_pass"].astype(int)
        if "vo_pass" in fm.meta.columns
        else pd.Series(1, index=fm.X.index)
    )

    # ── Dual XGBoost models (gap + LOS) ─────────────────────────────────
    model_metrics: dict = {}
    _lgdi_mode = "simple"
    gap_pred: Optional[np.ndarray] = None
    pred_gap: Optional[np.ndarray] = None
    pred_los: Optional[np.ndarray] = None

    if train_xgb:
        try:
            from ehr_sentinel.models.trainer import XGBTrainer

            trainer = XGBTrainer()  # uses random_state=20260513 by default
            mask_gap = y_next_gap.notna() & (vo_pass == 1)
            mask_los = y_next_los.notna() & (vo_pass == 1)

            if mask_gap.sum() >= 30:
                cv = trainer.cross_validate(
                    fm.X.loc[mask_gap],
                    y_next_gap.loc[mask_gap],
                    groups=fm.groups.loc[mask_gap],
                    n_splits=min(5, max(2, int(fm.groups.loc[mask_gap].nunique()))),
                )
                model_metrics["cv_r2_gap"] = float(np.mean(cv["r2_test"]))
                model_metrics["cv_mae_gap"] = float(np.mean(cv["mae_test"]))
                tr_gap = trainer.train_gap_model(
                    fm.X.loc[mask_gap], y_next_gap.loc[mask_gap]
                )
                pred_gap = trainer.predict(tr_gap.model, fm.X)
                gap_pred = pred_gap  # kept for backward-compat (simple-LGDI fallback)
                model_metrics["device"] = tr_gap.device
                model_metrics["n_train"] = int(mask_gap.sum())

            if mask_los.sum() >= 30:
                tr_los = trainer.train_los_model(
                    fm.X.loc[mask_los], y_next_los.loc[mask_los]
                )
                pred_los = trainer.predict(tr_los.model, fm.X)

            if pred_gap is not None and pred_los is not None:
                _lgdi_mode = "sliding_window"

        except ImportError as e:
            warnings.warn(
                f"XGBoost unavailable; skipping rhythm model ({e}).",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception as e:  # pragma: no cover
            warnings.warn(
                f"Rhythm model failed: {e}; continuing without it.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Attach predictions / group for downstream metrics
    df_metrics = df.copy()
    df_metrics["comorbidity_group"] = fm.meta["comorbidity_group"].values

    if gap_pred is not None:
        df_metrics["gap_pred"] = gap_pred

    if _lgdi_mode == "sliding_window":
        df_metrics["pred_next_gap_days"] = pred_gap
        df_metrics["pred_next_los_days"] = pred_los
        df_metrics["resid_next_gap_days"] = (
            df_metrics["next_gap_days"] - df_metrics["pred_next_gap_days"]
        )
        df_metrics["resid_next_los_days"] = (
            df_metrics["next_los_days"] - df_metrics["pred_next_los_days"]
        )

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

    # LGDI — publication-grade 4-week sliding-window dual signed-MASE when
    # both XGBoost models are available; simple unsigned-MASE fallback otherwise.
    if _lgdi_mode == "sliding_window":
        swl = SlidingWindowLGDI(
            target_group=config.target_group,
            comorbidity_groups=list(config.comorbidity_groups.keys()),
            baseline_start=config.baseline_start,
            baseline_end=config.baseline_end,
            min_total_n=config.lgdi_min_window_n,
            min_group_n=config.lgdi_min_group_n,
        )
        swl.fit_baseline_scale(df_metrics)
        lgdi_res = swl.compute(df_metrics)
        model_metrics["lgdi_mode"] = "sliding_window_dual_signed_mase"
    else:
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

    # Alerts — ConsensusRule requires the simple-LGDI timeline format (group, S);
    # the sliding-window timeline has a different schema so we skip to sustained=empty.
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

    # Warning — identify_at_risk_groups expects long-format (group, S) table;
    # reshape the sliding-window score columns when necessary.
    if _lgdi_mode == "sliding_window" and not lgdi_res.timeline.empty:
        score_cols = [c for c in lgdi_res.timeline.columns if c.startswith("score_")]
        mase_for_warning = (
            lgdi_res.timeline[score_cols]
            .rename(columns={c: c[len("score_"):] for c in score_cols})
            .melt(var_name="group", value_name="S")
        ) if score_cols else pd.DataFrame()
    else:
        mase_for_warning = lgdi_res.timeline
    predictor = EpidemicPredictor(config.target_disease, alert_threshold_sd=config.alert_threshold_sd)
    warning = predictor.generate_warning(lgdi_res.lgdi, mase_for_warning)

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
