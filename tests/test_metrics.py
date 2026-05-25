import math

import numpy as np
import pandas as pd

from ehr_sentinel import (
    EHRLoader,
    FeatureBuilder,
    PearsonProfileCorrelation,
    LGDIComputer,
    SlidingWindowLGDI,
    AlertEvaluator,
    CUSUMRule,
    EWMARule,
    SeasonalAdjustedRule,
    MultiScaleRule,
)
from ehr_sentinel.features.temporal import compute_next_targets


def test_pearson_rdi(synthetic_admissions_with_signal, synthetic_config_covid):
    cfg = synthetic_config_covid
    df = EHRLoader().from_dataframe(synthetic_admissions_with_signal)
    # Need comorbidity_group for grouping
    fm = FeatureBuilder(cfg).build(df)
    df["comorbidity_group"] = fm.meta["comorbidity_group"].values

    p = PearsonProfileCorrelation(lab_panel=cfg.lab_panel, target_group=cfg.target_group, min_admissions=3)
    ref, bmean, bstd = p.build_reference_profile(
        df,
        reference_icd10_codes=cfg.reference_icd10_codes,
        reference_years=cfg.reference_years,
        reference_months=cfg.reference_months,
    )
    weekly = p.weekly_correlation(df, ref, bmean, bstd)
    rdi = p.compute_rdi(weekly)
    assert "rdi" in rdi.columns
    assert len(rdi) > 0


def test_lgdi_basic(synthetic_admissions, synthetic_config_covid):
    cfg = synthetic_config_covid
    df = EHRLoader().from_dataframe(synthetic_admissions)
    fm = FeatureBuilder(cfg).build(df)
    df["comorbidity_group"] = fm.meta["comorbidity_group"].values

    lgdi = LGDIComputer(target_group=cfg.target_group,
                        baseline_start=cfg.baseline_start, baseline_end=cfg.baseline_end,
                        min_admissions=3)
    res = lgdi.run(df, y_col="gap")
    assert len(res.timeline) > 0
    assert len(res.lgdi) > 0
    assert "lgdi" in res.lgdi.columns


def test_alert_evaluator():
    ae = AlertEvaluator()
    alerts = np.array([1, 1, 0, 0, 1])
    truth = np.array([1, 0, 0, 1, 1])
    r = ae.evaluate(alerts, truth)
    assert r.tp == 2 and r.fp == 1 and r.fn == 1 and r.tn == 1
    assert 0.0 <= r.ppv <= 1.0


def test_ppv_at_top_k():
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
    truth = np.array([0, 1, 0, 1, 0])
    p = AlertEvaluator.ppv_at_top_k(scores, truth, k_pct=40)
    assert p == 1.0


def test_compute_next_targets():
    """compute_next_targets should shift LOS and gap forward by one visit per patient."""
    df = pd.DataFrame({
        "mrn": ["A", "A", "A", "B", "B"],
        "admission_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01",
                                          "2020-01-15", "2020-03-15"]),
        "discharge_date": pd.to_datetime(["2020-01-05", "2020-02-05", "2020-03-05",
                                          "2020-01-20", "2020-03-20"]),
        "los": [4.0, 4.0, 4.0, 5.0, 5.0],
        "gap": [27.0, 24.0, np.nan, 54.0, np.nan],
    })
    out = compute_next_targets(df)
    # Patient A: visit 0 → next_los=4, visit 1 → next_los=4, visit 2 → NaN
    a = out[out["mrn"] == "A"].sort_values("admission_date")
    assert a.iloc[0]["next_los_days"] == 4.0
    assert a.iloc[1]["next_los_days"] == 4.0
    assert math.isnan(a.iloc[2]["next_los_days"])
    # Last visit per patient always NaN
    b = out[out["mrn"] == "B"].sort_values("admission_date")
    assert math.isnan(b.iloc[1]["next_los_days"])


def test_sliding_window_lgdi_signed_direction():
    """SlidingWindowLGDI: respiratory group with elevated LOS residual should produce positive LGDI."""
    rng = np.random.default_rng(42)
    n = 600
    dates = pd.date_range("2019-01-01", periods=n, freq="3D")
    groups = rng.choice(["Respiratory", "Cardiovascular", "Hypertension",
                         "Diabetes", "Cerebrovascular", "Renal"], size=n)
    # Respiratory: elevated LOS residual (positive direction)
    resid_los = np.where(groups == "Respiratory", 2.0, 0.0) + rng.normal(0, 0.2, n)
    resid_gap = rng.normal(0, 0.2, n)
    df = pd.DataFrame({
        "mrn": [f"P{i % 80}" for i in range(n)],
        "admission_date": dates,
        "comorbidity_group": groups,
        "resid_next_los_days": resid_los,
        "resid_next_gap_days": resid_gap,
        "is_covid_positive": False,
    })
    sw = SlidingWindowLGDI(
        target_group="Respiratory",
        baseline_start="2019-01-01",
        baseline_end="2019-06-30",
        min_total_n=5,
        min_group_n=2,
    )
    sw.fit_baseline_scale(df)
    result = sw.compute(df, first_anchor="2019-07-07")
    valid = result.lgdi
    assert len(valid) > 0, "Should produce at least one valid LGDI window"
    assert valid["lgdi"].mean() > 0, "Respiratory elevation should yield positive LGDI mean"


def test_cusum_rule_detects_step_change():
    """CUSUMRule should trigger after a sustained step-up in LGDI."""
    baseline = pd.Series([0.1, -0.1, 0.2, -0.2, 0.05, 0.0, -0.05, 0.1] * 4)
    signal = pd.Series([0.1] * 16 + [3.0] * 8)  # step up after 16 weeks
    rule = CUSUMRule(k_sigma=0.5, h_sigma=4.0)
    rule.fit(baseline)
    result = rule.apply(signal)
    assert "cusum" in result.columns
    assert "alert" in result.columns
    # Alert should fire somewhere in the elevated region
    assert result["alert"].iloc[16:].sum() > 0, "CUSUM should detect step change"
    # No alert in stable baseline region
    assert result["alert"].iloc[:14].sum() == 0, "No false alarm in stable region"


def test_ewma_rule_detects_elevation():
    """EWMARule should fire during sustained LGDI elevation."""
    baseline = pd.Series([0.0] * 20)
    signal = pd.Series([0.0] * 10 + [5.0] * 10)
    rule = EWMARule(lam=0.2, L=3.0)
    rule.fit(baseline)
    result = rule.apply(signal)
    assert "ewma" in result.columns
    # Alert must fire in the elevated region
    assert result["alert"].iloc[10:].sum() > 0


def test_seasonal_adjusted_rule_month_thresholds():
    """SeasonalAdjustedRule should build 12 month-specific thresholds."""
    dates = pd.date_range("2016-01-01", periods=104, freq="W")
    vals = pd.Series(np.random.default_rng(7).normal(0, 1, 104), index=range(104))
    rule = SeasonalAdjustedRule(threshold_sd=1.5)
    rule.fit(vals, date_index=pd.Series(dates),
             baseline_start="2016-01-01", baseline_end="2017-12-31")
    assert len(rule._month_thresholds) == 12
    result = rule.apply(vals, date_index=pd.Series(dates))
    assert "threshold" in result.columns
    assert "alert" in result.columns


def test_multi_scale_rule():
    """MultiScaleRule should OR 4-week and 2-week thresholds."""
    baseline = pd.Series([0.0] * 20)
    signal = pd.Series([0.0] * 10 + [4.0, 4.0, 0.0, 0.0] + [0.0] * 6)
    rule = MultiScaleRule(threshold_sd=1.5)
    rule.fit(baseline)
    result = rule.apply(signal)
    assert "lgdi_4w" in result.columns
    assert "lgdi_2w" in result.columns
    # The 4.0 spike weeks should trigger
    assert result["alert"].iloc[10:12].sum() > 0


def test_pearson_rdi(synthetic_admissions_with_signal, synthetic_config_covid):
    cfg = synthetic_config_covid
    df = EHRLoader().from_dataframe(synthetic_admissions_with_signal)
    # Need comorbidity_group for grouping
    fm = FeatureBuilder(cfg).build(df)
    df["comorbidity_group"] = fm.meta["comorbidity_group"].values

    p = PearsonProfileCorrelation(lab_panel=cfg.lab_panel, target_group=cfg.target_group, min_admissions=3)
    ref, bmean, bstd = p.build_reference_profile(
        df,
        reference_icd10_codes=cfg.reference_icd10_codes,
        reference_years=cfg.reference_years,
        reference_months=cfg.reference_months,
    )
    weekly = p.weekly_correlation(df, ref, bmean, bstd)
    rdi = p.compute_rdi(weekly)
    assert "rdi" in rdi.columns
    assert len(rdi) > 0


def test_lgdi_basic(synthetic_admissions, synthetic_config_covid):
    cfg = synthetic_config_covid
    df = EHRLoader().from_dataframe(synthetic_admissions)
    fm = FeatureBuilder(cfg).build(df)
    df["comorbidity_group"] = fm.meta["comorbidity_group"].values

    lgdi = LGDIComputer(target_group=cfg.target_group,
                        baseline_start=cfg.baseline_start, baseline_end=cfg.baseline_end,
                        min_admissions=3)
    res = lgdi.run(df, y_col="gap")
    assert len(res.timeline) > 0
    assert len(res.lgdi) > 0
    assert "lgdi" in res.lgdi.columns


def test_alert_evaluator():
    ae = AlertEvaluator()
    alerts = np.array([1, 1, 0, 0, 1])
    truth = np.array([1, 0, 0, 1, 1])
    r = ae.evaluate(alerts, truth)
    assert r.tp == 2 and r.fp == 1 and r.fn == 1 and r.tn == 1
    assert 0.0 <= r.ppv <= 1.0


def test_ppv_at_top_k():
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
    truth = np.array([0, 1, 0, 1, 0])
    p = AlertEvaluator.ppv_at_top_k(scores, truth, k_pct=40)
    assert p == 1.0
