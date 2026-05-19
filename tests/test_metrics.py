import numpy as np

from ehr_sentinel import (
    EHRLoader,
    FeatureBuilder,
    PearsonProfileCorrelation,
    LGDIComputer,
    AlertEvaluator,
)


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
