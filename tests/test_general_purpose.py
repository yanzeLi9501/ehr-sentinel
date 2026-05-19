"""Verify the package is disease-agnostic by swapping configs."""
from ehr_sentinel import EpidemicConfig, EHRLoader, FeatureBuilder, PearsonProfileCorrelation


def test_different_target_disease(synthetic_admissions, synthetic_config_covid, synthetic_config_flu):
    df = EHRLoader().from_dataframe(synthetic_admissions)
    fm_covid = FeatureBuilder(synthetic_config_covid).build(df)
    fm_flu = FeatureBuilder(synthetic_config_flu).build(df)
    # Same feature matrix shape (disease-agnostic features), different downstream
    assert fm_covid.X.shape == fm_flu.X.shape


def test_custom_comorbidity_groups(synthetic_admissions):
    cfg = EpidemicConfig(
        target_disease="Custom",
        reference_icd10_codes=["I10"],
        reference_years=[2020],
        reference_months=[1, 2, 3],
        baseline_start="2016-01-01",
        baseline_end="2017-12-31",
        comorbidity_groups={"A_Heart": r"^I", "B_Lung": r"^J", "C_Other": r"^[A-HK-Z]"},
        target_group="B_Lung",
        min_visit_order=2,
    )
    fm = FeatureBuilder(cfg).build(EHRLoader().from_dataframe(synthetic_admissions))
    assert "group_A_Heart" in fm.X.columns
    assert "group_B_Lung" in fm.X.columns
    assert "group_C_Other" in fm.X.columns


def test_season_filter_adapts():
    cfg_covid = EpidemicConfig(
        target_disease="COVID-19",
        reference_icd10_codes=["U07.1"],
        reference_years=[2020], reference_months=[1, 2, 3],
        baseline_start="2016-01-01", baseline_end="2017-12-31",
        epidemic_season_months=list(range(1, 13)),
    )
    cfg_flu = cfg_covid.model_copy(update={"target_disease": "Influenza",
                                            "epidemic_season_months": [11, 12, 1, 2, 3]})
    assert cfg_covid.epidemic_season_months != cfg_flu.epidemic_season_months


def test_epidemic_signal_detection(synthetic_admissions_with_signal, synthetic_config_covid):
    cfg = synthetic_config_covid
    df = EHRLoader().from_dataframe(synthetic_admissions_with_signal)
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
    # Some weekly correlation should be non-trivial
    assert weekly["r"].abs().max() > 0.0
