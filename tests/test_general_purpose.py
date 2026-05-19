"""Verify the package is disease-agnostic by swapping configs."""
import pandas as pd

from ehr_sentinel import EpidemicConfig, EHRLoader, FeatureBuilder, PearsonProfileCorrelation
from ehr_sentinel.features.adaptive import AutoFeatureEngineer, DiseaseDetector, LabPanelAdapter, build_adaptive_config


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


def test_adaptive_lab_panel_selects_dataset_specific_labs():
    df = pd.DataFrame({
        "mrn": ["A", "B", "C", "D"],
        "admission_date": pd.date_range("2020-01-01", periods=4),
        "WBC": [5.1, 6.2, 7.3, 8.4],
        "CRP": [None, None, None, None],
        "HGB": [130, 131, 129, 132],
    })
    spec = LabPanelAdapter(min_coverage=0.5, min_non_null=2).select(df)
    assert spec.detected == ["WBC", "CRP", "HGB"]
    assert spec.selected == ["WBC", "HGB"]


def test_adaptive_disease_selection_and_config():
    df = pd.DataFrame({
        "mrn": ["A", "A", "B", "B"],
        "admission_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-15", "2020-03-01"]),
        "discharge_date": pd.to_datetime(["2020-01-04", "2020-02-03", "2020-01-20", "2020-03-04"]),
        "icd10": ["J10", "B34", "I10", "J11"],
        "WBC": [5.0, 6.0, 7.0, 8.0],
    })
    detector = DiseaseDetector()
    counts = detector.detect(df)
    signals = detector.select(counts)
    assert [s.name for s in signals] == ["influenza", "other_viral"]
    labs = LabPanelAdapter(min_non_null=2).select(df)
    plan = AutoFeatureEngineer().plan(EHRLoader().from_dataframe(df), labs)
    cfg = build_adaptive_config(df, signals[0], plan)
    assert cfg.target_disease == "Influenza"
    assert cfg.lab_panel == ["WBC"]
    assert "250" in cfg.comorbidity_groups["Diabetes"]


def test_disease_detector_min_signal_count_threshold():
    """MIMIC-IV 2.2 has only 2 B34.2 (SARS) records, which should fall through
    to influenza rather than triggering a near-empty COVID-19 analysis."""
    detector = DiseaseDetector()
    # Below MIN_SIGNAL_COUNT → should fall through to influenza
    sigs = detector.select({"covid19": 2, "influenza": 1859, "other_viral": 300})
    assert sigs[0].name == "influenza", "Sparse COVID should not override flu"
    assert sigs[0].count == 1859
    assert len(sigs) == 2  # flu + other_viral parallel

    # At or above MIN_SIGNAL_COUNT → COVID should be selected
    sigs_covid = detector.select({"covid19": 82, "influenza": 5, "other_viral": 0})
    assert sigs_covid[0].name == "covid19"
    assert len(sigs_covid) == 1  # COVID only


def test_build_adaptive_config_single_year_span():
    """Datasets with dates all in one calendar year (e.g. CDSL shifted to 2100)
    should produce a valid baseline/monitoring split using half-year boundary."""
    df = pd.DataFrame({
        "mrn": [str(i) for i in range(40)],
        "admission_date": pd.date_range("2100-01-01", periods=40, freq="W"),
        "discharge_date": pd.date_range("2100-01-04", periods=40, freq="W"),
        "icd10": ["U071"] * 40,
    })
    from ehr_sentinel.features.adaptive import DiseaseSignal, FeaturePlan
    signal = DiseaseSignal("covid19", "COVID-19", ["U07.1"], 40)
    plan = FeaturePlan(lab_panel=[], min_visit_order=1, gap_cap_days=30, los_cap_days=60,
                       enhanced_features=False, reason="test")
    cfg = build_adaptive_config(df, signal, plan)
    # Baseline end must be strictly before monitoring start to avoid empty LGDI
    import pandas as _pd
    assert _pd.Timestamp(cfg.baseline_end) < _pd.Timestamp(cfg.monitoring_start)
