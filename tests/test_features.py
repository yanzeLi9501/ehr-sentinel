from ehr_sentinel import FeatureBuilder, ComorbidityGrouper, EHRLoader


def test_comorbidity_grouper(synthetic_admissions):
    grouper = ComorbidityGrouper()
    out = grouper.assign(synthetic_admissions["icd10"])
    assert "comorbidity_group" in out.columns
    assert out["comorbidity_group"].notna().sum() > 0
    for g in grouper.group_names:
        assert f"group_{g}" in out.columns


def test_feature_builder_no_nan(synthetic_admissions, synthetic_config_covid):
    df = EHRLoader().from_dataframe(synthetic_admissions)
    fm = FeatureBuilder(synthetic_config_covid).build(df)
    assert fm.X.notna().all().all()
    assert len(fm.X) == len(df)
    assert fm.X.shape[1] > 30  # enhanced features active


def test_feature_builder_base_set(synthetic_admissions, synthetic_config_covid):
    cfg = synthetic_config_covid.model_copy(update={"enhanced_features": False})
    df = EHRLoader().from_dataframe(synthetic_admissions)
    fm_enh = FeatureBuilder(synthetic_config_covid).build(df)
    fm_base = FeatureBuilder(cfg).build(df)
    assert fm_base.X.shape[1] < fm_enh.X.shape[1]
