"""End-to-end pipeline tests on synthetic data."""
from ehr_sentinel import run_surveillance_pipeline, generate_admissions, generate_fhir_bundle, EHRLoader
from ehr_sentinel.reporting.fhir_export import FHIRExporter


def test_full_pipeline_synthetic(synthetic_admissions_with_signal, synthetic_config_covid, tmp_path):
    result = run_surveillance_pipeline(
        synthetic_admissions_with_signal,
        synthetic_config_covid,
        train_xgb=False,
        output_dir=tmp_path,
    )
    assert result.config.target_disease == "COVID-19"
    assert "lgdi" in result.lgdi_result.lgdi.columns or result.lgdi_result.lgdi.empty
    assert "alert" in result.alerts.columns
    s = result.summary()
    assert "COVID-19" in s


def test_multi_disease_swap(synthetic_admissions_with_signal, synthetic_config_covid, synthetic_config_flu):
    r1 = run_surveillance_pipeline(synthetic_admissions_with_signal, synthetic_config_covid, train_xgb=False)
    r2 = run_surveillance_pipeline(synthetic_admissions_with_signal, synthetic_config_flu, train_xgb=False)
    assert r1.config.target_disease != r2.config.target_disease


def test_fhir_round_trip(synthetic_config_covid, tmp_path):
    bundle = generate_fhir_bundle(n_patients=20, seed=1, start_date="2019-01-01", end_date="2021-12-31")
    df = EHRLoader().from_fhir_bundle(bundle)
    assert len(df) > 0
    cfg = synthetic_config_covid.model_copy(update={"min_visit_order": 1})
    result = run_surveillance_pipeline(df, cfg, train_xgb=False)
    exp = FHIRExporter(cfg.target_disease)
    report = exp.to_measure_report(result.lgdi_result.lgdi, alerts=result.alerts)
    p = exp.write(report, tmp_path / "measure_report.json")
    assert p.exists()
    assert report["resourceType"] == "MeasureReport"
