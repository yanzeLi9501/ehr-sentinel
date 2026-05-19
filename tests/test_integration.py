"""End-to-end pipeline tests — all input data is read from real CSV files on disk.

The CSV files are written by session-scoped conftest fixtures that call the
synthetic data generators.  This validates the full ingestion path:
  synthetic generator → CSV file → EHRLoader.from_csv() → pipeline.
"""
from ehr_sentinel import EHRLoader, run_surveillance_pipeline, generate_fhir_bundle
from ehr_sentinel.reporting.fhir_export import FHIRExporter


def test_full_pipeline_from_csv(synthetic_csv_with_signal, synthetic_config_covid, tmp_path):
    """Pipeline runs from a real CSV file (not a DataFrame passed in memory)."""
    df = EHRLoader().from_csv(synthetic_csv_with_signal)
    result = run_surveillance_pipeline(
        df,
        synthetic_config_covid,
        train_xgb=False,
        output_dir=tmp_path,
    )
    assert result.config.target_disease == "COVID-19"
    assert "lgdi" in result.lgdi_result.lgdi.columns or result.lgdi_result.lgdi.empty
    assert "alert" in result.alerts.columns
    # Output CSV files were written to output_dir
    written = list(tmp_path.glob("*.csv"))
    assert len(written) > 0, "ReportGenerator should have written at least one CSV"
    s = result.summary()
    assert "COVID-19" in s


def test_multi_disease_from_csv(synthetic_csv_with_signal, synthetic_config_covid, synthetic_config_flu):
    """Same CSV file, two disease configs → different results."""
    loader = EHRLoader()
    df = loader.from_csv(synthetic_csv_with_signal)
    r1 = run_surveillance_pipeline(df, synthetic_config_covid, train_xgb=False)
    r2 = run_surveillance_pipeline(df, synthetic_config_flu, train_xgb=False)
    assert r1.config.target_disease != r2.config.target_disease


def test_baseline_csv_no_signal(synthetic_csv, synthetic_config_covid):
    """Baseline CSV (no epidemic injected) should run without errors."""
    df = EHRLoader().from_csv(synthetic_csv)
    result = run_surveillance_pipeline(df, synthetic_config_covid, train_xgb=False)
    assert result.config.target_disease == "COVID-19"


def test_fhir_bundle_to_csv_to_pipeline(synthetic_config_covid, tmp_path):
    """FHIR Bundle → CSV on disk → EHRLoader.from_csv() → pipeline → MeasureReport."""
    bundle = generate_fhir_bundle(n_patients=20, seed=1, start_date="2019-01-01", end_date="2021-12-31")
    df_raw = EHRLoader().from_fhir_bundle(bundle)
    # Write to CSV (simulating a user exporting FHIR data)
    csv_path = tmp_path / "fhir_export.csv"
    df_raw.to_csv(csv_path, index=False)

    df = EHRLoader().from_csv(csv_path)
    assert len(df) > 0

    cfg = synthetic_config_covid.model_copy(update={"min_visit_order": 1})
    result = run_surveillance_pipeline(df, cfg, train_xgb=False)
    exp = FHIRExporter(cfg.target_disease)
    report = exp.to_measure_report(result.lgdi_result.lgdi, alerts=result.alerts)
    p = exp.write(report, tmp_path / "measure_report.json")
    assert p.exists()
    assert report["resourceType"] == "MeasureReport"
