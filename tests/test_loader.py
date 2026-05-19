import pandas as pd

from ehr_sentinel import EHRLoader


def test_from_dataframe_normalizes(synthetic_admissions):
    loader = EHRLoader()
    df = loader.from_dataframe(synthetic_admissions)
    assert "mrn" in df.columns
    assert "admission_date" in df.columns
    assert "visit_order" in df.columns
    assert "los" in df.columns
    assert "gap" in df.columns
    assert df["admission_date"].dtype.kind == "M"


def test_from_csv_matches_dataframe(synthetic_csv, synthetic_admissions):
    """EHRLoader.from_csv() must produce the same normalised result as from_dataframe()."""
    loader = EHRLoader()
    df_csv = loader.from_csv(synthetic_csv)
    df_mem = loader.from_dataframe(synthetic_admissions)

    assert set(df_csv.columns) == set(df_mem.columns), "Column sets diverge between CSV and in-memory paths"
    assert len(df_csv) == len(df_mem)
    # MRNs match (order may differ after sort-by-date normalisation)
    assert sorted(df_csv["mrn"].unique()) == sorted(df_mem["mrn"].unique())
    assert df_csv["admission_date"].dtype.kind == "M"


def test_from_csv_with_signal(synthetic_csv_with_signal, synthetic_config_covid):
    """Pipeline can be driven purely from a CSV file — no in-memory DataFrame required."""
    from ehr_sentinel import run_surveillance_pipeline

    df = EHRLoader().from_csv(synthetic_csv_with_signal)
    result = run_surveillance_pipeline(df, synthetic_config_covid, train_xgb=False)
    assert result.config.target_disease == "COVID-19"
    assert "alert" in result.alerts.columns


def test_auto_configure_detects_columns():
    df = pd.DataFrame({
        "Patient_ID": ["A", "A", "B"],
        "AdmitDate": ["2020-01-01", "2020-02-01", "2020-01-15"],
        "DischargeDate": ["2020-01-05", "2020-02-03", "2020-01-20"],
        "ICD_10": ["J11.1", "I10", "E11.9"],
        "Hemoglobin": [130, 120, 110],
    })
    loader = EHRLoader()
    profile = loader.auto_configure(df)
    assert "mrn" in profile.column_map
    assert "admission_date" in profile.column_map
    assert "icd10" in profile.column_map
    assert "HGB" in profile.detected_labs
    out = loader.from_dataframe(df)
    assert "mrn" in out.columns
    assert out["mrn"].iloc[0] == "A"


def test_csv_roundtrip_preserves_icd10(synthetic_csv_with_signal):
    """ICD-10 codes survive the CSV write/read cycle unchanged."""
    df = EHRLoader().from_csv(synthetic_csv_with_signal)
    # The epidemic signal injected U07.1 — must survive CSV roundtrip
    assert df["icd10"].str.contains("U07.1").any()
