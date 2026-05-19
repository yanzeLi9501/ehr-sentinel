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


def test_auto_configure_detects_chinese_admission_and_lab_columns():
    df = pd.DataFrame({
        "病案号": ["A", "A", "B"],
        "入院时间": ["2020-01-01", "2020-02-01", "2020-01-15"],
        "出院时间": ["2020-01-05", "2020-02-03", "2020-01-20"],
        "主要诊断": ["肺炎", "冠心病", "糖尿病"],
        "白细胞": [6.0, 7.1, 5.4],
        "超敏C反应蛋白": [2.0, 3.5, 1.5],
        "血红蛋白": [130, 120, 110],
        "白蛋白": [40, 38, 35],
        "肌酐": [80, 88, 90],
        "空腹血糖": [5.1, 6.2, 8.0],
        "钾": [4.0, 4.2, 3.8],
        "钠": [140, 138, 136],
    })
    loader = EHRLoader()
    profile = loader.auto_configure(df)
    assert profile.column_map["mrn"] == "病案号"
    assert profile.column_map["admission_date"] == "入院时间"
    assert profile.column_map["discharge_date"] == "出院时间"
    assert profile.column_map["diagnosis_text"] == "主要诊断"
    assert set(profile.detected_labs) == {"WBC", "CRP", "HGB", "ALB", "CREA", "GLU", "K", "Na"}
    out = loader.from_dataframe(df)
    assert {"WBC", "CRP", "HGB", "ALB", "CREA", "GLU", "K", "Na"}.issubset(out.columns)
    assert out["admission_date"].dtype.kind == "M"


def test_csv_roundtrip_preserves_icd10(synthetic_csv_with_signal):
    """ICD-10 codes survive the CSV write/read cycle unchanged."""
    df = EHRLoader().from_csv(synthetic_csv_with_signal)
    # The epidemic signal injected U07.1 — must survive CSV roundtrip
    assert df["icd10"].str.contains("U07.1").any()
