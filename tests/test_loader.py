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
