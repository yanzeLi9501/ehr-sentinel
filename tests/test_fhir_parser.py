from ehr_sentinel import EHRLoader
from ehr_sentinel.data.fhir_parser import FHIRParser


def test_fhir_parser_basic(synthetic_fhir_bundle):
    parser = FHIRParser()
    df = parser.parse(synthetic_fhir_bundle)
    assert len(df) > 0
    assert "mrn" in df.columns
    assert "icd10" in df.columns
    assert df["admission_date"].notna().sum() > 0


def test_fhir_via_loader(synthetic_fhir_bundle):
    loader = EHRLoader()
    df = loader.from_fhir_bundle(synthetic_fhir_bundle)
    assert "los" in df.columns
    assert "gap" in df.columns
    assert "visit_order" in df.columns
