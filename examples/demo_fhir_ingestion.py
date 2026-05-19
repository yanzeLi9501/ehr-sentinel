"""Demo: ingest synthetic FHIR R4 Bundle, then run surveillance."""
from ehr_sentinel import EHRLoader, PresetConfigs, generate_fhir_bundle, run_surveillance_pipeline


def main() -> None:
    bundle = generate_fhir_bundle(n_patients=50, seed=7, start_date="2019-01-01", end_date="2021-12-31")
    df = EHRLoader().from_fhir_bundle(bundle)
    print(f"Parsed {len(df)} encounters from synthetic FHIR Bundle.")
    cfg = PresetConfigs.covid_19().model_copy(update={"min_visit_order": 1})
    result = run_surveillance_pipeline(df, cfg, train_xgb=False)
    print(result.summary())


if __name__ == "__main__":
    main()
