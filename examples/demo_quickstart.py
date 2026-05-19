"""Quickstart — synthetic data, no real EHR required."""
from ehr_sentinel import EpidemicConfig, generate_admissions, run_surveillance_pipeline


def main() -> None:
    config = EpidemicConfig(
        target_disease="COVID-19",
        reference_icd10_codes=["U07.1", "U07.2"],
        reference_years=[2020],
        reference_months=[1, 2, 3, 4],
        baseline_start="2016-01-01",
        baseline_end="2018-12-31",
        monitoring_start="2019-01-01",
        target_group="Respiratory",
        min_visit_order=2,
    )
    df = generate_admissions(n_patients=200, seed=42)
    print(f"Generated {len(df)} synthetic admissions for {df['mrn'].nunique()} patients.")

    result = run_surveillance_pipeline(df, config, train_xgb=False)
    print(result.summary())
    if result.warning.onset_week:
        print(f"Onset detected: {result.warning.onset_week.date()}")
    for g, v in result.warning.at_risk_groups:
        print(f"  at-risk group: {g} (mean MASE={v:.3f})")


if __name__ == "__main__":
    main()
