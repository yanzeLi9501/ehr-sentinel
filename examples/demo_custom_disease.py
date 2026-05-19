"""Demo: user-defined target disease — anything you choose."""
from ehr_sentinel import EpidemicConfig, generate_admissions, run_surveillance_pipeline


def main() -> None:
    config = EpidemicConfig(
        target_disease="RSV",  # any user-chosen label
        reference_icd10_codes=["J21.0", "B97.4"],
        reference_years=[2019, 2020],
        reference_months=[10, 11, 12, 1, 2],
        baseline_start="2016-01-01",
        baseline_end="2017-12-31",
        target_group="Respiratory",
        epidemic_season_months=[10, 11, 12, 1, 2],
        min_visit_order=2,
    )
    df = generate_admissions(n_patients=200, seed=99)
    result = run_surveillance_pipeline(df, config, train_xgb=False)
    print(result.summary())


if __name__ == "__main__":
    main()
