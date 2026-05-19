"""Demo: Influenza surveillance with seasonal window."""
from ehr_sentinel import (
    PresetConfigs, generate_admissions, generate_epidemic_signal,
    run_surveillance_pipeline,
)


def main() -> None:
    config = PresetConfigs.influenza_seasonal()
    df = generate_admissions(n_patients=300, seed=21,
                             start_date="2014-01-01", end_date="2019-12-31")
    df = generate_epidemic_signal(
        df,
        target_icd10=config.reference_icd10_codes,
        outbreak_start="2017-12-01",
        outbreak_end="2018-02-28",
        target_group=config.target_group,
        effect_size=0.5,
    )
    cfg = config.model_copy(update={"min_visit_order": 2})
    result = run_surveillance_pipeline(df, cfg, train_xgb=False)
    print(result.summary())


if __name__ == "__main__":
    main()
