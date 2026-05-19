"""Demo: COVID-19 as one candidate epidemic."""
from ehr_sentinel import (
    PresetConfigs, generate_admissions, generate_epidemic_signal,
    run_surveillance_pipeline,
)


def main() -> None:
    config = PresetConfigs.covid_19()
    df = generate_admissions(n_patients=300, seed=42)
    df = generate_epidemic_signal(
        df,
        target_icd10=config.reference_icd10_codes,
        outbreak_start="2020-01-15",
        outbreak_end="2020-04-30",
        target_group=config.target_group,
        effect_size=0.6,
    )
    cfg = config.model_copy(update={"min_visit_order": 2})
    result = run_surveillance_pipeline(df, cfg, train_xgb=False)
    print(result.summary())


if __name__ == "__main__":
    main()
