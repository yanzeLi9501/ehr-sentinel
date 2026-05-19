"""Demo: Optuna tuning (small budget, synthetic data — API demonstration only)."""
from ehr_sentinel import EHRLoader, FeatureBuilder, PresetConfigs, generate_admissions, detect_device


def main() -> None:
    try:
        from ehr_sentinel.models.tuner import XGBTuner
    except ImportError as e:
        print(f"Optuna / xgboost not installed: {e}")
        return

    cfg = PresetConfigs.covid_19().model_copy(update={"min_visit_order": 2})
    df = EHRLoader().from_dataframe(generate_admissions(n_patients=120, seed=3))
    fm = FeatureBuilder(cfg).build(df)
    mask = fm.y_gap.notna()
    print(f"Device: {detect_device().device}")
    tuner = XGBTuner(n_trials=5)  # tiny budget for demo
    result = tuner.tune(fm.X.loc[mask], fm.y_gap.loc[mask], fm.groups.loc[mask], n_splits=3)
    print(f"Best CV R²: {result.best_value:.4f}")
    print(f"Best params: {result.best_params}")


if __name__ == "__main__":
    main()
