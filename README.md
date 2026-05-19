# ehr-sentinel

**General-purpose EHR epidemic surveillance toolkit** — a clean, MIT-licensed Python package that
encapsulates a retrospective XGBoost rhythm-prediction pipeline together with two surveillance
metrics — **Pearson profile correlation (RDI)** and the **LOS–Gap Deviation Index (LGDI)** — into a
disease-agnostic early-warning system for epidemic monitoring from electronic health records.

> COVID-19 is just one *candidate* epidemic. You configure the target disease, the reference
> profile, the comorbidity groups, the baseline window, the alert thresholds, and the seasonal
> window. The package never assumes anything beyond the configuration you supply.

## Highlights

- **Disease-agnostic** — every disease-specific parameter is exposed via `EpidemicConfig`
  (Pydantic v2). Switch from COVID-19 to influenza to RSV by swapping the config.
- **Multi-source ingestion** — CSV (auto-configuring column detection) and FHIR R4 Bundles
  (Patient / Encounter / Condition / Observation).
- **Terminology mapping** — configurable ICD-10 ↔ SNOMED ↔ LOINC crosswalks.
- **Local GPU XGBoost** — automatic CUDA detection with graceful CPU fallback.
- **Optuna tuning** — TPE search for hyperparameters and PPV-targeted objectives.
- **Surveillance metrics** — Pearson RDI and LGDI with bootstrap confidence intervals.
- **Consensus alerting** — configurable `k`-of-`n`, sustained, and seasonally-gated rules.
- **Reports** — CSV tables, HTML dashboards, FHIR R4 `MeasureReport` export.

## Requirements

- Python ≥ 3.10
- Core dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `pydantic >= 2`
- Optional `[lgdi]`: `xgboost >= 1.7`, `optuna >= 3`, `scikit-learn >= 1.2`
- Optional `[fhir]`: `fhir.resources >= 7`

## Installation

```bash
# minimal
pip install ehr-sentinel

# full (XGBoost tuning + FHIR)
pip install "ehr-sentinel[lgdi,fhir]"

# development (includes pytest, ruff)
pip install -e ".[lgdi,fhir,dev]"
```

The `lgdi` extra installs `xgboost`, `optuna`, and `scikit-learn`. The `fhir` extra installs `fhir.resources`.

## Quickstart (synthetic data, no real EHR required)

```python
from ehr_sentinel import EpidemicConfig, generate_admissions, run_surveillance_pipeline

config = EpidemicConfig(
    target_disease="COVID-19",
    reference_icd10_codes=["U07.1", "U07.2"],
    reference_years=[2020],
    reference_months=[1, 2, 3],
    baseline_start="2016-01-01",
    baseline_end="2018-12-31",
    monitoring_start="2019-01-01",
    epidemic_season_months=list(range(1, 13)),
)

df = generate_admissions(n_patients=500, seed=42)
result = run_surveillance_pipeline(df, config)
print(result.summary())
```

See [examples/](examples/) for COVID, Influenza, custom-disease, FHIR-ingestion, and GPU-tuning
walkthroughs.

## Running tests

```bash
pip install -e ".[lgdi,dev]"
pytest tests/ -v
```

All 27 tests use only synthetic data — no real EHR files needed.

## Package layout

```
src/ehr_sentinel/
├── utils/          # EpidemicConfig (Pydantic v2), GPU detection, validation helpers
├── data/           # EHRLoader, synthetic generators, FHIR parser, terminology mapper
├── features/       # ComorbidityGrouper, temporal/seasonal features, FeatureBuilder
├── models/         # XGBTrainer, XGBTuner (Optuna), ModelPersistence
├── metrics/        # PearsonProfileCorrelation (RDI), LGDIComputer, AlertEvaluator
├── alerts/         # ConsensusRule, SeasonFilter, SustainedRule, EpidemicPredictor
├── reporting/      # CSV tables, HTML dashboard, FHIR R4 MeasureReport export
└── pipeline.py     # run_surveillance_pipeline() — one-call entry point
```

## Data policy — ZERO embedded data

This repository contains **no patient data**, **no real EHR records**, and **no copies of
public datasets**. All tests and examples use **synthetic data only**, generated programmatically
by `ehr_sentinel.data.synthetic`.

Public datasets (MIMIC-IV, eICU, PhysioNet, WHO FluNet, etc.) are **referenced** in
documentation but are **never redistributed**. Each dataset has its own data use agreement,
and users must obtain it independently.

Verification of expected XGBoost reproduction metrics (R²≈0.541, MAE≈6.31 for the broad
`visit_order≥5`, `gap_cap=30` configuration; R²≈0.913 for `visit_order≥20`) is performed by the
**user-run** script `scripts/validate_gpu_reproduction.py`, which requires the user to supply
their own data paths. The script is not invoked as part of the test suite.

## Repository isolation

`ehr-sentinel` is a **clean-room reimplementation**. It does **not import** from, and does
**not depend on**, any other repository. The original analysis scripts in `NC_revision/` and
`DeepseekRevision/` served only as design references and are listed in
[REFERENCE_MANIFEST.md](REFERENCE_MANIFEST.md). They are never modified, copied, or imported.

## Retrospective POC disclaimer

This package is a **retrospective proof-of-concept**. It has **not** been validated for
prospective surveillance, has **not** been cleared for clinical decision support, and is **not**
a medical device. Any deployment requires local institutional review, recalibration on local
baseline data, and ongoing human oversight.

## License

MIT — see [LICENSE](LICENSE).
