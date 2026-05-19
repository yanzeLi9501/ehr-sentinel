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

All 32 tests use only synthetic data — no real EHR files needed.

## Aggregate-only multi-dataset validation

`scripts/run_multi_dataset_validation.py` validates that the package can read public PhysioNet/FluNet
tables plus local WHU 32k/42k prepared cohorts without copying source data into the repository. Where
available, it aggregates admission-level lab panels from source lab tables before running the
pipeline. It writes only aggregate JSON/Markdown summaries: row counts, patient counts, date ranges,
detected lab panels, RDI/LGDI week counts, sustained-alert counts, and warning dates.

The script expects data outside the package repository:

```text
Submit/
├── ehr-sentinel/          # this package repository
├── external_data/         # user-obtained public datasets; never committed
│   ├── flunet/
│   └── physionet/
└── NC_revision/           # optional local WHU prepared tables; never imported by package code
```

Run the full aggregate validation:

```bash
cd ehr-sentinel
pip install -e ".[lgdi,fhir,dev]"
python scripts/run_multi_dataset_validation.py
```

Run a fast smoke check or selected datasets:

```bash
# first 1,000 normalized admissions per EHR dataset
python scripts/run_multi_dataset_validation.py --max-rows 1000

# selected datasets only
python scripts/run_multi_dataset_validation.py --dataset whu32k_primary --dataset whu42k_cardiac

# optional XGBoost training; off by default for reproducible, fast ingestion/metric validation
python scripts/run_multi_dataset_validation.py --dataset whu32k_primary --train-xgb
```

Default outputs are written under `validation_outputs/`, which is git-ignored:

```text
validation_outputs/multi_dataset_validation_results.json
validation_outputs/multi_dataset_validation_results.md
```

Privacy and isolation rules:

- The script reads local data paths at runtime but does **not** copy source CSV/GZ files.
- The repository `.gitignore` blocks CSV, Excel, Parquet, HDF5, pickle/joblib, and validation-output files.
- `ehr_sentinel` package modules do **not** import `NC_revision`; the validation script is an external runner.
- README tables below are aggregate-only and contain no patient-level records.

Latest local validation summary:

| Dataset | Status | Rows tested | Patients | Date range | Labs detected | RDI weeks | LGDI weeks | Sustained alerts | Mode |
|---|---:|---:|---:|---|---|---:|---:|---:|---|
| `mimiciv_hosp` | passed | 546,028 | 223,452 | 2105-10-04 to 2214-12-15 | not joined in local run; `labevents.csv.gz` incomplete (`.part` only) | 0 | 4,069 | 0 | full pipeline |
| `mimiciv_ed` | passed | 425,087 | 205,504 | 2110-01-11 to 2212-04-05 | none; local ED release folder has vitalsign but no lab-result table | 0 | 853 | 0 | full pipeline |
| `nwicu_hosp` | passed | 61,843 | 25,923 | 2100-01-01 to 2201-07-17 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 0 | 431 | 0 | full pipeline |
| `cdsl_inpatient` | passed | 4,479 | 4,479 | 2100-09-16 to 2195-06-21 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 0 | 0 | 0 | full pipeline |
| `whu32k_primary` | passed | 50,781 | 31,802 | 2012-04-16 to 2020-05-24 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 257 | 254 | 10 | full pipeline |
| `whu42k_cardiac` | passed | 299,728 | 42,795 | 2007-03-06 to 2024-12-05 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 0 | 605 | 18 | full pipeline |
| `flunet_china_reference` | passed | 830 | — | 2008-12-29 to 2024-12-23 | aggregate reference series | — | — | — | reference only |

Notes:

- MIMIC-IV Hosp normally has laboratory events; this local checkout only contains
  `labevents.csv.gz.part`, so labs were not joined in the run above. If a complete
  `hosp/labevents.csv.gz` is present, the validation script automatically aggregates the configured
  WBC/CRP/HGB/ALB/CREA/GLU/K/Na panel by `hadm_id`.
- MIMIC-IV-ED in this local checkout contains vitalsign/medication tables but no lab-result table for
  the configured lab panel.
- NWICU and CDSL laboratory tables were joined successfully and detected the full configured panel.
  Pearson RDI may still be empty when the configured reference/target-group weeks do not meet the
  profile-correlation criteria; LGDI still validates admission rhythm compatibility.
- FluNet is an aggregate surveillance reference series rather than patient-level admissions data; the
  validation confirms it can be loaded and summarized without entering the EHR pipeline.
- WHU runs use local prepared cohorts and publish only aggregate counts/metrics in this README.

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
