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
- **Adaptive validation** — per-dataset lab-panel selection, ICD/text disease detection, and feature-plan selection.
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

Clone the repository and install with one command:

```bash
git clone https://github.com/yanzeLi9501/ehr-sentinel.git
cd ehr-sentinel
pip install -e ".[lgdi,fhir]"
```

Or install directly from GitHub without cloning:

```bash
pip install "ehr-sentinel[lgdi,fhir] @ git+https://github.com/yanzeLi9501/ehr-sentinel.git@main"
```

For development (includes pytest, ruff):

```bash
git clone https://github.com/yanzeLi9501/ehr-sentinel.git
cd ehr-sentinel
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
git clone https://github.com/yanzeLi9501/ehr-sentinel.git
cd ehr-sentinel
pip install -e ".[lgdi,dev]"
pytest tests/ -v
```

All 36 tests use only synthetic data — no real EHR files needed.

## Aggregate-only multi-dataset validation

`scripts/run_multi_dataset_validation.py` validates that the package can read public PhysioNet/FluNet
tables plus local WHU 32k/42k prepared cohorts without copying source data into the repository. It
does **not** force one universal laboratory panel onto all centers. Instead, it uses the package
adaptive helpers to:

- detect available lab tests per dataset and select only usable numeric, non-sparse, non-constant labs;
- detect epidemic targets from ICD/text evidence with priority: COVID-19 if present; otherwise
  influenza; if influenza and other viral infections both exist, run both in parallel; if neither
  exists, run an influenza fallback configuration;
- choose feature-engineering knobs per dataset (`min_visit_order`, LOS/gap caps, enhanced rolling/EMA
  features) based on cohort size, repeat-admission structure, and lab availability.

It writes only aggregate JSON/Markdown summaries: row counts, patient counts, date ranges, selected
lab panels, disease counts, feature-plan metadata, RDI/LGDI week counts, sustained-alert counts, and
warning dates.

The script expects data outside the package repository:

```text
Submit/
├── ehr-sentinel/          # this package repository
├── external_data/         # user-obtained public datasets; never committed
│   ├── flunet/
│   └── physionet/
└── NC_revision/           # optional local WHU prepared tables; never imported by package code
```

For a complete MIMIC-IV installation stored elsewhere, pass `--mimic-iv-root` instead of copying
`labevents.csv.gz` into this repository:

```text
D:\path\to\mimic-iv-2.2\mimic-iv-2.2\
└── hosp\
    ├── admissions.csv.gz
    ├── diagnoses_icd.csv.gz
    ├── d_labitems.csv.gz
    └── labevents.csv.gz
```

Run the full aggregate validation:

```bash
git clone https://github.com/yanzeLi9501/ehr-sentinel.git
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

# complete MIMIC-IV hospital run with lab events from an external data folder
python scripts/run_multi_dataset_validation.py \
  --dataset mimiciv_hosp \
  --mimic-iv-root "D:\path\to\mimic-iv-2.2\mimic-iv-2.2"

# optional XGBoost training/tuning; off by default for reproducible, fast ingestion/metric validation
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

Latest local validation summary (v0.3 — corrected disease selection, with signal diagnostics):

| Dataset | Analysis | Rows tested | Patients | Disease records | Labs selected | LGDI weeks | Peak S | Alert threshold S | Sustained alerts |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `mimiciv_hosp` | Influenza + Other viral ¹ | 431,231 | 180,733 | 1,859 flu / 1,200 other | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | TBD² | — | — | 0 |
| `mimiciv_ed` | Influenza | 425,087 | 205,504 | 1,731 | none (no lab table in ED release) | 3,886 | — | — | 0 |
| `mimiciv_ed` | Other viral infection | 425,087 | 205,504 | 2,254 | none (no lab table in ED release) | 3,886 | — | — | 0 |
| `nwicu_hosp` | COVID-19 | 61,843 | 25,923 | 82 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 437 | — | — | 0 |
| `eicu_crd` | Influenza | 200,859 | 139,367 | 45 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 18 | — | — | 0 |
| `eicu_crd` | Other viral infection | 200,859 | 139,367 | 421 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 18 | — | — | 0 |
| `cdsl_inpatient` | COVID-19 ³ | 4,479 | 4,479 | 4,479 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 0 ³ | — | — | 0 |
| `whu32k_primary` | COVID-19 | 50,781 | 31,802 | 13 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 257 | >1.5 SD | 1.5 SD | 11 |
| `whu42k_cardiac` | COVID-19 | 299,728 | 42,795 | 311 | WBC, CRP, HGB, ALB, CREA, GLU, K, Na | 605 | >1.5 SD | 1.5 SD | 18 |
| `flunet_china_reference` | aggregate reference series | 830 weeks | — | — | — | — | — | — | — |

¹ **MIMIC-IV Hosp correction**: v0.2 incorrectly classified MIMIC-IV 2.2 as a COVID-19 analysis because 2 records
coded B34.2 (SARS) matched the COVID regex. MIMIC-IV 2.2 contains **zero U07 (COVID-19)** ICD-10 codes; B34.2 is
SARS, not COVID-19. The adaptive `DiseaseDetector` now requires ≥ 10 matching records before promoting a disease
target (`MIN_SIGNAL_COUNT = 10`). MIMIC-IV 2.2 therefore falls through correctly to **Influenza (1,859 records) +
Other viral (1,200 records)** parallel analyses. Full re-run results pending (TBD²).

² TBD: MIMIC-IV full re-run with `--mimic-iv-root` was not re-executed after the disease-selection correction. Run
`python scripts/run_multi_dataset_validation.py --dataset mimiciv_hosp --mimic-iv-root <path>` locally to generate
updated LGDI weeks and peak-S diagnostics.

³ **CDSL LGDI = 0 weeks**: CDSL is a cross-sectional COVID-19 cohort — each patient appears exactly once
(`repeat_ratio ≈ 0`). The LGDI metric is based on readmission-gap rhythm between repeat hospital visits. With no
repeat admissions, no gap signal is computable and LGDI is empty. The dataset is still useful for validating
lab-panel ingestion and COVID ICD coding. Alerts = 0 is expected and correct.

**Why do only WHU cohorts produce sustained alerts?**

The alert engine fires when ≥ 2 comorbidity groups simultaneously exceed their 1.5 SD baseline for ≥ 2
consecutive weeks. WHU32k and WHU42k contain real longitudinal Chinese hospital data that spans COVID-19 epidemic
waves in 2020–2022 (including Omicron peak), producing genuine temporally-clustered weekly admission surges. Other
public datasets do not generate sustained alerts for known structural reasons:

| Dataset | Reason alerts = 0 |
|---|---|
| MIMIC-IV Hosp | Sparse flu/viral records over 90-year shifted date range; weekly counts near 0 |
| MIMIC-IV ED | 1,731 flu over 10+ years; no labs; weekly signal too diffuse to exceed threshold |
| NWICU | 82 COVID / 61,843 rows (0.13%); too sparse for weekly excess above 1.5 SD |
| eICU | 2014–2015 synthetic dates; only 18 LGDI weeks; time window too short |
| CDSL | No readmissions → no gap signal → LGDI empty (see ³ above) |

The `peak_lgdi_signal_S` and `alert_threshold_S` fields in the validation JSON show the actual signal magnitude
vs the threshold for each dataset — use these to diagnose alert sensitivity for any new dataset.

Notes:

- MIMIC-IV Hosp was validated with a complete external MIMIC-IV 2.2 root supplied via
  `--mimic-iv-root`; the package repository still does not contain or copy `labevents.csv.gz`.
- MIMIC-IV-ED in this local checkout contains vitalsign/medication tables but no lab-result table for
  an adaptive lab panel.
- NWICU, eICU, CDSL, MIMIC-IV Hosp, and WHU laboratory tables were joined successfully. The selected
  lab panel above is the panel that passed per-dataset coverage/variance checks in this local run.
- Pearson RDI may still be empty when the selected disease/reference window and target-group weeks do
  not meet profile-correlation criteria; LGDI still validates admission rhythm compatibility.
- FluNet is an aggregate surveillance reference series rather than patient-level admissions data; the
  validation confirms it can be loaded and summarized without entering the EHR pipeline.
- WHU runs use local prepared cohorts and publish only aggregate counts/metrics in this README.

## Package layout

```
src/ehr_sentinel/
├── utils/          # EpidemicConfig (Pydantic v2), GPU detection, validation helpers
├── data/           # EHRLoader, synthetic generators, FHIR parser, terminology mapper
├── features/       # ComorbidityGrouper, adaptive lab/disease/feature planning, FeatureBuilder
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
