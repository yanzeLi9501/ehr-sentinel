# Reference Manifest

The following files in the parent workspace (`NC_revision/` and
`NC_revision/DeepseekRevision/`) served **only as read-only design references** while
authoring the `ehr-sentinel` package. They are **never imported, modified, copied, or
embedded** in this repository.

`ehr-sentinel` is a clean-room reimplementation. All algorithmic logic was re-derived from
its specification; only the published formulas and parameter values were carried over.

| Reference file | Role |
|----------------|------|
| `NC_revision/run_lgdi_surveillance.py` | LGDI pipeline reference — comorbidity grouping (L65-78), temporal features (L140-156, L207-260), XGBoost training pattern (L262-365), MASE / LGDI computation (L367-516). |
| `NC_revision/run_weekly_rdi_42k.py` | Pearson profile correlation / RDI reference (L336-361). |
| `NC_revision/gpu_reproduction_test/scripts/comprehensive_reanalysis.py` | Enhanced feature engineering (L45-95) and GPU XGBoost training pattern (L33-40 device selection, L110-200 training). |
| `NC_revision/gpu_reproduction_test/scripts/tune_xgb_optuna.py` | Optuna TPE search space and tuned best parameters. |
| `NC_revision/gpu_reproduction_test/scripts/ppv_all_datasets.py` | Cross-cohort PPV evaluation pattern. |
| `NC_revision/DeepseekRevision/ehr_sentinel/profiles.py` | EHR profile vector / baseline pattern. |
| `NC_revision/DeepseekRevision/ehr_sentinel/lgdi.py` | LGDI class signature and MASE scoring pattern. |
| `NC_revision/DeepseekRevision/ehr_sentinel/consensus.py` | Consensus / season / sustained rule patterns. |
| `NC_revision/DeepseekRevision/ehr_sentinel/dashboard.py` | HTML dashboard rendering pattern. |
| `NC_revision/DeepseekRevision/pyproject.toml` | Dependency-structure reference. |

## Verification

`tests/test_isolation.py` greps every file under `src/ehr_sentinel/` and asserts that none of
the strings `NC_revision`, `gpu_reproduction_test`, or `DeepseekRevision` appear in any
import statement.
