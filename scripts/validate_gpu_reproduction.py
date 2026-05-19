"""User-run GPU reproduction validation.

This script is **not** part of the test suite. Users supply paths to their
own ``history_features.csv`` and ``train_data.csv`` (obtained independently
under their own data-use agreements) and the script compares actual model
metrics against the published benchmarks.

Usage:
    python scripts/validate_gpu_reproduction.py \
        --data-dir /path/to/your/data \
        --config broad

Configs:
    broad   — visit_order >= 5,  gap_cap = 30  → expects R² ≈ 0.541, MAE ≈ 6.31
    tuned   — visit_order >= 5,  gap_cap = 30  → expects R² 0.541–0.560, PPV@10 ≥ 0.47
    extreme — visit_order >= 20, gap_cap = 30  → expects R² ≈ 0.913

No data ships with the package; the user supplies all inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ehr_sentinel import EHRLoader, FeatureBuilder, EpidemicConfig
from ehr_sentinel.models.trainer import XGBTrainer


EXPECTED = {
    "broad":   {"min_visit_order": 5,  "gap_cap_days": 30, "r2_lo": 0.50, "r2_hi": 0.58, "mae_lo": 5.5, "mae_hi": 7.0},
    "tuned":   {"min_visit_order": 5,  "gap_cap_days": 30, "r2_lo": 0.54, "r2_hi": 0.58, "mae_lo": 5.5, "mae_hi": 7.0},
    "extreme": {"min_visit_order": 20, "gap_cap_days": 30, "r2_lo": 0.88, "r2_hi": 0.94, "mae_lo": 1.0, "mae_hi": 4.0},
}


def main() -> int:
    ap = argparse.ArgumentParser(description="User-run GPU reproduction validation.")
    ap.add_argument("--data-dir", required=True, help="Directory containing user-supplied EHR CSVs.")
    ap.add_argument("--csv-name", default="history_features.csv", help="Filename inside --data-dir.")
    ap.add_argument("--config", choices=list(EXPECTED), default="broad")
    ap.add_argument("--out", default="gpu_validation_report.json")
    args = ap.parse_args()

    csv_path = Path(args.data_dir) / args.csv_name
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. The user must supply their own data.", file=sys.stderr)
        return 2

    exp = EXPECTED[args.config]
    cfg = EpidemicConfig(
        target_disease="reproduction",
        reference_icd10_codes=["U07.1"],
        reference_years=[2020], reference_months=[1, 2, 3],
        baseline_start="2016-01-01", baseline_end="2018-12-31",
        min_visit_order=exp["min_visit_order"],
        gap_cap_days=exp["gap_cap_days"],
        enhanced_features=(args.config != "broad"),
    )

    df = EHRLoader().from_csv(csv_path)
    fm = FeatureBuilder(cfg).build(df)
    mask = fm.y_gap.notna() & (fm.meta.get("vo_pass", 1).astype(int) == 1)
    X, y, groups = fm.X.loc[mask], fm.y_gap.loc[mask], fm.groups.loc[mask]
    print(f"Training on {len(X)} admissions, {groups.nunique()} patients.")

    trainer = XGBTrainer()
    cv = trainer.cross_validate(X, y, groups=groups, n_splits=5)
    actual_r2 = float(np.mean(cv["r2_test"]))
    actual_mae = float(np.mean(cv["mae_test"]))
    passed = (exp["r2_lo"] <= actual_r2 <= exp["r2_hi"]) and (exp["mae_lo"] <= actual_mae <= exp["mae_hi"])

    report = {
        "config": args.config,
        "expected": exp,
        "actual": {"cv_r2_mean": actual_r2, "cv_mae_mean": actual_mae, "n_train": int(len(X))},
        "device": trainer.device_info.device,
        "passed": bool(passed),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
