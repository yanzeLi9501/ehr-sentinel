"""CSV report generator. All report filenames include the target disease."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


class ReportGenerator:
    def __init__(self, output_dir: str | Path, target_disease: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_disease = target_disease
        self._slug = _safe_name(target_disease)

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{self._slug}__{name}"

    def write_rolling4_weekly(self, df: pd.DataFrame) -> Path:
        p = self._path("rolling4_weekly.csv")
        df.to_csv(p, index=False)
        return p

    def write_threshold_table(self, df: pd.DataFrame) -> Path:
        p = self._path("threshold_table.csv")
        df.to_csv(p, index=False)
        return p

    def write_performance_summary(self, metrics: dict) -> Path:
        p = self._path("performance_summary.csv")
        pd.DataFrame([metrics]).to_csv(p, index=False)
        return p

    def write_feature_importance(self, importances: pd.Series | pd.DataFrame) -> Path:
        p = self._path("feature_importance.csv")
        if isinstance(importances, pd.Series):
            importances.to_csv(p, header=["importance"])
        else:
            importances.to_csv(p, index=False)
        return p

    def write_model_audit(self, audit: dict) -> Path:
        p = self._path("model_audit.csv")
        pd.DataFrame([audit]).to_csv(p, index=False)
        return p

    def write_alerts(self, alerts: pd.DataFrame) -> Path:
        p = self._path("alerts.csv")
        alerts.to_csv(p, index=False)
        return p
