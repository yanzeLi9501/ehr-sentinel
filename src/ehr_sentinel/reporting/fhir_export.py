"""FHIR R4 MeasureReport export for surveillance results."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


class FHIRExporter:
    """Emit a FHIR R4 MeasureReport summarizing surveillance output."""

    def __init__(self, target_disease: str, measure_id: str = "ehr-sentinel-surveillance") -> None:
        self.target_disease = target_disease
        self.measure_id = measure_id

    def to_measure_report(
        self,
        lgdi_timeline: pd.DataFrame,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        alerts: Optional[pd.DataFrame] = None,
    ) -> dict:
        if not lgdi_timeline.empty:
            weeks = pd.to_datetime(lgdi_timeline["week"])
            ps = period_start or weeks.min().strftime("%Y-%m-%d")
            pe = period_end or weeks.max().strftime("%Y-%m-%d")
        else:
            ps = period_start or "1970-01-01"
            pe = period_end or "1970-01-01"

        groups = []
        if not lgdi_timeline.empty:
            for _, row in lgdi_timeline.iterrows():
                groups.append({
                    "code": {"text": "LGDI"},
                    "population": [{
                        "code": {"text": "week"},
                        "count": int(pd.Timestamp(row["week"]).timestamp()),
                    }],
                    "measureScore": {"value": float(row["lgdi"]), "unit": "ratio"},
                })

        n_alerts = 0
        if alerts is not None and not alerts.empty and "alert" in alerts.columns:
            n_alerts = int(alerts["alert"].astype(int).sum())

        return {
            "resourceType": "MeasureReport",
            "id": self.measure_id,
            "status": "complete",
            "type": "summary",
            "measure": f"urn:ehr-sentinel:{self.target_disease.lower().replace(' ', '-')}",
            "date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "period": {"start": ps, "end": pe},
            "subject": {"display": f"Population surveillance for {self.target_disease}"},
            "group": groups,
            "extension": [{
                "url": "urn:ehr-sentinel:n_alerts",
                "valueInteger": n_alerts,
            }],
        }

    def write(self, report: dict, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return path
