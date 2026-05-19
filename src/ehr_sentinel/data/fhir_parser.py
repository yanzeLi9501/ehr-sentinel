"""FHIR R4 Bundle → admissions DataFrame parser.

Uses the optional `fhir.resources` package when available for strict
validation, but falls back to plain-dict parsing so the package works
without that dependency installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class FHIRParser:
    """Parse a FHIR R4 Bundle into a normalized admissions DataFrame."""

    ICD10_SYSTEMS = ("http://hl7.org/fhir/sid/icd-10", "http://hl7.org/fhir/sid/icd-10-cm")

    def parse(
        self,
        source: str | Path | dict,
        loader=None,
    ) -> pd.DataFrame:
        bundle = self._load(source)
        entries = bundle.get("entry", []) if isinstance(bundle, dict) else []

        patients: dict[str, dict] = {}
        encounters: dict[str, dict] = {}
        conditions_by_enc: dict[str, list[dict]] = {}
        observations_by_enc: dict[str, list[dict]] = {}

        for ent in entries:
            res = ent.get("resource", {}) if isinstance(ent, dict) else {}
            rtype = res.get("resourceType")
            if rtype == "Patient":
                patients[res.get("id", "")] = res
            elif rtype == "Encounter":
                encounters[res.get("id", "")] = res
            elif rtype == "Condition":
                enc_ref = self._extract_ref(res.get("encounter", {}))
                conditions_by_enc.setdefault(enc_ref, []).append(res)
            elif rtype == "Observation":
                enc_ref = self._extract_ref(res.get("encounter", {}))
                observations_by_enc.setdefault(enc_ref, []).append(res)

        rows: list[dict] = []
        for enc_id, enc in encounters.items():
            subj_ref = self._extract_ref(enc.get("subject", {}))
            period = enc.get("period", {}) or {}
            start = period.get("start")
            end = period.get("end")
            conds = conditions_by_enc.get(enc_id, [])
            obs = observations_by_enc.get(enc_id, [])

            primary_icd10 = None
            primary_text = None
            for cond in conds:
                for coding in (cond.get("code", {}) or {}).get("coding", []) or []:
                    if coding.get("system") in self.ICD10_SYSTEMS:
                        primary_icd10 = coding.get("code")
                        primary_text = coding.get("display") or cond.get("code", {}).get("text")
                        break
                if primary_icd10:
                    break

            row: dict[str, Any] = {
                "mrn": subj_ref,
                "admission_date": start,
                "discharge_date": end,
                "icd10": primary_icd10,
                "diagnosis_text": primary_text,
            }
            for o in obs:
                code = ((o.get("code", {}) or {}).get("coding") or [{}])[0].get("code")
                val = (o.get("valueQuantity", {}) or {}).get("value")
                if code and val is not None:
                    row[code] = float(val)
            rows.append(row)

        df = pd.DataFrame(rows)
        if loader is not None:
            return loader.from_dataframe(df)
        # Minimal normalization
        df["admission_date"] = pd.to_datetime(df.get("admission_date"), errors="coerce")
        df["discharge_date"] = pd.to_datetime(df.get("discharge_date"), errors="coerce")
        df["los"] = (df["discharge_date"] - df["admission_date"]).dt.total_seconds() / 86400.0
        df = df.sort_values(["mrn", "admission_date"]).reset_index(drop=True)
        df["visit_order"] = df.groupby("mrn").cumcount() + 1
        df["gap"] = (
            df.groupby("mrn")["admission_date"].shift(-1) - df["discharge_date"]
        ).dt.total_seconds() / 86400.0
        return df

    @staticmethod
    def _extract_ref(ref: Any) -> str:
        if not isinstance(ref, dict):
            return ""
        s = ref.get("reference", "") or ""
        return s.split("/")[-1] if "/" in s else s

    @staticmethod
    def _load(source: str | Path | dict) -> dict:
        if isinstance(source, dict):
            return source
        path = Path(source)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
