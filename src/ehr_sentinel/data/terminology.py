"""Terminology mapping: ICD-10 ↔ comorbidity group, LOINC ↔ lab name,
SNOMED CT ↔ ICD-10 crosswalk.

All mappings are configurable. Defaults are conventional but never
disease-specific in the sense of forcing a particular epidemic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from ehr_sentinel.utils.config import DEFAULT_COMORBIDITY_GROUPS


# Minimal LOINC→internal lab name table; users can extend via JSON config.
DEFAULT_LOINC_MAP: dict[str, str] = {
    "6690-2":  "WBC",
    "1988-5":  "CRP",
    "718-7":   "HGB",
    "1751-7":  "ALB",
    "2160-0":  "CREA",
    "2345-7":  "GLU",
    "6298-4":  "K",
    "2951-2":  "Na",
}


class TerminologyMapper:
    """Configurable terminology mapper."""

    def __init__(
        self,
        comorbidity_groups: Optional[dict[str, str]] = None,
        loinc_map: Optional[dict[str, str]] = None,
        snomed_to_icd10: Optional[dict[str, str]] = None,
    ) -> None:
        self.comorbidity_groups = dict(comorbidity_groups or DEFAULT_COMORBIDITY_GROUPS)
        self.loinc_map = dict(loinc_map or DEFAULT_LOINC_MAP)
        self.snomed_to_icd10 = dict(snomed_to_icd10 or {})
        self._compiled = {g: re.compile(p) for g, p in self.comorbidity_groups.items()}

    # ── ICD-10 → comorbidity group ─────────────────────────────────────
    def icd10_to_group(self, code: str) -> Optional[str]:
        if not isinstance(code, str) or not code:
            return None
        code = code.strip().upper().replace(" ", "")
        for group, pat in self._compiled.items():
            if pat.match(code):
                return group
        return None

    def assign_groups(self, df: pd.DataFrame, icd10_col: str = "icd10") -> pd.DataFrame:
        df = df.copy()
        df["comorbidity_group"] = df[icd10_col].apply(self.icd10_to_group)
        return df

    # ── LOINC → lab name ────────────────────────────────────────────────
    def loinc_to_lab(self, code: str) -> Optional[str]:
        return self.loinc_map.get(str(code).strip())

    # ── SNOMED → ICD-10 ─────────────────────────────────────────────────
    def snomed_to_icd(self, sct_code: str) -> Optional[str]:
        return self.snomed_to_icd10.get(str(sct_code).strip())

    # ── Persistence ─────────────────────────────────────────────────────
    @classmethod
    def from_json(cls, path: str | Path) -> "TerminologyMapper":
        with Path(path).open("r", encoding="utf-8") as f:
            blob = json.load(f)
        return cls(
            comorbidity_groups=blob.get("comorbidity_groups"),
            loinc_map=blob.get("loinc_map"),
            snomed_to_icd10=blob.get("snomed_to_icd10"),
        )

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump({
                "comorbidity_groups": self.comorbidity_groups,
                "loinc_map": self.loinc_map,
                "snomed_to_icd10": self.snomed_to_icd10,
            }, f, indent=2)
