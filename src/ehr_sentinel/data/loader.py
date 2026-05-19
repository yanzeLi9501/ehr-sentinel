"""Multi-source EHR loader with column auto-detection."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from ehr_sentinel.utils.validation import ensure_columns, ensure_datetime


# Column-name synonyms used by auto_configure
_SYNONYMS: dict[str, list[str]] = {
    "mrn":             ["mrn", "patient_id", "subject_id", "patientid", "pid", "病案号"],
    "admission_date":  ["admission_date", "admit_date", "admit_dt", "admitdate", "admittime", "intime", "入院日期", "入院时间"],
    "discharge_date":  ["discharge_date", "disch_date", "dischtime", "outtime", "出院日期", "出院时间"],
    "los":             ["los", "length_of_stay", "los_days", "住院天数", "实际住院天数", "时间差"],
    "gap":             ["gap", "gap_days", "interval", "interval_days"],
    "icd10":           ["icd10", "icd_10", "icd_code", "diagnosis_code", "icd9code", "dia_ppal", "诊断编码"],
    "diagnosis_text":  ["diagnosis_text", "diagnosis", "dx_text", "diag_inpat", "diagnosisstring", "诊断", "主要诊断", "诊断文本", "emr_初步诊断"],
    "visit_order":     ["visit_order", "vo", "visit_number", "vn", "admission_seq", "住院次数"],
}

_LAB_SYNONYMS: dict[str, list[str]] = {
    "WBC":  ["wbc", "lab_wbc", "white_blood_cell", "white_blood_cells", "白细胞"],
    "CRP":  ["crp", "lab_crp", "c_reactive_protein", "超敏c反应蛋白"],
    "HGB":  ["hgb", "lab_hgb", "hemoglobin", "hb", "血红蛋白"],
    "ALB":  ["alb", "lab_alb", "albumin", "白蛋白"],
    "CREA": ["crea", "lab_crea", "creatinine", "scr", "肌酐"],
    "GLU":  ["glu", "lab_glu", "glucose", "blood_glucose", "空腹血糖"],
    "K":    ["k", "lab_k", "potassium", "钾"],
    "Na":   ["na", "lab_na", "sodium", "钠"],
}


@dataclass
class DataSourceProfile:
    """Describes the columns and labs detected in a source DataFrame."""

    column_map: dict[str, str] = field(default_factory=dict)
    detected_labs: dict[str, str] = field(default_factory=dict)
    n_rows: int = 0
    n_patients: int = 0
    date_range: tuple[Optional[str], Optional[str]] = (None, None)
    extra_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "column_map": dict(self.column_map),
            "detected_labs": dict(self.detected_labs),
            "n_rows": int(self.n_rows),
            "n_patients": int(self.n_patients),
            "date_range": list(self.date_range),
            "extra_columns": list(self.extra_columns),
        }


def _norm(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]", "_", str(name).strip().lower())


def _find_match(columns_norm: dict[str, str], candidates: list[str]) -> Optional[str]:
    """Return the original column name matching any of the candidate synonyms."""
    cand_norm = [_norm(c) for c in candidates]
    for orig, n in columns_norm.items():
        if n in cand_norm:
            return orig
    # substring match fallback
    for orig, n in columns_norm.items():
        for c in cand_norm:
            if c in n:
                return orig
    return None


class EHRLoader:
    """Load EHR admissions data from CSV, FHIR Bundle, or in-memory DataFrame."""

    def __init__(self) -> None:
        self.profile: Optional[DataSourceProfile] = None

    # ── Public entry points ─────────────────────────────────────────────
    def from_csv(self, path: str | Path, encoding: Optional[str] = None) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if encoding is None:
            # Try utf-8, then gbk (common for Chinese hospital exports)
            for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, f"could not decode {path}")
        else:
            df = pd.read_csv(path, encoding=encoding)
        return self._normalize(df)

    def from_dataframe(
        self, df: pd.DataFrame, column_map: Optional[dict[str, str]] = None
    ) -> pd.DataFrame:
        df = df.copy()
        if column_map:
            df = df.rename(columns={v: k for k, v in column_map.items()})
        return self._normalize(df)

    def from_fhir_bundle(self, source: str | Path | dict) -> pd.DataFrame:
        """Parse a FHIR R4 Bundle and return an admissions DataFrame."""
        from ehr_sentinel.data.fhir_parser import FHIRParser
        return FHIRParser().parse(source, loader=self)

    # ── Column auto-detection ──────────────────────────────────────────
    def auto_configure(self, df: pd.DataFrame) -> DataSourceProfile:
        cols_norm = {c: _norm(c) for c in df.columns}
        column_map: dict[str, str] = {}
        for canonical, synonyms in _SYNONYMS.items():
            match = _find_match(cols_norm, synonyms)
            if match is not None:
                column_map[canonical] = match

        detected_labs: dict[str, str] = {}
        for lab, syns in _LAB_SYNONYMS.items():
            m = _find_match(cols_norm, syns)
            if m is not None:
                detected_labs[lab] = m

        date_range = (None, None)
        if "admission_date" in column_map:
            try:
                ad = pd.to_datetime(df[column_map["admission_date"]], errors="coerce")
                date_range = (
                    str(ad.min().date()) if pd.notna(ad.min()) else None,
                    str(ad.max().date()) if pd.notna(ad.max()) else None,
                )
            except Exception:
                pass

        n_patients = 0
        if "mrn" in column_map:
            n_patients = int(df[column_map["mrn"]].nunique())

        extra = [c for c in df.columns
                 if c not in column_map.values() and c not in detected_labs.values()]
        profile = DataSourceProfile(
            column_map=column_map,
            detected_labs=detected_labs,
            n_rows=len(df),
            n_patients=n_patients,
            date_range=date_range,
            extra_columns=extra,
        )
        return profile

    # ── Internal ────────────────────────────────────────────────────────
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        profile = self.auto_configure(df)
        self.profile = profile
        # Rename detected columns to canonical names
        rename = {v: k for k, v in profile.column_map.items() if k != v}
        rename.update({v: k for k, v in profile.detected_labs.items() if k != v})
        df = df.rename(columns=rename)
        # Datetime coercion
        df = ensure_datetime(df, ["admission_date", "discharge_date"])
        try:
            ensure_columns(df)
        except Exception:
            # Auto-detection failed for required columns: surface a clear error
            raise
        # Recompute LOS / gap when missing
        if "los" not in df.columns and "discharge_date" in df.columns:
            df["los"] = (df["discharge_date"] - df["admission_date"]).dt.total_seconds() / 86400.0
        if "gap" not in df.columns:
            df = df.sort_values(["mrn", "admission_date"])
            df["gap"] = (
                df.groupby("mrn")["admission_date"].shift(-1)
                - df.get("discharge_date", df["admission_date"])
            ).dt.total_seconds() / 86400.0
        if "visit_order" not in df.columns:
            df = df.sort_values(["mrn", "admission_date"])
            df["visit_order"] = df.groupby("mrn").cumcount() + 1
        return df.reset_index(drop=True)
