"""Input schema validation helpers used by the loader and pipeline."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ("mrn", "admission_date")
OPTIONAL_COLUMNS = ("discharge_date", "los", "gap", "diagnosis_text", "icd10")


class SchemaError(ValueError):
    """Raised when an input DataFrame does not satisfy the minimal schema."""


def ensure_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")


def ensure_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_non_empty(df: pd.DataFrame, name: str = "input") -> None:
    if df is None or len(df) == 0:
        raise SchemaError(f"{name} dataframe is empty")
