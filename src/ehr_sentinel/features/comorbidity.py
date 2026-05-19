"""Assign each admission to one or more comorbidity groups via ICD-10 regex."""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from ehr_sentinel.utils.config import DEFAULT_COMORBIDITY_GROUPS


class ComorbidityGrouper:
    """Configurable ICD-10 → comorbidity group assigner.

    Multi-label: an admission can belong to several groups if its ICD-10 code
    matches multiple regexes. The first match is also stored as
    ``comorbidity_group`` for convenience.
    """

    def __init__(self, groups: Optional[dict[str, str]] = None) -> None:
        self.groups = dict(groups or DEFAULT_COMORBIDITY_GROUPS)
        self._compiled = [(name, re.compile(pat)) for name, pat in self.groups.items()]

    @property
    def group_names(self) -> list[str]:
        return list(self.groups.keys())

    def assign(self, codes: pd.Series) -> pd.DataFrame:
        """Return a DataFrame with one column per group + 'comorbidity_group'."""
        codes = codes.fillna("").astype(str).str.upper().str.replace(" ", "", regex=False)
        result = pd.DataFrame(index=codes.index)
        primary: list[Optional[str]] = []
        for name, pat in self._compiled:
            result[f"group_{name}"] = codes.str.match(pat).astype("int8")
        for code in codes:
            assigned = None
            for name, pat in self._compiled:
                if pat.match(code):
                    assigned = name
                    break
            primary.append(assigned)
        result["comorbidity_group"] = primary
        return result

    def add_to(self, df: pd.DataFrame, icd10_col: str = "icd10") -> pd.DataFrame:
        out = self.assign(df[icd10_col])
        df = df.drop(columns=[c for c in out.columns if c in df.columns], errors="ignore")
        return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
