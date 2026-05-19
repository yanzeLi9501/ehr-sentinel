"""High-level feature matrix builder.

Combines comorbidity assignment, temporal features, lab values, and
prior-admission statistics into a model-ready matrix.

Set ``EpidemicConfig.enhanced_features=False`` to skip the EMA/rolling
prior-stats expansion and produce the smaller base feature set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ehr_sentinel.features.comorbidity import ComorbidityGrouper
from ehr_sentinel.features.temporal import (
    compute_los_gap,
    compute_seasonal_features,
    compute_visit_order,
    compute_prior_stats,
)
from ehr_sentinel.utils.config import EpidemicConfig


@dataclass
class FeatureMatrix:
    X: pd.DataFrame
    y_gap: Optional[pd.Series]
    y_los: Optional[pd.Series]
    groups: pd.Series              # patient MRN — for GroupKFold
    meta: pd.DataFrame             # mrn, admission_date, comorbidity_group


class FeatureBuilder:
    """Build a model-ready feature matrix from a raw admissions DataFrame."""

    def __init__(self, config: EpidemicConfig) -> None:
        self.config = config
        self.grouper = ComorbidityGrouper(config.comorbidity_groups)

    # ── Public API ──────────────────────────────────────────────────────
    def build(self, df: pd.DataFrame) -> FeatureMatrix:
        cfg = self.config
        df = df.copy()

        # 1. Assign comorbidity groups
        df = self.grouper.add_to(df, icd10_col="icd10")

        # 2. Visit order + temporal features
        df = compute_visit_order(df)
        df = compute_seasonal_features(df)

        # 3. Apply min_visit_order filter (mark rather than drop)
        df["vo_pass"] = (df["visit_order"] >= cfg.min_visit_order).astype("int8")

        # 4. LOS / Gap caps and log transforms
        df = compute_los_gap(df, los_cap=cfg.los_cap_days, gap_cap=cfg.gap_cap_days)

        # 5. Prior-admission statistics (optional, enhanced set)
        lab_cols = [c for c in cfg.lab_panel if c in df.columns]
        prior_cols = ["los", "gap"] + lab_cols
        if cfg.enhanced_features:
            df = compute_prior_stats(df, columns=prior_cols, windows=(3, 5), ema_spans=(3, 5))
        else:
            df = compute_prior_stats(df, columns=prior_cols, windows=(3,), ema_spans=())

        # 6. Assemble X
        feature_cols = self._select_feature_cols(df, lab_cols)
        X = df[feature_cols].copy()
        # Mean imputation column-wise
        X = X.apply(lambda s: s.fillna(s.mean()) if s.dtype.kind in "fc" else s.fillna(0))
        X = X.fillna(0.0)

        y_gap = df["gap_capped"] if "gap_capped" in df.columns else None
        y_los = df["los_capped"] if "los_capped" in df.columns else None
        groups = df["mrn"].astype(str)
        meta_cols = [c for c in ("mrn", "admission_date", "comorbidity_group", "visit_order", "vo_pass")
                     if c in df.columns]
        return FeatureMatrix(X=X, y_gap=y_gap, y_los=y_los, groups=groups, meta=df[meta_cols].copy())

    # ── Internal ────────────────────────────────────────────────────────
    def _select_feature_cols(self, df: pd.DataFrame, lab_cols: Iterable[str]) -> list[str]:
        cols: list[str] = []
        # Group one-hot
        cols += [c for c in df.columns if c.startswith("group_")]
        # Temporal / seasonal
        cols += [
            "visit_order", "n_total_visits", "visit_frac",
            "month", "dow", "year", "weekofyear",
            "month_sin", "month_cos", "dow_sin", "dow_cos", "woy_sin", "woy_cos",
        ]
        # Labs (current admission)
        cols += list(lab_cols)
        # Prior stats (whatever was generated)
        cols += [c for c in df.columns if any(
            c.endswith(f"_{suffix}") or f"_{suffix}" in c
            for suffix in ("prev", "rmean3", "rmean5", "rstd3", "rstd5", "ema3", "ema5")
        )]
        # Dedup, keep only present columns
        seen: set[str] = set()
        out: list[str] = []
        for c in cols:
            if c in df.columns and c not in seen and pd.api.types.is_numeric_dtype(df[c]):
                seen.add(c)
                out.append(c)
        return out
