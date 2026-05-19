"""Temporal feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_los_gap(
    df: pd.DataFrame,
    los_cap: float = 60.0,
    gap_cap: float = 30.0,
    los_col: str = "los",
    gap_col: str = "gap",
) -> pd.DataFrame:
    """Add capped LOS and capped Gap columns + log-transforms."""
    df = df.copy()
    if los_col in df.columns:
        df["los_capped"] = df[los_col].clip(lower=0.0, upper=los_cap)
        df["log_los"] = np.log1p(df["los_capped"])
    if gap_col in df.columns:
        df["gap_capped"] = df[gap_col].clip(lower=0.0, upper=gap_cap)
        df["log_gap"] = np.log1p(df["gap_capped"])
    return df


def compute_visit_order(df: pd.DataFrame, mrn_col: str = "mrn", date_col: str = "admission_date") -> pd.DataFrame:
    df = df.sort_values([mrn_col, date_col]).copy()
    df["visit_order"] = df.groupby(mrn_col).cumcount() + 1
    df["n_total_visits"] = df.groupby(mrn_col)[mrn_col].transform("count")
    df["visit_frac"] = df["visit_order"] / df["n_total_visits"]
    return df.reset_index(drop=True)


def compute_seasonal_features(df: pd.DataFrame, date_col: str = "admission_date") -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[date_col], errors="coerce")
    df["month"] = d.dt.month
    df["dow"] = d.dt.dayofweek
    df["year"] = d.dt.year
    df["weekofyear"] = d.dt.isocalendar().week.astype("Int32").fillna(0).astype("int32")
    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["woy_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52.0)
    return df


def compute_prior_stats(
    df: pd.DataFrame,
    columns: list[str],
    mrn_col: str = "mrn",
    date_col: str = "admission_date",
    windows: tuple[int, ...] = (3, 5),
    ema_spans: tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    """Per-patient rolling mean/std/last + EMA over previous admissions.

    Uses ``shift(1)`` so the current admission's values are excluded —
    prevents leakage when these features feed a model predicting that
    admission.
    """
    df = df.sort_values([mrn_col, date_col]).copy()
    for col in columns:
        if col not in df.columns:
            continue
        g = df.groupby(mrn_col)[col]
        prev = g.shift(1)
        df[f"{col}_prev"] = prev
        for w in windows:
            df[f"{col}_rmean{w}"] = prev.groupby(df[mrn_col]).transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )
            df[f"{col}_rstd{w}"] = prev.groupby(df[mrn_col]).transform(
                lambda s, w=w: s.rolling(w, min_periods=2).std()
            )
        for span in ema_spans:
            df[f"{col}_ema{span}"] = prev.groupby(df[mrn_col]).transform(
                lambda s, span=span: s.ewm(span=span, min_periods=1).mean()
            )
    return df.reset_index(drop=True)
