"""Synthetic EHR data generators.

These are the **only** data generators shipped with the package. They produce
deterministic, fully synthetic admissions with realistic distributions —
they contain no patient information, no copies of any real dataset, and no
identifiers tied to real systems.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# A small ICD-10 pool mapped to the default `EpidemicConfig` comorbidity groups.
_ICD10_POOL: dict[str, list[tuple[str, str]]] = {
    "Cardiovascular": [
        ("I10", "Essential (primary) hypertension"),
        ("I25.10", "Atherosclerotic heart disease"),
        ("I50.9", "Heart failure, unspecified"),
        ("I48.91", "Atrial fibrillation, unspecified"),
    ],
    "Diabetes": [
        ("E11.9", "Type 2 diabetes mellitus without complications"),
        ("E11.65", "Type 2 diabetes with hyperglycemia"),
    ],
    "Respiratory": [
        ("J18.9", "Pneumonia, unspecified organism"),
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified"),
        ("J45.909", "Unspecified asthma, uncomplicated"),
        ("J11.1", "Influenza with respiratory manifestations"),
        ("U07.1", "COVID-19, virus identified"),
    ],
    "Renal": [
        ("N17.9", "Acute kidney failure, unspecified"),
        ("N18.3", "Chronic kidney disease, stage 3"),
    ],
    "Cancer": [
        ("C50.911", "Malignant neoplasm of breast"),
        ("C34.90", "Malignant neoplasm of lung"),
    ],
    "Neurological": [
        ("G93.40", "Encephalopathy, unspecified"),
        ("F03.90", "Unspecified dementia"),
    ],
}

# Realistic lab reference distributions (mean, sd, lo, hi)
_LAB_DIST = {
    "WBC":  (7.5, 3.0, 0.5, 50.0),
    "CRP":  (10.0, 30.0, 0.1, 400.0),
    "HGB":  (130.0, 20.0, 50.0, 200.0),
    "ALB":  (38.0, 6.0, 15.0, 55.0),
    "CREA": (80.0, 40.0, 20.0, 800.0),
    "GLU":  (6.0, 2.5, 2.0, 30.0),
    "K":    (4.2, 0.6, 2.5, 7.0),
    "Na":   (140.0, 4.0, 120.0, 160.0),
}


def generate_admissions(
    n_patients: int = 500,
    n_admissions_per_patient: tuple[int, int] = (2, 30),
    start_date: str = "2016-01-01",
    end_date: str = "2022-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a fully synthetic admissions DataFrame.

    Returns columns:
      mrn, admission_date, discharge_date, los, gap, visit_order,
      icd10, diagnosis_text, comorbidity_group, plus the 8 lab tests.

    All values are sampled programmatically; the output contains no real
    patient data.
    """
    rng = np.random.default_rng(seed)
    t0 = datetime.strptime(start_date, "%Y-%m-%d")
    t1 = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (t1 - t0).days
    if total_days <= 0:
        raise ValueError("end_date must be after start_date")

    lo, hi = n_admissions_per_patient
    rows: list[dict] = []
    group_keys = list(_ICD10_POOL.keys())

    for i in range(n_patients):
        mrn = f"SYN-{i:06d}"
        # Poisson count between lo and hi
        lam = (lo + hi) / 2.0
        n_adm = int(np.clip(rng.poisson(lam), lo, hi))
        # Patient's preferred comorbidity group (most admissions land here)
        primary_group = rng.choice(group_keys)

        # First admission date uniform in window
        first_offset_days = int(rng.integers(0, max(1, total_days - 30)))
        current_date = t0 + timedelta(days=first_offset_days)

        for vo in range(1, n_adm + 1):
            # Seasonality: winter peak (months 11-2) gets a +25% admission-shift
            month = current_date.month
            winter_boost = 1.0 + (0.25 if month in (11, 12, 1, 2) else 0.0)
            # 70% chance of primary group, else random
            group = primary_group if rng.random() < 0.7 else rng.choice(group_keys)
            icd, desc = _ICD10_POOL[group][int(rng.integers(0, len(_ICD10_POOL[group])))]

            los = float(np.clip(rng.lognormal(mean=1.5, sigma=0.6), 0.1, 180.0))
            discharge = current_date + timedelta(days=los)

            row = {
                "mrn": mrn,
                "admission_date": current_date,
                "discharge_date": discharge,
                "los": los,
                "visit_order": vo,
                "icd10": icd,
                "diagnosis_text": desc,
                "comorbidity_group": group,
            }
            for lab, (mu, sd, lo_lab, hi_lab) in _LAB_DIST.items():
                row[lab] = float(np.clip(rng.normal(mu, sd), lo_lab, hi_lab))
            rows.append(row)

            # Gap until next admission
            gap = float(np.clip(rng.exponential(scale=60.0) / winter_boost, 1.0, 365 * 2))
            current_date = discharge + timedelta(days=gap)
            if current_date > t1:
                break

    df = pd.DataFrame(rows)
    df = df.sort_values(["mrn", "admission_date"]).reset_index(drop=True)
    # Gap = next admission − current discharge, per patient
    df["gap"] = (
        df.groupby("mrn")["admission_date"].shift(-1) - df["discharge_date"]
    ).dt.total_seconds() / 86400.0
    return df


def generate_epidemic_signal(
    df: pd.DataFrame,
    target_icd10: list[str],
    outbreak_start: str,
    outbreak_end: str,
    target_group: str = "Respiratory",
    effect_size: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Inject a synthetic epidemic signal into an admissions DataFrame.

    During ``[outbreak_start, outbreak_end]``:
      * the gap for ``target_group`` admissions shrinks (more frequent visits)
      * a random fraction of admissions in that window are re-coded to one of
        ``target_icd10`` codes
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    t0 = pd.Timestamp(outbreak_start)
    t1 = pd.Timestamp(outbreak_end)
    mask = (df["admission_date"] >= t0) & (df["admission_date"] <= t1)
    target_mask = mask & (df["comorbidity_group"] == target_group)

    # Shrink gap by (1 - effect_size)
    if "gap" in df.columns:
        df.loc[target_mask, "gap"] = df.loc[target_mask, "gap"] * max(0.05, 1.0 - effect_size)

    # Recode ~effect_size fraction of in-window admissions to target ICD-10
    candidates = df.index[mask]
    n_recode = int(len(candidates) * effect_size)
    if n_recode > 0 and target_icd10:
        chosen = rng.choice(candidates, size=n_recode, replace=False)
        codes = rng.choice(target_icd10, size=n_recode)
        df.loc[chosen, "icd10"] = codes
        df.loc[chosen, "comorbidity_group"] = target_group
    return df


def generate_fhir_bundle(
    n_patients: int = 50,
    seed: int = 42,
    start_date: str = "2018-01-01",
    end_date: str = "2022-12-31",
) -> dict:
    """Generate a synthetic FHIR R4 Bundle (as a plain Python dict)."""
    df = generate_admissions(
        n_patients=n_patients, start_date=start_date, end_date=end_date, seed=seed
    )
    entries: list[dict] = []
    for mrn in df["mrn"].unique():
        entries.append({
            "resource": {
                "resourceType": "Patient",
                "id": mrn,
                "identifier": [{"system": "urn:synthetic:mrn", "value": mrn}],
            }
        })

    for i, row in df.iterrows():
        enc_id = f"enc-{i:08d}"
        entries.append({
            "resource": {
                "resourceType": "Encounter",
                "id": enc_id,
                "status": "finished",
                "subject": {"reference": f"Patient/{row['mrn']}"},
                "period": {
                    "start": row["admission_date"].isoformat(),
                    "end": row["discharge_date"].isoformat(),
                },
            }
        })
        entries.append({
            "resource": {
                "resourceType": "Condition",
                "id": f"cond-{i:08d}",
                "subject": {"reference": f"Patient/{row['mrn']}"},
                "encounter": {"reference": f"Encounter/{enc_id}"},
                "code": {
                    "coding": [{
                        "system": "http://hl7.org/fhir/sid/icd-10",
                        "code": row["icd10"],
                        "display": row["diagnosis_text"],
                    }]
                },
                "recordedDate": row["admission_date"].isoformat(),
            }
        })
        for lab in _LAB_DIST:
            if lab in df.columns:
                entries.append({
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"obs-{i:08d}-{lab}",
                        "status": "final",
                        "subject": {"reference": f"Patient/{row['mrn']}"},
                        "encounter": {"reference": f"Encounter/{enc_id}"},
                        "code": {"coding": [{"system": "urn:synthetic:lab", "code": lab}]},
                        "valueQuantity": {"value": float(row[lab]), "unit": "unit"},
                        "effectiveDateTime": row["admission_date"].isoformat(),
                    }
                })

    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def write_fhir_bundle(bundle: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)
