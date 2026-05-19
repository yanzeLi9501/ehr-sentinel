"""Run isolated multi-dataset validation without copying source data.

The script reads public PhysioNet/FluNet tables and optional local WHU cohorts,
normalizes them to the ehr-sentinel admissions schema, runs the surveillance
pipeline, and writes aggregate-only JSON/Markdown summaries. No patient-level
rows are written to disk.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import pandas as pd

from ehr_sentinel import EHRLoader, EpidemicConfig, run_surveillance_pipeline


MIMIC_IV_ROOT: Path | None = None
LAB_COLUMNS = ["WBC", "CRP", "HGB", "ALB", "CREA", "GLU", "K", "Na"]
WHU_LAB_MAP = {
    "白细胞": "WBC",
    "超敏C反应蛋白": "CRP",
    "血红蛋白": "HGB",
    "白蛋白": "ALB",
    "肌酐": "CREA",
    "空腹血糖": "GLU",
    "钾": "K",
    "钠": "Na",
    "lab_WBC": "WBC",
    "lab_CRP": "CRP",
    "lab_HGB": "HGB",
    "lab_ALB": "ALB",
    "lab_CREA": "CREA",
    "lab_GLU": "GLU",
    "lab_K": "K",
    "lab_Na": "Na",
}


def _lab_panel_from_label(label: object, fluid: object | None = None) -> str | None:
    text = "" if pd.isna(label) else str(label).strip().lower()
    fluid_text = "" if fluid is None or pd.isna(fluid) else str(fluid).strip().lower()
    if fluid_text and fluid_text != "blood":
        return None
    if "white blood cells" in text or text.startswith("wbc --"):
        return "WBC"
    if "c-reactive protein" in text or "c reactive protein" in text:
        return "CRP"
    if (text == "hemoglobin" or text.startswith("hgb -- hemoglobin")) and "glycated" not in text:
        return "HGB"
    if text == "albumin" or text.startswith("alb -- albumin"):
        return "ALB"
    if text == "creatinine" or text.startswith("crea -- creatinine"):
        return "CREA"
    if text == "glucose" or text.startswith("glu -- glucose") or text.startswith("gluors -- glucose"):
        return "GLU"
    if text == "potassium" or text.startswith("k -- potassium"):
        return "K"
    if text == "sodium" or text.startswith("na -- sodium"):
        return "Na"
    return None


def _lab_panel_from_eicu_name(label: object) -> str | None:
    text = "" if pd.isna(label) else str(label).strip().lower()
    if text == "wbc x 1000":
        return "WBC"
    if text in {"crp", "crp-hs"}:
        return "CRP"
    if text in {"hgb", "hemoglobin"}:
        return "HGB"
    if text == "albumin":
        return "ALB"
    if text == "creatinine":
        return "CREA"
    if text in {"glucose", "bedside glucose"}:
        return "GLU"
    if text == "potassium":
        return "K"
    if text == "sodium":
        return "Na"
    return None


def _aggregate_labevents(lab_path: Path, item_path: Path, key: str) -> pd.DataFrame:
    items = pd.read_csv(item_path, usecols=["itemid", "label", "fluid"])
    items["panel"] = [_lab_panel_from_label(label, fluid) for label, fluid in zip(items["label"], items["fluid"])]
    item_map = items.dropna(subset=["panel"]).set_index("itemid")["panel"].to_dict()
    if not item_map:
        return pd.DataFrame(columns=[key, *LAB_COLUMNS])

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        lab_path,
        usecols=[key, "itemid", "valuenum"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        chunk = chunk[chunk["itemid"].isin(item_map) & chunk[key].notna() & chunk["valuenum"].notna()].copy()
        if chunk.empty:
            continue
        chunk["panel"] = chunk["itemid"].map(item_map)
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["valuenum"])
        if not chunk.empty:
            parts.append(chunk.groupby([key, "panel"])["valuenum"].agg(["sum", "count"]).reset_index())
    if not parts:
        return pd.DataFrame(columns=[key, *LAB_COLUMNS])

    agg = pd.concat(parts, ignore_index=True).groupby([key, "panel"], as_index=False)[["sum", "count"]].sum()
    agg["mean"] = agg["sum"] / agg["count"].replace({0: pd.NA})
    return agg.pivot(index=key, columns="panel", values="mean").reset_index()


def _aggregate_cdsl_labs(path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["patient_id", "item_lab", "val_result"],
        chunksize=500_000,
        low_memory=False,
    ):
        chunk["panel"] = chunk["item_lab"].map(_lab_panel_from_label)
        chunk = chunk.dropna(subset=["patient_id", "panel"]).copy()
        if chunk.empty:
            continue
        chunk["val_result"] = pd.to_numeric(chunk["val_result"], errors="coerce")
        chunk = chunk.dropna(subset=["val_result"])
        if not chunk.empty:
            parts.append(chunk.groupby(["patient_id", "panel"])["val_result"].agg(["sum", "count"]).reset_index())
    if not parts:
        return pd.DataFrame(columns=["patient_id", *LAB_COLUMNS])

    agg = pd.concat(parts, ignore_index=True).groupby(["patient_id", "panel"], as_index=False)[["sum", "count"]].sum()
    agg["mean"] = agg["sum"] / agg["count"].replace({0: pd.NA})
    return agg.pivot(index="patient_id", columns="panel", values="mean").reset_index()


def _aggregate_eicu_labs(path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["patientunitstayid", "labname", "labresult"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        chunk["panel"] = chunk["labname"].map(_lab_panel_from_eicu_name)
        chunk = chunk.dropna(subset=["patientunitstayid", "panel"]).copy()
        if chunk.empty:
            continue
        chunk["labresult"] = pd.to_numeric(chunk["labresult"], errors="coerce")
        chunk = chunk.dropna(subset=["labresult"])
        if not chunk.empty:
            parts.append(chunk.groupby(["patientunitstayid", "panel"])["labresult"].agg(["sum", "count"]).reset_index())
    if not parts:
        return pd.DataFrame(columns=["patientunitstayid", *LAB_COLUMNS])

    agg = pd.concat(parts, ignore_index=True).groupby(["patientunitstayid", "panel"], as_index=False)[["sum", "count"]].sum()
    agg["mean"] = agg["sum"] / agg["count"].replace({0: pd.NA})
    return agg.pivot(index="patientunitstayid", columns="panel", values="mean").reset_index()

TEXT_TO_SURROGATE_ICD = [
    (re.compile(r"肺炎|肺部感染|呼吸衰竭|慢阻肺|COPD|哮喘|肺纤维化|肺栓塞", re.I), "J18"),
    (re.compile(r"糖尿病|血糖", re.I), "E11"),
    (re.compile(r"肾功能|肾病|肾衰|透析|肾脏", re.I), "N18"),
    (re.compile(r"肿瘤|癌|恶性", re.I), "C80"),
    (re.compile(r"脑梗|脑出血|脑血管|脑卒中|中风|神经", re.I), "G99"),
    (re.compile(r"冠状动脉|冠心病|心绞痛|心肌梗|心力衰竭|心功能|心脏病|心律失常|房颤|瓣膜|动脉粥样硬化|支架植入|心肌病|高血压", re.I), "I25"),
    (re.compile(r"新型冠状病毒|冠状病毒感染|冠状病毒肺炎|COVID", re.I), "U07.1"),
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    loader: Callable[[Path, Path], pd.DataFrame | dict]
    target_group: str = "Respiratory"


def _first_diagnosis(path: Path, key: str, code_col: str = "icd_code") -> pd.DataFrame:
    dx = pd.read_csv(path, usecols=[key, "seq_num", code_col], low_memory=False)
    dx = dx.sort_values([key, "seq_num"]).drop_duplicates(key, keep="first")
    return dx[[key, code_col]].rename(columns={code_col: "icd10"})


def _mimic_hosp(root: Path, _nc_root: Path) -> pd.DataFrame:
    base = MIMIC_IV_ROOT / "hosp" if MIMIC_IV_ROOT and (MIMIC_IV_ROOT / "hosp").exists() else root / "physionet" / "mimiciv" / "3.1" / "hosp"
    adm = pd.read_csv(base / "admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime", "dischtime"], low_memory=False)
    dx = _first_diagnosis(base / "diagnoses_icd.csv.gz", "hadm_id")
    out = adm.merge(dx, on="hadm_id", how="left")
    lab_path = base / "labevents.csv.gz"
    if lab_path.exists():
        labs = _aggregate_labevents(lab_path, base / "d_labitems.csv.gz", "hadm_id")
        out = out.merge(labs, on="hadm_id", how="left")
        out.attrs["lab_source_status"] = f"joined from {base.name}/labevents.csv.gz"
    elif (base / "labevents.csv.gz.part").exists():
        out.attrs["lab_source_status"] = "not joined: labevents.csv.gz is incomplete (.part file only)"
    else:
        out.attrs["lab_source_status"] = "not joined: labevents.csv.gz not found"
    return out.rename(
        columns={"subject_id": "mrn", "admittime": "admission_date", "dischtime": "discharge_date"}
    )


def _mimic_ed(root: Path, _nc_root: Path) -> pd.DataFrame:
    base = root / "physionet" / "mimic-iv-ed" / "2.2" / "ed"
    stays = pd.read_csv(base / "edstays.csv.gz", usecols=["subject_id", "stay_id", "intime", "outtime"], low_memory=False)
    dx = _first_diagnosis(base / "diagnosis.csv.gz", "stay_id")
    out = stays.merge(dx, on="stay_id", how="left")
    out.attrs["lab_source_status"] = "not joined: MIMIC-IV-ED release folder has vitalsign but no lab-result table"
    return out.rename(
        columns={"subject_id": "mrn", "intime": "admission_date", "outtime": "discharge_date"}
    )


def _nwicu_hosp(root: Path, _nc_root: Path) -> pd.DataFrame:
    base = root / "physionet" / "nwicu-northwestern-icu" / "0.1.0" / "data" / "nw_hosp"
    adm = pd.read_csv(base / "admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime", "dischtime"], low_memory=False)
    dx = _first_diagnosis(base / "diagnoses_icd.csv.gz", "hadm_id")
    labs = _aggregate_labevents(base / "labevents.csv.gz", base / "d_labitems.csv.gz", "hadm_id")
    out = adm.merge(dx, on="hadm_id", how="left").merge(labs, on="hadm_id", how="left")
    out.attrs["lab_source_status"] = "joined from nw_hosp/labevents.csv.gz"
    return out.rename(
        columns={"subject_id": "mrn", "admittime": "admission_date", "dischtime": "discharge_date"}
    )


def _eicu_crd(root: Path, _nc_root: Path) -> pd.DataFrame:
    base = root / "physionet" / "eicu-crd" / "2.0"
    patient = pd.read_csv(
        base / "patient.csv.gz",
        usecols=[
            "patientunitstayid",
            "uniquepid",
            "hospitaldischargeyear",
            "hospitaladmitoffset",
            "hospitaldischargeoffset",
        ],
        low_memory=False,
    )
    patient["admission_date"] = (
        pd.to_datetime(patient["hospitaldischargeyear"].astype("Int64").astype(str) + "-01-01", errors="coerce")
        + pd.to_timedelta(pd.to_numeric(patient["hospitaladmitoffset"], errors="coerce").fillna(0), unit="m")
    )
    patient["discharge_date"] = patient["admission_date"] + pd.to_timedelta(
        (
            pd.to_numeric(patient["hospitaldischargeoffset"], errors="coerce")
            - pd.to_numeric(patient["hospitaladmitoffset"], errors="coerce")
        ).clip(lower=0).fillna(0),
        unit="m",
    )
    dx = pd.read_csv(
        base / "diagnosis.csv.gz",
        usecols=["patientunitstayid", "diagnosispriority", "icd9code", "diagnosisstring"],
        low_memory=False,
    )
    dx = dx.sort_values(["patientunitstayid", "diagnosispriority"]).drop_duplicates("patientunitstayid", keep="first")
    dx = dx.rename(columns={"icd9code": "icd10", "diagnosisstring": "diagnosis_text"})
    labs = _aggregate_eicu_labs(base / "lab.csv.gz")
    out = patient.merge(dx[["patientunitstayid", "icd10", "diagnosis_text"]], on="patientunitstayid", how="left").merge(
        labs, on="patientunitstayid", how="left"
    )
    out.attrs["lab_source_status"] = "joined from eicu-crd/lab.csv.gz"
    return out.rename(columns={"uniquepid": "mrn"})


def _cdsl(root: Path, _nc_root: Path) -> pd.DataFrame:
    base = root / "physionet" / "cdsl"
    patient = pd.read_csv(
        base / "patient_01.csv",
        usecols=["patient_id", "admission_d_inpat", "discharge_date", "diag_inpat"],
        low_memory=False,
    )
    dx = pd.read_csv(base / "diagnosis_hosp_03.csv", usecols=["patient_id", "dia_ppal"], low_memory=False)
    dx = dx.drop_duplicates("patient_id", keep="first").rename(columns={"dia_ppal": "icd10"})
    labs = _aggregate_cdsl_labs(base / "lab_06.csv")
    out = patient.merge(dx, on="patient_id", how="left").merge(labs, on="patient_id", how="left")
    out.attrs["lab_source_status"] = "joined from lab_06.csv"
    return out.rename(
        columns={
            "patient_id": "mrn",
            "admission_d_inpat": "admission_date",
            "diag_inpat": "diagnosis_text",
        }
    )


def _surrogate_icd_from_text(text: object) -> str:
    s = "" if pd.isna(text) else str(text)
    for pattern, code in TEXT_TO_SURROGATE_ICD:
        if pattern.search(s):
            return code
    return "Z00"


def _whu_from_csv(path: Path, date_cols: tuple[str, str], diagnosis_col: str) -> pd.DataFrame:
    usecols = ["病案号", date_cols[0], date_cols[1], diagnosis_col]
    usecols += [c for c in WHU_LAB_MAP if c in pd.read_csv(path, nrows=0).columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    rename = {"病案号": "mrn", date_cols[0]: "admission_date", date_cols[1]: "discharge_date", diagnosis_col: "diagnosis_text"}
    rename.update({c: WHU_LAB_MAP[c] for c in usecols if c in WHU_LAB_MAP})
    df = df.rename(columns=rename)
    df["icd10"] = df["diagnosis_text"].map(_surrogate_icd_from_text)
    return df


def _whu32k(_root: Path, nc_root: Path) -> pd.DataFrame:
    prepared = nc_root / "_tmp_whu_primary_for_lgdi.csv"
    if prepared.exists():
        return _whu_from_csv(prepared, ("入院时间", "出院时间"), "主要诊断")
    raw = nc_root.parents[3] / "all_admissions.csv"
    return _whu_from_csv(raw, ("入院日期", "出院日期"), "EMR_初步诊断")


def _whu42k(_root: Path, nc_root: Path) -> pd.DataFrame:
    return _whu_from_csv(nc_root / "expanded_cardiac_wide_table.csv", ("入院时间", "出院时间"), "主要诊断")


def _flunet_china(root: Path, _nc_root: Path) -> dict:
    path = root / "flunet" / "flunet_china_2009_2024.csv"
    df = pd.read_csv(path, low_memory=False)
    week_col = "ISO_WEEKSTARTDATE"
    return {
        "dataset": "flunet_china_reference",
        "status": "passed",
        "mode": "reference_series_only",
        "source_rows": int(len(df)),
        "date_start": str(pd.to_datetime(df[week_col], errors="coerce").min().date()),
        "date_end": str(pd.to_datetime(df[week_col], errors="coerce").max().date()),
        "weeks": int(pd.to_datetime(df[week_col], errors="coerce").nunique()),
        "note": "FluNet is an aggregate epidemic reference series, not patient-level admissions data.",
    }


DATASETS = [
    DatasetSpec("mimiciv_hosp", "ehr", _mimic_hosp),
    DatasetSpec("mimiciv_ed", "ehr", _mimic_ed),
    DatasetSpec("nwicu_hosp", "ehr", _nwicu_hosp),
    DatasetSpec("eicu_crd", "ehr", _eicu_crd),
    DatasetSpec("cdsl_inpatient", "ehr", _cdsl),
    DatasetSpec("whu32k_primary", "ehr", _whu32k, target_group="Cardiovascular"),
    DatasetSpec("whu42k_cardiac", "ehr", _whu42k, target_group="Cardiovascular"),
    DatasetSpec("flunet_china_reference", "reference", _flunet_china),
]


def _config_for(df: pd.DataFrame, target_group: str) -> EpidemicConfig:
    dates = pd.to_datetime(df["admission_date"], errors="coerce").dropna()
    min_year = int(dates.dt.year.min()) if len(dates) else 2016
    max_year = int(dates.dt.year.max()) if len(dates) else min_year + 2
    baseline_end_year = min(max_year, min_year + 1)
    return EpidemicConfig(
        target_disease="Multi-dataset validation",
        reference_icd10_codes=["U07.1", "U07.2", "J09", "J10", "J11", "J18"],
        reference_years=list(range(min_year, min(max_year, min_year + 4) + 1)),
        reference_months=list(range(1, 13)),
        baseline_start=f"{min_year}-01-01",
        baseline_end=f"{baseline_end_year}-12-31",
        monitoring_start=f"{min(baseline_end_year + 1, max_year)}-01-01",
        target_group=target_group,
        min_visit_order=1,
        enhanced_features=False,
    )


def _run_ehr_dataset(spec: DatasetSpec, root: Path, nc_root: Path, train_xgb: bool, max_rows: int | None) -> dict:
    t0 = perf_counter()
    raw = spec.loader(root, nc_root)
    assert isinstance(raw, pd.DataFrame)
    source_rows = int(len(raw))
    lab_source_status = str(raw.attrs.get("lab_source_status", "not applicable"))
    df = EHRLoader().from_dataframe(raw)
    df = df.dropna(subset=["mrn", "admission_date"]).copy()
    if "icd10" not in df.columns:
        df["icd10"] = "Z00"
    df["icd10"] = df["icd10"].fillna("Z00").astype(str)
    df = df.sort_values(["mrn", "admission_date"])
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
    config = _config_for(df, spec.target_group)
    result = run_surveillance_pipeline(df, config, train_xgb=train_xgb)
    dates = pd.to_datetime(df["admission_date"], errors="coerce")
    return {
        "dataset": spec.name,
        "status": "passed",
        "mode": "full_pipeline",
        "train_xgb": bool(train_xgb),
        "source_rows": source_rows,
        "rows_tested": int(len(df)),
        "patients": int(df["mrn"].nunique()),
        "date_start": str(dates.min().date()) if pd.notna(dates.min()) else None,
        "date_end": str(dates.max().date()) if pd.notna(dates.max()) else None,
        "labs_detected": [c for c in LAB_COLUMNS if c in df.columns],
        "lab_source_status": lab_source_status,
        "rdi_weeks": int(len(result.rdi_timeline)),
        "lgdi_weeks": int(len(result.lgdi_result.lgdi)),
        "sustained_alerts": int(result.alerts["alert_sustained"].sum()) if "alert_sustained" in result.alerts else 0,
        "onset_week": str(result.warning.onset_week) if result.warning.onset_week is not None else None,
        "peak_week_estimate": str(result.warning.peak_week_estimate) if result.warning.peak_week_estimate is not None else None,
        "model_metrics": result.model_metrics,
        "elapsed_seconds": round(perf_counter() - t0, 2),
    }


def _write_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "# Multi-dataset validation results",
        "",
        "Aggregate-only validation output. No patient-level rows or source data are included.",
        "",
        "| Dataset | Status | Rows tested | Patients | Date range | Labs detected | RDI weeks | LGDI weeks | Sustained alerts | Mode |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for r in results:
        date_range = f"{r.get('date_start')} to {r.get('date_end')}"
        labs = r.get("labs_detected", "")
        if isinstance(labs, list):
            labs = ", ".join(labs) if labs else r.get("lab_source_status", "none")
        lines.append(
            f"| {r['dataset']} | {r['status']} | {r.get('rows_tested', r.get('source_rows', ''))} | "
            f"{r.get('patients', '')} | {date_range} | {labs} | {r.get('rdi_weeks', '')} | "
            f"{r.get('lgdi_weeks', '')} | {r.get('sustained_alerts', '')} | {r.get('mode', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global MIMIC_IV_ROOT
    repo = Path(__file__).resolve().parents[1]
    default_root = repo.parent / "external_data"
    default_nc = repo.parent / "NC_revision"
    ap = argparse.ArgumentParser(description="Aggregate-only public + WHU multi-dataset validation.")
    ap.add_argument("--data-root", type=Path, default=default_root)
    ap.add_argument("--nc-root", type=Path, default=default_nc)
    ap.add_argument(
        "--mimic-iv-root",
        type=Path,
        default=None,
        help="Optional complete MIMIC-IV root containing hosp/labevents.csv.gz (for example mimic-iv-2.2/mimic-iv-2.2).",
    )
    ap.add_argument("--dataset", action="append", choices=[d.name for d in DATASETS], help="Run only selected dataset(s).")
    ap.add_argument("--train-xgb", action="store_true", help="Enable XGBoost training; default validates ingestion and metrics only.")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke runs. Default uses all available rows.")
    ap.add_argument("--json-out", type=Path, default=repo / "validation_outputs" / "multi_dataset_validation_results.json")
    ap.add_argument("--md-out", type=Path, default=repo / "validation_outputs" / "multi_dataset_validation_results.md")
    args = ap.parse_args()
    if args.mimic_iv_root is not None:
        if not (args.mimic_iv_root / "hosp" / "labevents.csv.gz").exists():
            raise FileNotFoundError(args.mimic_iv_root / "hosp" / "labevents.csv.gz")
        MIMIC_IV_ROOT = args.mimic_iv_root

    selected = [d for d in DATASETS if args.dataset is None or d.name in set(args.dataset)]
    results: list[dict] = []
    for spec in selected:
        try:
            if spec.kind == "reference":
                out = spec.loader(args.data_root, args.nc_root)
                assert isinstance(out, dict)
            else:
                out = _run_ehr_dataset(spec, args.data_root, args.nc_root, args.train_xgb, args.max_rows)
        except Exception as e:
            out = {"dataset": spec.name, "status": "failed", "error": f"{type(e).__name__}: {e}"}
        results.append(out)
        print(json.dumps(out, ensure_ascii=False))

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "data_root": str(args.data_root),
        "nc_root": str(args.nc_root),
        "privacy": "aggregate-only; no source rows copied; raw data paths are local runtime inputs",
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(results, args.md_out)
    return 0 if all(r.get("status") == "passed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
