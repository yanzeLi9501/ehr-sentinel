"""ehr-sentinel — general-purpose EHR epidemic surveillance toolkit.

A disease-agnostic, MIT-licensed package that combines XGBoost rhythm
prediction with Pearson profile correlation (RDI) and the LOS–Gap Deviation
Index (LGDI) for epidemic early warning.
"""
from __future__ import annotations

__version__ = "0.2.0"
__license__ = "MIT"
__author__ = "Yanze Li, Lingfeng Zha"

# Eager (cheap) imports
from ehr_sentinel.utils.config import EpidemicConfig, PresetConfigs
from ehr_sentinel.data.synthetic import (
    generate_admissions,
    generate_epidemic_signal,
    generate_fhir_bundle,
)
from ehr_sentinel.data.loader import EHRLoader, DataSourceProfile
from ehr_sentinel.data.terminology import TerminologyMapper
from ehr_sentinel.features.comorbidity import ComorbidityGrouper
from ehr_sentinel.features.builder import FeatureBuilder
from ehr_sentinel.metrics.pearson import PearsonProfileCorrelation
from ehr_sentinel.metrics.lgdi import LGDIComputer
from ehr_sentinel.metrics.evaluation import AlertEvaluator
from ehr_sentinel.alerts.engine import ConsensusRule, SeasonFilter, SustainedRule
from ehr_sentinel.alerts.epidemic import EpidemicPredictor
from ehr_sentinel.reporting.tables import ReportGenerator
from ehr_sentinel.utils.gpu import detect_device


def __getattr__(name: str):
    # Lazy imports for heavy optional dependencies
    if name in {"XGBTrainer", "XGBTuner", "ModelPersistence"}:
        from ehr_sentinel.models import trainer, tuner, persistence
        return {
            "XGBTrainer": trainer.XGBTrainer,
            "XGBTuner": tuner.XGBTuner,
            "ModelPersistence": persistence.ModelPersistence,
        }[name]
    if name == "FHIRParser":
        from ehr_sentinel.data.fhir_parser import FHIRParser
        return FHIRParser
    if name == "FHIRExporter":
        from ehr_sentinel.reporting.fhir_export import FHIRExporter
        return FHIRExporter
    if name == "DashboardRenderer":
        from ehr_sentinel.reporting.dashboard import DashboardRenderer
        return DashboardRenderer
    if name == "run_surveillance_pipeline":
        from ehr_sentinel.pipeline import run_surveillance_pipeline as _f
        return _f
    if name == "SurveillanceResult":
        from ehr_sentinel.pipeline import SurveillanceResult as _C
        return _C
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__license__",
    "EpidemicConfig",
    "PresetConfigs",
    "EHRLoader",
    "DataSourceProfile",
    "TerminologyMapper",
    "FHIRParser",
    "ComorbidityGrouper",
    "FeatureBuilder",
    "PearsonProfileCorrelation",
    "LGDIComputer",
    "AlertEvaluator",
    "ConsensusRule",
    "SeasonFilter",
    "SustainedRule",
    "EpidemicPredictor",
    "ReportGenerator",
    "DashboardRenderer",
    "FHIRExporter",
    "XGBTrainer",
    "XGBTuner",
    "ModelPersistence",
    "detect_device",
    "generate_admissions",
    "generate_epidemic_signal",
    "generate_fhir_bundle",
    "run_surveillance_pipeline",
    "SurveillanceResult",
]
