"""Feature engineering — disease-agnostic."""
from ehr_sentinel.features.comorbidity import ComorbidityGrouper
from ehr_sentinel.features.temporal import (
    compute_los_gap,
    compute_visit_order,
    compute_seasonal_features,
    compute_prior_stats,
)
from ehr_sentinel.features.builder import FeatureBuilder
from ehr_sentinel.features.adaptive import (
    AutoFeatureEngineer,
    DiseaseDetector,
    DiseaseSignal,
    FeaturePlan,
    LabPanelAdapter,
    LabPanelSpec,
    build_adaptive_config,
)

__all__ = [
    "ComorbidityGrouper",
    "compute_los_gap",
    "compute_visit_order",
    "compute_seasonal_features",
    "compute_prior_stats",
    "FeatureBuilder",
    "AutoFeatureEngineer",
    "DiseaseDetector",
    "DiseaseSignal",
    "FeaturePlan",
    "LabPanelAdapter",
    "LabPanelSpec",
    "build_adaptive_config",
]
