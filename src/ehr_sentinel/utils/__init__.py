"""Shared utilities — configuration, GPU detection, validation."""
from ehr_sentinel.utils.config import EpidemicConfig, PresetConfigs
from ehr_sentinel.utils.gpu import detect_device, DeviceInfo

__all__ = ["EpidemicConfig", "PresetConfigs", "detect_device", "DeviceInfo"]
