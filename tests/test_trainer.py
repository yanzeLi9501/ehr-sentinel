import pytest

from ehr_sentinel import EHRLoader, FeatureBuilder, detect_device

xgboost = pytest.importorskip("xgboost")


def test_train_gap_model_synthetic(synthetic_admissions, synthetic_config_covid):
    from ehr_sentinel.models.trainer import XGBTrainer

    df = EHRLoader().from_dataframe(synthetic_admissions)
    fm = FeatureBuilder(synthetic_config_covid).build(df)
    mask = fm.y_gap.notna()
    trainer = XGBTrainer(params={"n_estimators": 80, "max_depth": 4, "learning_rate": 0.1})
    result = trainer.train_gap_model(fm.X.loc[mask], fm.y_gap.loc[mask])
    assert result.metrics["train_r2"] > 0
    assert result.target == "gap"
    assert result.device in ("cuda", "cpu")


def test_cross_validate(synthetic_admissions, synthetic_config_covid):
    from ehr_sentinel.models.trainer import XGBTrainer

    df = EHRLoader().from_dataframe(synthetic_admissions)
    fm = FeatureBuilder(synthetic_config_covid).build(df)
    mask = fm.y_gap.notna()
    trainer = XGBTrainer(params={"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1})
    cv = trainer.cross_validate(fm.X.loc[mask], fm.y_gap.loc[mask], fm.groups.loc[mask], n_splits=3)
    assert len(cv["r2_test"]) == 3
    assert all(isinstance(x, float) for x in cv["r2_test"])


def test_gpu_detection():
    dev = detect_device()
    assert dev.device in ("cuda", "cpu")
    assert dev.tree_method == "hist"


def test_model_persistence(tmp_path, synthetic_admissions, synthetic_config_covid):
    from ehr_sentinel.models.trainer import XGBTrainer
    from ehr_sentinel.models.persistence import ModelPersistence

    df = EHRLoader().from_dataframe(synthetic_admissions)
    fm = FeatureBuilder(synthetic_config_covid).build(df)
    mask = fm.y_gap.notna()
    trainer = XGBTrainer(params={"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1})
    result = trainer.train_gap_model(fm.X.loc[mask], fm.y_gap.loc[mask])
    path = tmp_path / "model.joblib"
    ModelPersistence.save(result.model, path, fm.X.columns.tolist(), "gap", result.metrics)
    model, meta = ModelPersistence.load(path)
    assert meta["target"] == "gap"
    assert "feature_names" in meta


def test_default_random_state():
    """XGBTrainer default random_state must match the GPU-tuned 2026-05-13 version."""
    from ehr_sentinel.models.trainer import XGBTrainer
    assert XGBTrainer().random_state == 20260513


def test_pipeline_dual_xgb_sliding_window_lgdi(synthetic_admissions_with_signal, synthetic_config_covid):
    """Full pipeline with train_xgb=True should use SlidingWindowLGDI (dual signed-MASE)."""
    from ehr_sentinel import run_surveillance_pipeline

    # Use fast XGBoost params and low LGDI window thresholds for synthetic test data
    import ehr_sentinel.models.trainer as _t
    _orig = _t.DEFAULT_PARAMS.copy()
    _t.DEFAULT_PARAMS.update({"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1})

    # Lower LGDI window minimums for the small synthetic dataset
    cfg = synthetic_config_covid.model_copy(
        update={"lgdi_min_window_n": 10, "lgdi_min_group_n": 2}
    )

    try:
        result = run_surveillance_pipeline(
            synthetic_admissions_with_signal,
            cfg,
            train_xgb=True,
        )
    finally:
        _t.DEFAULT_PARAMS.clear()
        _t.DEFAULT_PARAMS.update(_orig)

    # Dual-model path should be taken
    assert result.model_metrics.get("lgdi_mode") == "sliding_window_dual_signed_mase", (
        f"Expected sliding_window mode, got: {result.model_metrics}"
    )
    # Timeline should have window_anchor column (sliding-window format)
    assert "window_anchor" in result.lgdi_result.timeline.columns
    assert len(result.lgdi_result.lgdi) > 0, "Should produce valid LGDI rows with relaxed min_n"
