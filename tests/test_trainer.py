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
