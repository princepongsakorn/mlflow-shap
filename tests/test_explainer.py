"""Integration tests for log_explainer / load_explainer."""

from __future__ import annotations

import mlflow
import pytest
import shap

from mlflow_shap import (
    ExplainerCreationError,
    NoActiveRunError,
    load_explainer,
    log_explainer,
)


def test_log_and_load_round_trip(tmp_mlflow_tracking, trained_rf, background_data):
    model, _, _ = trained_rf

    with mlflow.start_run() as run:
        uri = log_explainer(model, background_data)
        run_id = run.info.run_id

    assert uri == f"runs:/{run_id}/shap_explainer/shap_explainer.pkl"

    explainer = load_explainer(run_id)
    assert isinstance(explainer, shap.Explainer)

    # The loaded explainer should still work end-to-end on a small input.
    shap_values = explainer(background_data[:2])
    assert shap_values.values.shape[0] == 2


def test_log_explainer_returns_artifact_uri(tmp_mlflow_tracking, trained_rf, background_data):
    model, _, _ = trained_rf

    with mlflow.start_run() as run:
        uri = log_explainer(
            model,
            background_data,
            artifact_path="custom/dir",
            file_name="my_explainer.pkl",
        )
        assert uri == f"runs:/{run.info.run_id}/custom/dir/my_explainer.pkl"


def test_log_explainer_without_active_run_raises(tmp_mlflow_tracking, trained_rf, background_data):
    model, _, _ = trained_rf

    # No active run, no run_id passed → should raise.
    with pytest.raises(NoActiveRunError):
        log_explainer(model, background_data)


def test_log_explainer_accepts_explicit_run_id(tmp_mlflow_tracking, trained_rf, background_data):
    model, _, _ = trained_rf

    # Create a run, end it, then log into it via run_id.
    with mlflow.start_run() as run:
        run_id = run.info.run_id

    uri = log_explainer(model, background_data, run_id=run_id)
    assert run_id in uri

    loaded = load_explainer(run_id)
    assert isinstance(loaded, shap.Explainer)


def test_log_explainer_accepts_prebuilt_explainer(
    tmp_mlflow_tracking, trained_rf, background_data
):
    model, _, _ = trained_rf
    prebuilt = shap.TreeExplainer(model)

    with mlflow.start_run() as run:
        log_explainer(model, background_data, explainer=prebuilt)
        run_id = run.info.run_id

    loaded = load_explainer(run_id)
    assert isinstance(loaded, shap.TreeExplainer)


def test_explainer_creation_error_wraps_underlying(tmp_mlflow_tracking, background_data):
    class NotAModel:
        """An object shap.Explainer can't infer."""

    with mlflow.start_run():
        with pytest.raises(ExplainerCreationError):
            log_explainer(NotAModel(), background_data)


def test_custom_artifact_path_round_trip(tmp_mlflow_tracking, trained_rf, background_data):
    model, _, _ = trained_rf

    with mlflow.start_run() as run:
        log_explainer(
            model,
            background_data,
            artifact_path="explainability/shap",
            file_name="rf_v2.pkl",
        )
        run_id = run.info.run_id

    loaded = load_explainer(run_id, artifact_path="explainability/shap/rf_v2.pkl")
    assert isinstance(loaded, shap.Explainer)
