"""Core API for logging and loading SHAP explainers as MLflow artifacts."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import joblib
import mlflow
import shap

from .exceptions import ExplainerCreationError, NoActiveRunError

__all__ = ["log_explainer", "load_explainer"]

_DEFAULT_ARTIFACT_DIR = "shap_explainer"
_DEFAULT_FILE_NAME = "shap_explainer.pkl"


def log_explainer(
    model: Any,
    data: Any,
    *,
    explainer: shap.Explainer | None = None,
    artifact_path: str = _DEFAULT_ARTIFACT_DIR,
    file_name: str = _DEFAULT_FILE_NAME,
    explainer_kwargs: dict | None = None,
    run_id: str | None = None,
) -> str:
    """Log a SHAP explainer as an MLflow artifact.

    By default this constructs ``shap.Explainer(model, data)`` and serializes it
    via ``joblib`` into the active MLflow run. You can also pass a pre-built
    ``explainer`` to skip construction (useful when you need a specific
    ``TreeExplainer``, ``DeepExplainer``, etc.).

    Args:
        model: The trained model to explain. Ignored if ``explainer`` is supplied.
        data: Background data passed to ``shap.Explainer``. For tabular data,
            a sample of training rows is typical. Ignored if ``explainer`` is supplied.
        explainer: Optional pre-built ``shap.Explainer`` instance. If provided,
            ``model`` and ``data`` are still required for API consistency but not
            used to construct an explainer.
        artifact_path: Directory inside the MLflow run where the artifact is stored.
            Defaults to ``"shap_explainer"``.
        file_name: File name of the serialized pickle. Defaults to ``"shap_explainer.pkl"``.
        explainer_kwargs: Extra kwargs forwarded to ``shap.Explainer(...)``.
        run_id: Target run ID. If omitted, the currently active run is used.

    Returns:
        The full artifact URI of the logged explainer
        (e.g. ``runs:/<id>/shap_explainer/shap_explainer.pkl``).

    Raises:
        NoActiveRunError: If no ``run_id`` is provided and no MLflow run is active.
        ExplainerCreationError: If ``shap.Explainer(model, data)`` raises.

    Example:
        >>> import mlflow
        >>> from mlflow_shap import log_explainer
        >>> with mlflow.start_run():
        ...     log_explainer(model, X_train)
    """
    active = mlflow.active_run()
    if run_id is None and active is None:
        raise NoActiveRunError(
            "No active MLflow run. Wrap the call in `with mlflow.start_run():` "
            "or pass an explicit `run_id`."
        )

    if explainer is None:
        kwargs = explainer_kwargs or {}
        try:
            explainer = shap.Explainer(model, data, **kwargs)
        except Exception as exc:
            raise ExplainerCreationError(
                f"Failed to construct shap.Explainer for model={type(model).__name__}: {exc}"
            ) from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, file_name)
        with open(local_path, "wb") as fh:
            joblib.dump(explainer, fh)

        if run_id is not None:
            client = mlflow.tracking.MlflowClient()
            client.log_artifact(run_id, local_path, artifact_path=artifact_path)
            target_run_id = run_id
        else:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            target_run_id = active.info.run_id  # type: ignore[union-attr]

    return f"runs:/{target_run_id}/{artifact_path}/{file_name}"


def load_explainer(
    run_id: str,
    artifact_path: str = f"{_DEFAULT_ARTIFACT_DIR}/{_DEFAULT_FILE_NAME}",
    *,
    dst_path: str | None = None,
) -> shap.Explainer:
    """Load a SHAP explainer artifact previously logged by :func:`log_explainer`.

    Args:
        run_id: The MLflow run ID where the artifact was logged.
        artifact_path: Path of the artifact relative to the run's artifact root.
            Defaults to ``"shap_explainer/shap_explainer.pkl"``.
        dst_path: Optional local directory to download the artifact into. If omitted,
            MLflow uses a managed temp directory.

    Returns:
        The deserialized ``shap.Explainer`` instance.

    Example:
        >>> from mlflow_shap import load_explainer
        >>> explainer = load_explainer(run_id="abc123")
        >>> shap_values = explainer(X_test)
    """
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=dst_path,
    )
    return joblib.load(local_path)
