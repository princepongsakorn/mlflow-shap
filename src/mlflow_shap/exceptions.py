"""Custom exceptions raised by ``mlflow_shap``."""

from __future__ import annotations

__all__ = ["MLflowSHAPError", "NoActiveRunError", "ExplainerCreationError"]


class MLflowSHAPError(Exception):
    """Base class for all errors raised by ``mlflow_shap``."""


class NoActiveRunError(MLflowSHAPError):
    """Raised when ``log_explainer`` is called without an active MLflow run."""


class ExplainerCreationError(MLflowSHAPError):
    """Raised when ``shap.Explainer(model, data)`` fails to construct."""
