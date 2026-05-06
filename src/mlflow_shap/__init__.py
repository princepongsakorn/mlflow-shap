"""``mlflow-shap`` — log and load SHAP explainers as MLflow artifacts.

Public API:
    log_explainer:  serialize a ``shap.Explainer`` and attach it to an MLflow run.
    load_explainer: download and deserialize a previously logged explainer.

Exceptions:
    MLflowSHAPError, NoActiveRunError, ExplainerCreationError.
"""

from __future__ import annotations

from .exceptions import ExplainerCreationError, MLflowSHAPError, NoActiveRunError
from .explainer import load_explainer, log_explainer

__version__ = "0.1.0"

__all__ = [
    "log_explainer",
    "load_explainer",
    "MLflowSHAPError",
    "NoActiveRunError",
    "ExplainerCreationError",
    "__version__",
]
