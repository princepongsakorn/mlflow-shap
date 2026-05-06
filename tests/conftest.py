"""Shared pytest fixtures."""

from __future__ import annotations

import mlflow
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def tmp_mlflow_tracking(tmp_path, monkeypatch):
    """Point MLflow at a clean local file-store under tmp_path."""
    uri = f"file://{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri


@pytest.fixture
def trained_rf():
    """A small RandomForest fit on synthetic tabular data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def background_data(trained_rf):
    _, X, _ = trained_rf
    # Use a tiny background sample to keep tests fast
    return np.asarray(X[:20])
