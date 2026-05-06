# mlflow-shap

[![PyPI version](https://img.shields.io/pypi/v/mlflow-shap.svg)](https://pypi.org/project/mlflow-shap/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlflow-shap.svg)](https://pypi.org/project/mlflow-shap/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Log and load [SHAP](https://github.com/shap/shap) explainers as [MLflow](https://mlflow.org/) artifacts with a one-liner. Drop-in for any model that `shap.Explainer` accepts (tree models, linear models, deep nets, etc.).

---

## Installation

```bash
pip install mlflow-shap
```

## Quick start

### Log an explainer

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from mlflow_shap import log_explainer

with mlflow.start_run() as run:
    model = RandomForestClassifier().fit(X_train, y_train)
    mlflow.sklearn.log_model(model, artifact_path="model")
    log_explainer(model, X_train)
```

### Load it back

```python
from mlflow_shap import load_explainer

explainer = load_explainer(run_id="abc123")
shap_values = explainer(X_test)
```

That's it. The explainer is serialized with `joblib` and stored under the run's `shap_explainer/` artifact directory.

---

## API

### `log_explainer`

```python
log_explainer(
    model,
    data,
    *,
    explainer=None,
    artifact_path="shap_explainer",
    file_name="shap_explainer.pkl",
    explainer_kwargs=None,
    run_id=None,
) -> str
```

Returns the artifact URI (`runs:/<id>/<artifact_path>/<file_name>`).

| Argument | Description |
|---|---|
| `model` | Trained model. Ignored when `explainer` is supplied. |
| `data` | Background data passed to `shap.Explainer`. |
| `explainer` | Pre-built `shap.Explainer` (e.g. `TreeExplainer`, `DeepExplainer`). Skips auto-construction. |
| `artifact_path` | Sub-directory inside the run's artifact root. Default `shap_explainer`. |
| `file_name` | Pickle file name. Default `shap_explainer.pkl`. |
| `explainer_kwargs` | Extra kwargs forwarded to `shap.Explainer(...)`. |
| `run_id` | Target run. Default uses the active run. |

### `load_explainer`

```python
load_explainer(
    run_id,
    artifact_path="shap_explainer/shap_explainer.pkl",
    *,
    dst_path=None,
) -> shap.Explainer
```

Downloads the artifact and deserializes via `joblib`.

---

## Advanced usage

### Use a specific explainer

```python
import shap
from mlflow_shap import log_explainer

tree_explainer = shap.TreeExplainer(model)
log_explainer(model, X_train, explainer=tree_explainer)
```

### Log into a specific run (no active run)

```python
log_explainer(model, X_train, run_id="abc123")
```

### Custom artifact location

```python
log_explainer(model, X_train, artifact_path="explainability/shap", file_name="rf_v2.pkl")
```

---

## Exceptions

| Exception | When raised |
|---|---|
| `NoActiveRunError` | No active MLflow run and no `run_id` was passed. |
| `ExplainerCreationError` | `shap.Explainer(model, data)` failed. Wraps the original error. |
| `MLflowSHAPError` | Base class for the above. |

---

## Compatibility

- Python: 3.9, 3.10, 3.11, 3.12
- MLflow: >= 2.0
- SHAP: >= 0.42

---

## Development

```bash
git clone https://github.com/princepongsakorn/mlflow-shap.git
cd mlflow-shap
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
