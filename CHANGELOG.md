# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-06

### Added
- `log_explainer(model, data, ...)` — serialize a `shap.Explainer` and attach it to an MLflow run.
- `load_explainer(run_id, artifact_path)` — download and deserialize a previously logged explainer.
- Support for pre-built explainers via `explainer=` kwarg (e.g. `TreeExplainer`, `DeepExplainer`).
- Support for explicit `run_id` (log to a non-active run).
- Custom `artifact_path` and `file_name`.
- Custom exception hierarchy: `MLflowSHAPError`, `NoActiveRunError`, `ExplainerCreationError`.
- Type hints + `py.typed` marker.

[Unreleased]: https://github.com/princepongsakorn/mlflow-shap/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/princepongsakorn/mlflow-shap/releases/tag/v0.1.0
