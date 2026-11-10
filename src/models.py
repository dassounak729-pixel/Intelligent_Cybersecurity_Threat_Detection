"""Model layer: classifier candidates, anomaly detector and persisted bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from src.utils import get_logger

logger = get_logger("models")


def build_candidate_models(seed: int = config.RANDOM_SEED) -> dict[str, Pipeline]:
    """Return the supervised candidates, each wrapped with an imputer."""
    params = config.MODEL_PARAMS
    imputer = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "Logistic Regression": Pipeline([
            ("impute", imputer()),
            ("log", FunctionTransformer(np.log1p)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(random_state=seed, **params["Logistic Regression"])),
        ]),
        "Decision Tree": Pipeline([
            ("impute", imputer()),
            ("model", DecisionTreeClassifier(random_state=seed, **params["Decision Tree"])),
        ]),
        "Random Forest": Pipeline([
            ("impute", imputer()),
            ("model", RandomForestClassifier(random_state=seed, n_jobs=config.N_JOBS,
                                             **params["Random Forest"])),
        ]),
        "Gradient Boosting": Pipeline([
            ("impute", imputer()),
            ("model", HistGradientBoostingClassifier(random_state=seed, **params["Gradient Boosting"])),
        ]),
    }


def compare_models(X: pd.DataFrame, y: pd.Series, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Cross-validate every candidate and return a ranked comparison table."""
    folds = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for name, pipeline in build_candidate_models(seed).items():
        scores = cross_validate(pipeline, X, y, cv=folds, scoring=["accuracy", "f1_macro"],
                                n_jobs=1)
        rows.append({
            "model": name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_f1_macro": scores["test_f1_macro"].mean(),
            "cv_f1_macro_std": scores["test_f1_macro"].std(),
            "fit_seconds": scores["fit_time"].mean(),
        })
        logger.info("%-20s accuracy=%.4f f1_macro=%.4f", name, rows[-1]["cv_accuracy"], rows[-1]["cv_f1_macro"])
    table = pd.DataFrame(rows)
    return table.sort_values(f"cv_{config.MODEL_SELECTION_METRIC}", ascending=False).reset_index(drop=True)


def train_classifier(model_name: str, X: pd.DataFrame, y: pd.Series,
                     seed: int = config.RANDOM_SEED) -> Pipeline:
    """Fit the named candidate on the full training data."""
    candidates = build_candidate_models(seed)
    if model_name not in candidates:
        raise KeyError(f"Unknown model '{model_name}'. Options: {list(candidates)}")
    return candidates[model_name].fit(X, y)


def compute_feature_importances(model: Pipeline, X: pd.DataFrame, y: pd.Series,
                                seed: int = config.RANDOM_SEED) -> pd.Series:
    """Global feature importance (native if available, otherwise permutation)."""
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        sample = X.sample(min(len(X), config.PERMUTATION_SAMPLE), random_state=seed)
        result = permutation_importance(model, sample, y.loc[sample.index], n_repeats=3,
                                        random_state=seed, n_jobs=config.N_JOBS)
        values = np.clip(result.importances_mean, 0, None)
    importances = pd.Series(values, index=X.columns, dtype=float)
    return (importances / importances.sum()).sort_values(ascending=False)


class AnomalyDetector:
    """Isolation Forest trained on benign traffic only, with a 0-1 calibrated score."""

    def __init__(self, seed: int = config.RANDOM_SEED) -> None:
        self.pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", IsolationForest(random_state=seed, n_jobs=config.N_JOBS, **config.ANOMALY_PARAMS)),
        ])
        self.low_: float | None = None
        self.high_: float | None = None

    def _raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Higher value = more anomalous."""
        imputed = self.pipeline.named_steps["impute"].transform(X)
        return -self.pipeline.named_steps["model"].score_samples(imputed)

    def fit(self, X_benign: pd.DataFrame) -> "AnomalyDetector":
        """Learn the benign baseline and calibrate the score range."""
        self.pipeline.fit(X_benign)
        raw = self._raw_scores(X_benign)
        self.low_ = float(np.percentile(raw, config.ANOMALY_CALIBRATION_LOW_PCT))
        self.high_ = float(np.percentile(raw, config.ANOMALY_CALIBRATION_HIGH_PCT))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores in [0, 1]."""
        if self.low_ is None or self.high_ is None:
            raise RuntimeError("AnomalyDetector must be fitted before scoring.")
        span = max(self.high_ - self.low_, 1e-9)
        return np.clip((self._raw_scores(X) - self.low_) / span, 0.0, 1.0)


@dataclass
class ModelBundle:
    """Everything the detection engine needs, persisted as a single artefact."""

    classifier: Pipeline
    anomaly_detector: AnomalyDetector
    feature_columns: list[str]
    classes: list[str]
    importances: pd.Series
    baseline_median: pd.Series
    baseline_log_mean: pd.Series
    baseline_log_std: pd.Series
    metadata: dict = field(default_factory=dict)
    entity_baseline: object | None = None  # EntityBaselineStore, when per-entity features are used


def build_baseline(X_benign: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Benign-traffic reference statistics used for explanations."""
    logged = np.log1p(X_benign.clip(lower=0))
    return (X_benign.median(), logged.mean(), logged.std().clip(lower=1e-3))


def save_bundle(bundle: ModelBundle, path: Path = config.MODEL_BUNDLE_FILE) -> None:
    """Persist the model bundle with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Model bundle saved to %s", path)


def load_bundle(path: Path = config.MODEL_BUNDLE_FILE) -> ModelBundle:
    """Load a persisted model bundle."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Model bundle not found at {path}. Run `python main.py train` first.")
    return joblib.load(path)
