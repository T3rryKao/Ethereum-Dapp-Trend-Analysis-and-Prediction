"""Q5 chronological Train/Test evaluation of fixed models; no model selection.

Run the full workflow with python retention_model/run_all.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from features import FEATURE_COLS
from labels import DATA_END
from splits import time_split, FINAL_CUTOFF, SCHEMA_VERSION, WORKFLOW_VERSION
from baseline import make_baselines

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "tables" / "dataset_k3.csv"
DEFAULT_PREDICTION_MODEL = "ridge"
TARGET = "y_retention_12"
KEYS = ["protocol", "cohort_month"]
SEED = 42


# ---------------------------------------------------------------------------
# Data validation / splits
# ---------------------------------------------------------------------------
def eligible_frame(ds: pd.DataFrame) -> pd.DataFrame:
    if "schema_version" not in ds or not ds["schema_version"].eq(SCHEMA_VERSION).all():
        raise ValueError("Outdated dataset semantics; rerun retention_model/features.py")
    required = KEYS + [TARGET] + FEATURE_COLS
    missing = sorted(set(required) - set(ds.columns))
    if missing:
        raise ValueError(
            f"Missing columns in dataset_k3.csv: {missing}. "
            "Rerun: python retention_model/features.py"
        )
    out = ds.copy()
    out["cohort_month"] = pd.to_datetime(out["cohort_month"], errors="raise")
    if out.duplicated(KEYS).any():
        raise ValueError("Duplicate (protocol, cohort_month) keys")
    for col in FEATURE_COLS + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[FEATURE_COLS] = out[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    if out[TARGET].isna().any() or not out[TARGET].between(0, 1).all():
        raise ValueError("Target must be observed and in [0, 1]")
    return out


# ---------------------------------------------------------------------------
# Preprocessing / models
# ---------------------------------------------------------------------------
def make_preprocessor(scale: bool = False) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def make_models() -> dict[str, Pipeline]:
    """Fit all preprocessing on training data only."""
    return {
        "ridge": Pipeline([
            ("pre", make_preprocessor(scale=True)),
            ("reg", Ridge(alpha=10.0)),
        ]),
        "decision_tree": Pipeline([
            ("pre", make_preprocessor()),
            ("reg", DecisionTreeRegressor(
                max_depth=5, min_samples_leaf=20, random_state=SEED
            )),
        ]),
        "random_forest": Pipeline([
            ("pre", make_preprocessor()),
            ("reg", RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=SEED, n_jobs=-1
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("pre", make_preprocessor()),
            ("reg", HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0,
                random_state=SEED
            )),
        ]),
    }


def fit_predict(model: Pipeline, train: pd.DataFrame, evaluation: pd.DataFrame,
                feature_cols: list[str] = FEATURE_COLS, sample_weight=None) -> np.ndarray:
    fit_params = {}
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(train),) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Training weights must be finite, positive and aligned with rows")
        fit_params["reg__sample_weight"] = weights
    # Preprocessing stays training-only and unchanged; weights affect the regressor.
    model.fit(train[feature_cols], train[TARGET], **fit_params)
    return np.clip(model.predict(evaluation[feature_cols]), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def score_regression(y, pred, name: str, **extra) -> dict:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return {
        "model": name,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "n": int(len(y)),
        **extra,
    }


def evaluate_models(training, evaluation, feature_cols=FEATURE_COLS,
                    include_baselines=True, return_predictions=False):
    """Fit each fixed predictor once, on training only; score identical test rows."""
    models = {**(make_baselines() if include_baselines else {}), **make_models()}
    rows, predictions = [], evaluation[KEYS + [TARGET]].copy()
    for name, model in models.items():
        pred = fit_predict(model, training, evaluation, feature_cols)
        kind = "baseline" if name in make_baselines() else "machine_learning"
        rows.append(score_regression(evaluation[TARGET], pred, name,
                                     model_type=kind, n_train=len(training),
                                     n_test=len(evaluation)))
        predictions[name] = pred
    scores = pd.DataFrame(rows).sort_values(["mae", "model"]).reset_index(drop=True)
    if return_predictions:
        return scores, models, predictions
    return scores, models


def model_bundle(model, name, training, cutoff):
    return {
        "model": model, "model_name": name, "feature_cols": list(FEATURE_COLS),
        "target": TARGET, "prediction_policy": "predefined_not_test_selected",
        "schema_version": SCHEMA_VERSION, "workflow_version": WORKFLOW_VERSION,
        "data_end": str(DATA_END), "training_label_cutoff": str(cutoff),
        "training_cohort_min": str(training["cohort_month"].min().date()),
        "training_cohort_max": str(training["cohort_month"].max().date()),
        "coverage_assumption": "complete sparse activity export through data_end",
    }


def save_model_bundle(bundle, path):
    with Path(path).open("xb") as stream:
        joblib.dump(bundle, stream)


def run_experiment(dataset, scoring, output, prediction_model=DEFAULT_PREDICTION_MODEL):
    """The prediction model is fixed by configuration before test evaluation."""
    import predict
    if prediction_model not in make_models():
        raise ValueError("Prediction model must be one of the four predefined learners")
    dataset = eligible_frame(dataset)
    training, test = time_split(dataset)
    print(f"Chronological split: Train={len(training)} Test={len(test)}; cutoff={FINAL_CUTOFF}")
    print(f"Predefined prediction model: {prediction_model} (not chosen by test performance)")
    scores, fitted, test_predictions = evaluate_models(training, test, return_predictions=True)
    scores.to_csv(output / "test_model_comparison.csv", index=False)
    test_predictions.to_csv(output / "test_predictions.csv", index=False)
    artifacts = output / "artifacts"
    artifacts.mkdir(exist_ok=False)
    for name, model in fitted.items():
        bundle = model_bundle(model, name, training, FINAL_CUTOFF)
        save_model_bundle(bundle, artifacts / f"{name}.joblib")
        if name == prediction_model:
            save_model_bundle(bundle, artifacts / "q5_train_test_model.joblib")
    # Explicit path prevents accidentally loading a previous experiment.
    loaded, cols = predict.load_trained_model("predefined", artifacts)
    scored = predict.score_dataset(scoring, loaded, cols)
    scored.to_csv(output / "predictions_all_k3.csv", index=False)
    print("\n2024 TEST COMPARISON — descriptive ranking only; ascending test MAE")
    print(scores[["model", "model_type", "mae", "rmse", "r2", "n_train", "n_test"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"Predictions: {len(scored)} cohorts using predefined {prediction_model}.")
    print("The 2024 test set was examined in prior experiments; it is not an untouched independent holdout.")
    print("No tuning or model selection uses this test ranking. All-cohort predictions include training rows.")
    return dict(prediction_model=prediction_model, prediction_policy="predefined_not_test_selected",
                n_train=len(training), n_test=len(test), n_predictions=len(scored))


def main():
    from datetime import datetime, timezone
    parser = argparse.ArgumentParser(description="Fixed-model chronological Train/Test experiment")
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--features", type=Path, default=ROOT / "tables/features_k3.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prediction-model", choices=list(make_models()), default=DEFAULT_PREDICTION_MODEL)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "runs_train_test" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if output.exists():
        parser.error("Choose a fresh --output-dir; existing outputs are preserved")
    output.mkdir(parents=True, exist_ok=False)
    run_experiment(pd.read_csv(args.input), pd.read_csv(args.features), output, args.prediction_model)
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
