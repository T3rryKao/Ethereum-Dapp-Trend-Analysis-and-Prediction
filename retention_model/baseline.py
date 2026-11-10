"""Q5: Simple M1-M3 baselines for predicting M12 cohort retention.

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

DATA_PATH = Path(__file__).resolve().parent / "tables" / "dataset_k3.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "tables"
TARGET = "y_retention_12"
FEATURE = "retention_m3"
KEYS = ["protocol", "cohort_month"]


def score_regression(y, pred, name: str, **extra) -> dict:
    """Compute regression metrics on the same evaluation cohorts."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError(f"Non-finite target or prediction in {name}")
    return {
        "model": name,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "n": int(len(y)),
        **extra,
    }


class RetentionBaseline:
    """Serializable forms of the existing median, persistence and linear rules."""
    def __init__(self, kind):
        self.kind = kind

    def fit(self, X, y):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if self.kind == "historical_median":
            self.constant_ = float(np.median(y))
        elif self.kind == "m3_persistence":
            self.constant_ = float(X[FEATURE].median())
            if not np.isfinite(self.constant_):
                raise ValueError("Training has no observed M3 retention values")
        elif self.kind == "univariate_linear":
            self.pipeline_ = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
            self.pipeline_.fit(X[[FEATURE]], y)
        else:
            raise ValueError(f"Unknown baseline: {self.kind}")
        return self

    def predict(self, X):
        if self.kind == "historical_median":
            pred = np.full(len(X), self.constant_)
        elif self.kind == "m3_persistence":
            pred = X[FEATURE].fillna(self.constant_).to_numpy(dtype=float)
        else:
            pred = self.pipeline_.predict(X[[FEATURE]])
        return np.clip(pred, 0., 1.)


def make_baselines():
    return {name: RetentionBaseline(name) for name in
            ("historical_median", "m3_persistence", "univariate_linear")}


def _baseline_result(training, test, target, name):
    model = make_baselines()[name].fit(training[[FEATURE]], training[target])
    pred = model.predict(test[[FEATURE]])
    return pred, score_regression(test[target], pred, name)


def historical_median(train, test, target):
    return _baseline_result(train, test, target, "historical_median")


def m3_persistence(train, test, target):
    return _baseline_result(train, test, target, "m3_persistence")


def univariate_linear(train, test, target):
    return _baseline_result(train, test, target, "univariate_linear")


def run_all_baselines(train, test, target=TARGET):
    return pd.DataFrame([fn(train, test, target)[1] for fn in
                         (historical_median, m3_persistence, univariate_linear)])


def main():
    import argparse
    from datetime import datetime, timezone
    from splits import time_split
    from train import eligible_frame
    parser = argparse.ArgumentParser(description="Fixed baselines on chronological Train/Test split")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or OUTPUT_DIR.parent / "runs_baselines_train_test" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if output.exists():
        parser.error("Choose a fresh --output-dir")
    training, test = time_split(eligible_frame(pd.read_csv(DATA_PATH)))
    scores = run_all_baselines(training, test).sort_values("mae")
    scores["n_train"], scores["n_test"] = len(training), len(test)
    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output / "test_baseline_comparison.csv", index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
