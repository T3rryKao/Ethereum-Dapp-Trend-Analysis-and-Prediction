"""Q5: Predict M12 wallet-cohort retention from M1-M3 features.

Run from the repository root:
    python retention_model/predict.py
    python retention_model/predict.py --cohort-year 2024
    python retention_model/predict.py --unlabeled-only
    python retention_model/predict.py --model final

Default: score all available feature cohorts. The validation artifact is fitted
using labels available through March 2022; default scores include training rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from splits import SCHEMA_VERSION
from labels import DATA_END

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
ARTIFACTS = ROOT / "artifacts"
TARGET = "y_retention_12"
KEYS = ["protocol", "cohort_month"]


def load_trained_model(which: str = "validation", artifacts_dir: Path | None = None):
    """Load a run-local Train/Test pipeline or a legacy validation/final artifact."""
    names = {
        "predefined": "q5_train_test_model.joblib",
        "validation": "q5_best_validation_model.joblib",
        "final": "q5_final_model.joblib",
    }
    if which not in names:
        raise ValueError("Model must be predefined, validation or final")
    path = (Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS) / names[which]
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "For the current Train/Test workflow, run retention_model/run_all.py "
            "and load 'predefined' from that run's artifacts directory."
        )
    blob = joblib.load(path)  # Load only files you trust.
    if blob.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Outdated model semantics. Regenerate features and rerun train.py")
    model = blob["model"]
    feature_cols = list(blob["feature_cols"])
    if blob.get("target") != TARGET:
        raise ValueError("Saved model target does not match Q5 retention target")
    if list(model.feature_names_in_) != feature_cols:
        raise ValueError("Saved feature order does not match the fitted pipeline")
    return model, feature_cols


def unlabeled_cohorts(ds: pd.DataFrame) -> pd.DataFrame:
    """Select immature outcomes by calendar, independent of cached labels."""
    months = pd.to_datetime(ds["cohort_month"], errors="raise").dt.to_period("M")
    return ds.loc[(months + 3 <= DATA_END) & (months + 12 > DATA_END)].copy()


def load_scoring_data(path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Prefer the unlabeled feature table so new cohorts can be scored."""
    if path is None:
        features = TABLES / "features_k3.csv"
        dataset = TABLES / "dataset_k3.csv"
        path = features if features.exists() else dataset
    if not path.exists():
        raise FileNotFoundError(
            f"Scoring data not found: {path}\n"
            "Run: python retention_model/features.py"
        )
    df = pd.read_csv(path)
    required = KEYS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing identifier columns: {missing}")
    df["cohort_month"] = pd.to_datetime(df["cohort_month"], errors="raise")
    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate (protocol, cohort_month) rows in scoring data")
    if df[KEYS].isna().any().any():
        raise ValueError("Missing cohort identifiers in scoring data")
    return df, path


def score_dataset(ds: pd.DataFrame, model, feature_cols: list[str]) -> pd.DataFrame:
    """Predict a retention fraction, not a classification probability."""
    if "schema_version" not in ds or not ds["schema_version"].eq(SCHEMA_VERSION).all():
        raise ValueError("Outdated feature semantics; rerun retention_model/features.py")
    missing = [c for c in feature_cols if c not in ds.columns]
    if missing:
        raise ValueError(
            f"Missing model features: {missing}. "
            "Regenerate features_k3.csv with the matching features.py."
        )
    out = ds.copy()
    X = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if len(X) == 0:
        raise ValueError("No cohorts remain after filtering")
    predictions = np.asarray(model.predict(X), dtype=float)
    if predictions.shape != (len(out),) or not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid retention predictions")
    out["predicted_retention_12"] = np.clip(predictions, 0.0, 1.0)
    out["predicted_retention_12_pct"] = 100 * out["predicted_retention_12"]
    if "retention_m3" in out.columns:
        out["expected_decline_m3_to_m12"] = (
            pd.to_numeric(out["retention_m3"], errors="coerce")
            - out["predicted_retention_12"]
        )
    if TARGET in out.columns:
        actual = pd.to_numeric(out[TARGET], errors="coerce")
        out["prediction_error"] = out["predicted_retention_12"] - actual
        out["absolute_error"] = out["prediction_error"].abs()
    out = out.sort_values("predicted_retention_12", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    priority = KEYS + [
        "cohort_size", "retention_m1", "retention_m2", "retention_m3",
        "predicted_retention_12", "predicted_retention_12_pct",
        "expected_decline_m3_to_m12", "rank", TARGET,
        "prediction_error", "absolute_error",
    ]
    return out[[c for c in priority if c in out.columns] +
               [c for c in out.columns if c not in priority]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Q5 M3 -> M12 cohort retention predictions")
    parser.add_argument("--model", choices=["validation", "final"], default="validation")
    parser.add_argument("--input", type=Path, default=None,
                        help="CSV to score; default features_k3.csv, else dataset_k3.csv")
    parser.add_argument("--cohort-year", type=int, default=0,
                        help="Score cohorts from a given year; 0 = all")
    parser.add_argument("--unlabeled-only", action="store_true",
                        help="Only score cohorts without an observed M12 label")
    parser.add_argument("--min-cohort-size", type=int, default=0,
                        help="Minimum original cohort size (default: no filter)")
    args = parser.parse_args()
    if args.min_cohort_size < 0:
        parser.error("--min-cohort-size must be nonnegative")

    model, feature_cols = load_trained_model(args.model)
    ds, source = load_scoring_data(args.input)
    print(f"== Q5 predict | model={args.model} | source={source.name} ==")
    if source.name == "dataset_k3.csv":
        print("NOTE: dataset_k3.csv contains labeled cohorts only; "
              "use features_k3.csv for new/unlabeled cohorts.")
    if args.cohort_year:
        ds = ds[ds["cohort_month"].dt.year == args.cohort_year].copy()
    if args.min_cohort_size:
        if "cohort_size" not in ds.columns:
            raise ValueError("cohort_size is required for --min-cohort-size")
        ds = ds[pd.to_numeric(ds["cohort_size"], errors="coerce")
                >= args.min_cohort_size].copy()
    if args.unlabeled_only:
        ds = unlabeled_cohorts(ds)
    if ds.empty:
        print("No cohorts match the selected filters; no file written.")
        return

    scored = score_dataset(ds, model, feature_cols)
    TABLES.mkdir(parents=True, exist_ok=True)
    suffix = f"cohort{args.cohort_year}" if args.cohort_year else "all"
    if args.unlabeled_only:
        suffix += "_unlabeled"
    if args.min_cohort_size:
        suffix += f"_size{args.min_cohort_size}"
    output = TABLES / f"predictions_{suffix}_k3_{args.model}.csv"
    scored.to_csv(output, index=False)
    print(f"Scored {len(scored)} cohorts; saved: {output}")
    print(scored[[c for c in KEYS + ["retention_m3", "predicted_retention_12", "rank"]
                  if c in scored.columns]].head(10).to_string(index=False))
    print("== Q5 predict done ==")


if __name__ == "__main__":
    main()
