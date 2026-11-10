"""Q5: M1-M3 cohort features for M12 retention prediction.

Absent activity rows in covered months mean zero retention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from labels import DATA_END, build_labels, load_data
from splits import SCHEMA_VERSION

OUTPUT_DIR = Path(__file__).resolve().parent / "tables"
KEYS = ["protocol", "cohort_month"]

LEVEL_FEATURES = [
    "log_cohort_size",
    "mean_retention",
    "max_retention",
]

SHAPE_FEATURES = [
    "retention_slope",
    "retention_volatility",
    "last_to_max_ratio",
]

QUALITY_FEATURES = [
    "retention_m1",
    "retention_m2",
    "retention_m3",
    "observed_months",
]

FEATURE_COLS = LEVEL_FEATURES + SHAPE_FEATURES + QUALITY_FEATURES


def early_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Build Level, Shape, and Quality features from M1-M3 only."""
    rows = []

    for (protocol, cohort_month), g in cohort.groupby(KEYS):
        g = g.sort_values("months_since")
        m0 = g[g["months_since"] == 0]
        if m0.empty:
            continue

        cohort_size = float(m0["cohort_size"].iloc[0])
        if not np.isfinite(cohort_size) or cohort_size <= 0:
            continue

        # Keep only cohorts old enough to have an M3 observation.
        if pd.Timestamp(cohort_month).to_period("M") + 3 > DATA_END:
            continue

        early = g[g["months_since"].between(1, 3)]
        monthly = dict(zip(early["months_since"].astype(int),
                           pd.to_numeric(early["retention_rate"], errors="coerce")))
        values = np.array([monthly.get(i, 0.0) for i in (1, 2, 3)], dtype=float)
        complete = np.isfinite(values).all()

        # Covered absent activity rows have already been reconstructed as zeros.
        has_observation = np.isfinite(values).any()

        mean_ret = (
            float(np.nanmean(values)) if has_observation else np.nan
        )

        max_ret = (
            float(np.nanmax(values)) if has_observation else np.nan
        )

        # Shape: calculate trends only for complete M1-M3 histories.
        slope = float(np.polyfit([1, 2, 3], values, 1)[0]) if complete else np.nan
        volatility = float(np.std(np.diff(values))) if complete else np.nan
        last_to_max = (
            float(values[2] / max_ret)
            if complete and max_ret > 0 else (0.0 if complete else np.nan)
        )

        rows.append({
            "protocol": protocol,
            "cohort_month": cohort_month,
            "cohort_size": cohort_size,
            # Level
            "log_cohort_size": float(np.log1p(cohort_size)),
            "mean_retention": mean_ret,
            "max_retention": max_ret,
            # Shape
            "retention_slope": slope,
            "retention_volatility": volatility,
            "last_to_max_ratio": last_to_max,
            # Quality
            "retention_m1": values[0],
            "retention_m2": values[1],
            "retention_m3": values[2],
            "observed_months": int(np.isfinite(values).sum()),
        })

    result = pd.DataFrame(rows, columns=KEYS + ["cohort_size"] + FEATURE_COLS)
    result["schema_version"] = SCHEMA_VERSION
    return result


def build_dataset(cohort: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join M1-M3 features to observed M12 labels by cohort identity."""
    if cohort is None:
        cohort = load_data()

    feats = early_features(cohort)
    labs = build_labels(cohort)[KEYS + ["y_retained_wallets_12", "y_retention_12",
                                             "label_status", "label_available_month"]]
    return feats.merge(labs, on=KEYS, how="inner", validate="one_to_one")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_data()

    print("== Q5 M1-M3 features ==")
    features = early_features(cohort)
    features.to_csv(OUTPUT_DIR / "features_k3.csv", index=False)

    dataset = build_dataset(cohort)
    output_path = OUTPUT_DIR / "dataset_k3.csv"
    dataset.to_csv(output_path, index=False)

    print("\nMissing feature values (labeled cohorts):")
    print(dataset[FEATURE_COLS].isna().sum().to_string())

    print(f"Feature cohorts: {len(features)}")
    print(f"Labeled cohorts: {len(dataset)}")
    print(f"Unique protocols: {dataset['protocol'].nunique()}")
    print(f"Mean M12 retention: {dataset['y_retention_12'].mean():.4f}")
    print(f"Saved to: {output_path}")
    print("== Q5 features done ==")


if __name__ == "__main__":
    main()
