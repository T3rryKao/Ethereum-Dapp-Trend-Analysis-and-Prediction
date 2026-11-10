
"""Build 12-month cohort retention labels for Q5.

Target: y_retention_12
Each observation represents one (protocol, cohort_month).
"""

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dapp_data_v2" / "cohort_retention.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "tables"

DATA_END = pd.Period("2025-12", freq="M")


def load_data() -> pd.DataFrame:
    """Load cohort retention data and validate its structure."""

    df = pd.read_csv(DATA_PATH)

    required = [
        "protocol", "cohort_month", "months_since",
        "retained", "cohort_size", "retention_rate"
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["cohort_month"] = pd.to_datetime(df["cohort_month"])
    if df[["protocol", "cohort_month"]].isna().any().any():
        raise ValueError("Missing cohort identifiers")
    if df["protocol"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty protocol identifier")
    if not df["cohort_month"].eq(df["cohort_month"].dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("cohort_month must be the first day of its month")
    df["months_since"] = pd.to_numeric(
        df["months_since"], errors="raise"
    )

    for col in ["retained", "cohort_size", "retention_rate"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[["months_since", "retained", "cohort_size", "retention_rate"]].isna().any().any():
        raise ValueError("Explicit missing values are coverage gaps, not zero activity")
    if not np.isfinite(df[["months_since", "retained", "cohort_size", "retention_rate"]]).all().all():
        raise ValueError("Non-finite cohort observations")
    if (df["months_since"] < 0).any() or (df["months_since"] % 1 != 0).any():
        raise ValueError("months_since must be a nonnegative integer")
    if (df["cohort_size"] <= 0).any() or (df["retained"] < 0).any() or (df["retained"] > df["cohort_size"]).any():
        raise ValueError("Invalid wallet counts")
    if (df[["retained", "cohort_size"]] % 1 != 0).any().any():
        raise ValueError("Wallet counts must be integers")
    m0 = df["months_since"].eq(0)
    if not df.loc[m0, "retained"].eq(df.loc[m0, "cohort_size"]).all():
        raise ValueError("M0 retained count must equal original cohort size")
    if not np.allclose(df["retention_rate"], df["retained"] / df["cohort_size"]):
        raise ValueError("Retention rates do not match wallet counts")
    keys = ["protocol", "cohort_month", "months_since"]

    if df.duplicated(keys).any():
        raise ValueError("Duplicate cohort-month observations.")

    # Ensure cohort_size is consistent within each cohort
    size_counts = df.groupby(
        ["protocol", "cohort_month"]
    )["cohort_size"].nunique()

    if (size_counts > 1).any():
        raise ValueError("Inconsistent cohort_size.")

    return df


def clean_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """Select valid cohorts with sufficient M12 observation time."""

    keys = ["protocol", "cohort_month"]

    # Identify each cohort and its original size
    cohorts = (
        df.groupby(keys, as_index=False)
          .agg(
              cohort_size=("cohort_size", "first"),
              min_month=("months_since", "min")
          )
    )

    # Keep cohorts with positive size and an observed M0
    cohorts = cohorts[
        (cohorts["cohort_size"] > 0) &
        (cohorts["min_month"] == 0)
    ].copy()

    # Calculate the calendar month corresponding to M12
    cohorts["target_period"] = (
        cohorts["cohort_month"].dt.to_period("M") + 12
    )

    # Exclude cohorts whose M12 is beyond the dataset end
    cohorts = cohorts[
        cohorts["target_period"] <= DATA_END
    ].copy()

    return cohorts


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Build mature labels; absent activity rows are zeros in this sparse export.

    Assumes complete source coverage through DATA_END. Do not use for exports
    with missing partitions without an explicit coverage mask.
    """

    keys = ["protocol", "cohort_month"]

    cohorts = clean_cohorts(df)

    m12 = df.loc[
        df["months_since"] == 12,
        keys + ["retained", "retention_rate"]
    ].copy()

    m12 = m12.rename(columns={
        "retained": "y_retained_wallets_12",
        "retention_rate": "y_retention_12"
    })

    labels = cohorts.merge(
        m12,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True
    )

    present = labels["_merge"].eq("both")
    if labels.loc[present, ["y_retained_wallets_12", "y_retention_12"]].isna().any().any():
        raise ValueError("Explicit missing M12 observations are unknown, not zero activity")
    labels["label_status"] = np.where(
        present, "observed", "zero_activity"
    )
    labels[["y_retained_wallets_12", "y_retention_12"]] = labels[
        ["y_retained_wallets_12", "y_retention_12"]
    ].fillna(0.0)
    labels["label_available_month"] = labels["target_period"].astype(str)

    labels = labels[
        [
            "protocol",
            "cohort_month",
            "cohort_size",
            "y_retained_wallets_12",
            "y_retention_12",
            "label_status",
            "label_available_month"
        ]
    ].copy()

    return labels


def audit_tables(df: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Account for every source cohort, including immature and invalid cohorts."""
    keys = ["protocol", "cohort_month"]
    cohorts = df.groupby(keys, as_index=False).agg(
        cohort_size=("cohort_size", "first"), min_month=("months_since", "min")
    )
    audit = cohorts.merge(labels[keys + ["label_status", "y_retention_12"]],
                          on=keys, how="left", validate="one_to_one")
    invalid = (audit.cohort_size <= 0) | (audit.min_month != 0)
    immature = audit.cohort_month.dt.to_period("M") + 12 > DATA_END
    audit["label_status"] = audit.label_status.fillna("unaccounted")
    audit.loc[immature, "label_status"] = "immature"
    audit.loc[invalid, "label_status"] = "invalid_cohort"
    if audit.label_status.eq("unaccounted").any():
        raise ValueError("Eligible cohorts missing from label audit")
    summary = audit.groupby("label_status", as_index=False).agg(
        n_cohorts=("protocol", "size"), n_protocols=("protocol", "nunique"),
        mean_retention_12=("y_retention_12", "mean")
    )
    return audit, summary


def main() -> None:
    """Build and save the final M12 label dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    labels = build_labels(df)

    output_path = OUTPUT_DIR / "labels_m12.csv"
    labels.to_csv(output_path, index=False)

    print("== Q5 M12 Retention Labels ==")
    print(f"Training cohorts: {len(labels)}")
    print(f"Unique protocols: {labels['protocol'].nunique()}")
    print(f"Mean M12 retention: {labels['y_retention_12'].mean():.4f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
