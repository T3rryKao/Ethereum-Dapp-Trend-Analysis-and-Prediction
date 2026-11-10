"""Chronological Train/Test only, with fully matured fitting labels."""
import pandas as pd

FINAL_CUTOFF = pd.Period("2024-03", freq="M")
SCHEMA_VERSION = "q5_sparse_zero_v2"  # Feature semantics are unchanged.
WORKFLOW_VERSION = "q5_train_test_v1"
WEIGHTED_WORKFLOW_VERSION = "q5_train_test_weights_v1"


def cohort_months(df):
    dates = pd.to_datetime(df["cohort_month"], errors="raise")
    if dates.isna().any():
        raise ValueError("Missing cohort dates")
    return dates.dt.to_period("M")


def available_training(df, cutoff):
    """A label becomes usable after its entire M12 calendar month completes."""
    return df.loc[cohort_months(df) + 12 <= pd.Period(cutoff, freq="M")].copy()


def time_split(df):
    """Return eligible fitting pool and reserved 2024 cohorts only."""
    df = df.copy()
    df["cohort_month"] = pd.to_datetime(df["cohort_month"], errors="raise")
    months = cohort_months(df)
    pool = available_training(df, FINAL_CUTOFF)
    test = df.loc[months.dt.year == 2024].copy()
    if pool.empty or test.empty:
        raise ValueError("Empty training pool or reserved test set")
    if FINAL_CUTOFF >= (cohort_months(test) + 3).min():
        raise ValueError("Final cutoff overlaps test feature availability")
    return pool, test


def weighting_validation_split(pool):
    """One chronological validation block within the final eligible pool."""
    months = cohort_months(pool)
    if (months + 12 > FINAL_CUTOFF).any():
        raise ValueError("Validation pool contains labels unavailable at final fitting")
    cutoff = pd.Period("2022-06", freq="M")
    training = available_training(pool, cutoff)
    validation = pool.loc[months.between(pd.Period("2022-04"), pd.Period("2023-03"))].copy()
    if training.empty or validation.empty:
        raise ValueError("Empty weighting training/validation partition")
    if (cohort_months(training) + 12).max() >= (cohort_months(validation) + 3).min():
        raise ValueError("Training labels overlap validation feature availability")
    if (cohort_months(validation) + 12).max() > FINAL_CUTOFF:
        raise ValueError("Validation outcomes unavailable at final fitting")
    return training, validation
