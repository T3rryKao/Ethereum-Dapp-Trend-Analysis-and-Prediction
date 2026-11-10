# Predicting 12-month wallet cohort retention from early behavior

**Data:** `dapp_data_v2/cohort_retention.csv` (activity through 2025-12)  
**Code:** `retention_model/labels.py`, `features.py`, `baseline.py`, `splits.py`, `train.py`, `predict.py`, `run_all.py`  
**Reproduce:** `python retention_model/run_all.py`  
---

## Research question

How well can we predict a cohort's 12-month wallet retention rate using its first three months of retention behavior?

A **cohort** groups wallets first observed using the same protocol in the same month within the available history (M0). M1–M3 retention measures the fraction of that original cohort active in each following month.

M12 measures longer-term engagement beyond the early window. This is a **regression task** with a target between 0 and 1: activity specifically at M12, including returns after inactivity, rather than uninterrupted participation.

---

## Label design

**Target:** `y_retention_12` = active wallets from the cohort at M12 / original M0 cohort size.

| Requirement | Treatment |
|-------------|-----------|
| Valid cohort | Positive original size and an observed M0 |
| Mature outcome | M12 must fall within the dataset's coverage |
| No activity at a covered M12 | Retention is zero; the cohort remains included |
| M12 beyond the dataset endpoint | Outcome is unknown; exclude from labeled data |

The source records actual activity, so an absent row in a completed, covered month is reconstructed as zero. **This assumes complete source coverage:** an unavailable source month must not be interpreted as confirmed inactivity.

---

## Features (M1–M3)

The model uses ten features derived from original cohort size and early retention:

- **Levels:** `log_cohort_size` is log(1 + M0 wallet count); `mean_retention` and `max_retention` summarize M1–M3 rates.
- **Shape:** `retention_slope` measures the linear trend; `retention_volatility` is the standard deviation of successive rate changes; `last_to_max_ratio` compares M3 with the early maximum.
- **Early activity:** `retention_m1`, `retention_m2`, `retention_m3`, and `observed_months`.

All rates use M0 size as denominator. Covered missing activity is zero-filled; the last/max ratio is zero for an all-zero history. `observed_months` is three under complete coverage. The summaries transform the same three observations rather than adding independent measurements.

---

## Models and evaluation design

The primary experiment uses **chronological Train/Test splitting**, fixed model configurations, and equal weight for each cohort.

| Partition | Cohort period and eligibility | Cohorts |
|-----------|-------------------------------|--------:|
| Training | January 2020–March 2023; M12 labels complete by March 2024 | **5,364** |
| Testing | January–December 2024 | **3,425** |

The earliest test M3 is April 2024, after all fitting labels mature in March 2024. April–December 2023 cohorts are excluded from fitting because their M12 outcomes mature after the cutoff. Models remain fixed throughout evaluation.

This default experiment uses no random split, validation folds, or weighted training. **Ridge is predefined for scoring**, not selected automatically from the test ranking. Hyperparameters are fixed, stochastic estimators use seed 42, and all predictions are clipped to [0, 1].

**Metrics:** MAE measures average absolute error; RMSE emphasizes larger errors; R² compares squared error with predicting the test-target mean for every cohort. R² can be negative. Errors are fractions: MAE 0.03 means three percentage points. The reported evaluation is cohort-weighted, not wallet-weighted.

---

## Baseline models

| Baseline | Prediction approach |
|----------|---------------------|
| **Historical Median** | Predicts the training-set median M12 retention for every cohort. It provides a reference without individual early-behavior predictors. |
| **M3 Persistence** | Predicts that M12 retention equals M3 retention. It tests whether simply carrying the latest observed rate forward is sufficient. |
| **Univariate Linear Regression** | Fits a training-set linear relationship between M3 and M12 retention. It tests how much a single early retention measurement can explain. |

For missing M3 values, the persistence rule uses a training-median fallback and the linear model uses training-fitted median imputation.

---

## Machine learning models

| Model | Prediction approach |
|-------|---------------------|
| **Ridge Regression** | Combines all early features in a regularized linear model, limiting unstable coefficients when predictors overlap. The penalty is fixed at alpha = 10. |
| **Decision Tree** | Uses feature thresholds to capture nonlinear relationships. Maximum depth is 5, with at least 20 training samples per leaf. |
| **Random Forest** | Averages 200 trees to capture nonlinear patterns and interactions. Maximum depth is 10 and minimum leaf size is 5. |
| **Histogram Gradient Boosting** | Builds trees sequentially to correct earlier errors. Settings are up to 200 iterations, learning rate 0.05, at most 15 leaves, minimum leaf size 20, and L2 regularization 1. |

No hyperparameters are tuned using the 2024 test outcomes.

---

## Test results (`y_retention_12`)

All seven models are evaluated on the same **3,425 test cohorts**. The table is sorted by MAE. Values were independently recomputed from this run's [saved test predictions](runs_train_test/20261110T005216_616505Z/test_predictions.csv) and checked against [the evaluation CSV](runs_train_test/20261110T005216_616505Z/test_model_comparison.csv).

| Model | Type | MAE | RMSE | R² |
|-------|------|----:|-----:|---:|
| Univariate Linear Regression | Baseline | **0.034985** | 0.093428 | 0.168435 |
| Ridge Regression | Machine learning | 0.035354 | **0.090930** | **0.212316** |
| Historical Median | Baseline | 0.035500 | 0.105464 | −0.059621 |
| Random Forest | Machine learning | 0.036084 | 0.093402 | 0.168907 |
| Histogram Gradient Boosting | Machine learning | 0.036384 | 0.095654 | 0.128347 |
| Decision Tree | Machine learning | 0.036520 | 0.092635 | 0.182488 |
| M3 Persistence | Baseline | 0.065519 | 0.151594 | −1.189297 |

**Takeaways:**

1. **M3 Linear Regression has the lowest MAE:** approximately 3.50 percentage points. M3 alone is a strong baseline.
2. **Ridge has the best RMSE and R²:** 9.09 percentage points and 0.212. This means **21.2% less squared error than the constant test-mean reference**, not 21.2% prediction accuracy; substantial unexplained variation remains.
3. Tree models do not improve aggregate errors here. Greater flexibility cannot necessarily recover information absent from the early window; this comparison does not isolate why they underperform.

---

## Error analysis: cohort size

Small cohorts contribute disproportionately to squared error. **One-wallet cohorts represent 379 test rows (11.07%) but account for 53.02% of Ridge's total squared error**, recalculated from the saved predictions. A one-wallet cohort can realize only 0% or 100% retention, so one unexpected return can create a large fraction error.

Larger cohorts have lower absolute errors in this sample. Ridge's MAE/RMSE for cohorts with at least 30 wallets are **0.016913/0.030119**, compared with **0.078508/0.199030** for one-wallet cohorts.

Excluding one-wallet cohorts reduces Ridge RMSE from **0.090930 to 0.066092**, but barely changes R²: **0.212316 → 0.211799**. The remaining targets also have less variance. These are sensitivity diagnostics, not a reason to choose a minimum size from test performance or replace the primary all-cohort evaluation.

---

## Scoring cohorts

From the repository root, using the project Python environment:

```bash
python retention_model/run_all.py
```

The workflow generates labels and audits, builds features, fits all seven predictors, evaluates the chronological test set, and reloads the new run's predefined Ridge artifact for scoring. **It does not discover or execute unit tests.**

Each run uses a fresh directory under `retention_model/runs_train_test/<UTC timestamp>/`, refusing to overwrite an existing destination. It saves:

- `labels_m12.csv`, `features_k3.csv`, `dataset_k3.csv`, and label audit tables — generated inputs and cohort accounting.
- `test_model_comparison.csv` — seven-model metrics and training/test counts.
- `test_predictions.csv` — actual outcomes and every model's test predictions.
- `predictions_all_k3.csv` — **13,924** M3-ready cohorts, including immature M12 outcomes.
- `artifacts/` — seven fitted model bundles plus `q5_train_test_model.joblib`, the predefined Ridge alias.
- `run_summary.txt` and `run_metadata.json` — execution log, input hash, versions, seed, counts, and prediction policy.



## File map

```text
retention_model/
  labels.py         # mature M12 labels, including covered zero activity
  features.py       # M0 size and M1–M3 retention summaries
  baseline.py       # median, persistence, and linear baselines
  splits.py         # chronological split and label-availability checks
  train.py          # fixed-model fitting and evaluation
  predict.py        # artifact loading and cohort scoring
  run_all.py        # full workflow; no automatic unit-test execution
  model_report.md   # this report
```
