"""Build slug + three flags from the source tables and report coverage.

1 = confirmed, 0 = looked up and confirmed absent, blank = unknown. All flags describe today.
has_governance 0 means no governance was found after searching with the DApp's own identifiers.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
STATIC_FLAGS = ["is_audited", "is_open_source", "is_verified"]
FLAGS = [*STATIC_FLAGS, "has_governance"]
TRAIN_MAX_YEAR = 2022
K = 3
WINDOWS = range(1, 10)
MONTHS = range(13)


def write_cohort() -> pd.DataFrame:
    """The study DApps: the model's eligible frame at k = 3."""
    sys.path.insert(0, str(ROOT / "model"))
    warnings.filterwarnings("ignore")
    import train_eval as te
    from features import annotate_lifecycle, build_dataset, load_category_month, load_lifecycle, load_panel, load_static

    retention = pd.read_csv(ROOT / "dapp_data_v2" / "cohort_retention.csv")
    retention["cohort_month"] = pd.to_datetime(retention["cohort_month"], errors="coerce")
    data = build_dataset(K, panel=load_panel(), life=annotate_lifecycle(load_lifecycle(), load_static()),
                         cohort=retention, cat_month=load_category_month())
    names = pd.read_csv(ROOT / "dapp_data_v2" / "protocols.csv").set_index("slug")["name"]
    eligible = te.eligible_frame(data)
    eligible = eligible.assign(name=eligible["slug"].map(names).fillna(eligible["slug"]))
    cohort = eligible[["slug", "name", "parent_protocol", "launch_month", "launch_year"]].copy()
    cohort["split"] = np.where(cohort["launch_year"] <= TRAIN_MAX_YEAR, "train", "test")
    cohort.to_csv(DATA / "cohort.csv", index=False)
    return cohort


def build_flags(dapps: pd.Series, audits: pd.DataFrame, github: pd.DataFrame, verified: pd.DataFrame, governance: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"slug": dapps})
    audits = audits.set_index("slug")
    status = pd.to_numeric(audits["audits"], errors="coerce")
    flag = pd.Series(pd.NA, index=audits.index, dtype="Int64")
    flag[status == 0] = 0
    flag[(status > 0) | (audits["n_audit_links"] > 0)] = 1
    out["is_audited"] = out["slug"].map(flag).astype("Int64")

    repos = github.set_index("slug")["public_repos"]
    out["is_open_source"] = out["slug"].map(repos.gt(0).astype("Int64").where(repos.notna())).astype("Int64")

    confirmed = verified["blockscout_verified"] | verified["sourcify_verified"]
    seen = verified["is_contract"].eq(True) | verified["sourcify_verified"]
    any_verified = confirmed.groupby(verified["slug"]).any().astype("Int64")
    any_seen = seen.groupby(verified["slug"]).any()
    out["is_verified"] = out["slug"].map(any_verified.where(any_seen)).astype("Int64")

    governance = governance.set_index("slug")
    has = (governance["space"].notna() | governance["onchain"]).astype("Int64")
    searched = governance["search_ok"] & (governance["n_identifiers"] > 0)
    out["has_governance"] = out["slug"].map(has.where(has.eq(1) | searched)).astype("Int64")
    return out


def month_start(values: pd.Series) -> pd.Series:
    """Any date in a month becomes that month's first day, as the model's month 0 does."""
    return pd.to_datetime(values, utc=True).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()


def governance_monthly(cohort: pd.DataFrame, today: pd.DataFrame, governance: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    """One row per study DApp and month 0-12. n_proposals counts all history, n_proposals_window starts at month 0."""
    created = pd.to_datetime(proposals["created"], utc=True).dt.tz_localize(None)
    dates = {space: np.sort(group.values) for space, group in created.groupby(proposals["space"])}
    table = cohort[["slug", "split", "launch_month"]].merge(today[["slug", "has_governance"]], on="slug").merge(governance[["slug", "space"]], on="slug")
    rows = []
    for row in table.itertuples():
        launch = month_start(pd.Series([row.launch_month])).iloc[0]
        known = isinstance(row.space, str) or (pd.notna(row.has_governance) and row.has_governance == 0)
        when = dates.get(row.space, np.array([], dtype="datetime64[ns]"))
        before = np.searchsorted(when, launch.to_datetime64())
        previous = before
        for month in MONTHS:
            cutoff = np.searchsorted(when, (launch + pd.DateOffset(months=month + 1)).to_datetime64())
            rows.append(dict(slug=row.slug, split=row.split, month_index=month, month=(launch + pd.DateOffset(months=month)).strftime("%Y-%m-%d"),
                             new_proposals=cutoff - previous if known else pd.NA, n_proposals=cutoff if known else pd.NA,
                             n_proposals_window=cutoff - before if known else pd.NA, has_governance=int(cutoff > 0) if known else pd.NA))
            previous = cutoff
    counts = ["new_proposals", "n_proposals", "n_proposals_window", "has_governance"]
    return pd.DataFrame(rows).astype({c: "Int64" for c in counts})


def by_k(monthly: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Window k ends at month k-1, so each k takes that month's governance."""
    table = monthly[monthly["month_index"].isin([k - 1 for k in WINDOWS])].assign(k=lambda d: d["month_index"] + 1)
    table = table.merge(flags, on="slug", how="left")
    return table[["slug", "split", "k", *STATIC_FLAGS, "has_governance", "n_proposals", "n_proposals_window"]]


def coverage(flags: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    groups = {"dapp_static (676)": flags,
              "study DApps": flags[flags["slug"].isin(cohort["slug"])],
              "train 2020-22": flags[flags["slug"].isin(cohort.loc[cohort["launch_year"] <= TRAIN_MAX_YEAR, "slug"])],
              "test 2023-24": flags[flags["slug"].isin(cohort.loc[cohort["launch_year"] > TRAIN_MAX_YEAR, "slug"])]}
    rows = []
    for name, group in groups.items():
        for flag in FLAGS:
            rows.append(dict(sample=name, n=len(group), flag=flag, yes=int((group[flag] == 1).sum()), no=int((group[flag] == 0).sum()),
                             unknown=int(group[flag].isna().sum()), found_pct=round(group[flag].notna().mean() * 100, 1)))
    return pd.DataFrame(rows)


def main() -> None:
    dapps = pd.read_csv(HERE.parents[1] / "dapp_data_v2" / "dapp_static.csv")["slug"]
    sources = [pd.read_csv(DATA / f"source_{name}.csv") for name in ("audits", "github", "verified", "governance")]
    flags = build_flags(dapps, *sources)
    flags[["slug", *STATIC_FLAGS]].to_csv(DATA / "dapp_offchain_flags.csv", index=False)
    cohort = write_cohort()
    table = coverage(flags, cohort)
    table.to_csv(DATA / "flag_coverage.csv", index=False)
    print(table.to_string(index=False))

    monthly = governance_monthly(cohort, flags, sources[3], pd.read_csv(DATA / "source_proposals.csv"))
    monthly.to_csv(DATA / "governance_monthly.csv", index=False)
    final = by_k(monthly, flags[["slug", *STATIC_FLAGS]])
    final.to_csv(DATA / "dapp_offchain_by_k.csv", index=False)
    print(final.groupby(["k", "split"])["has_governance"].sum().unstack().to_string())


if __name__ == "__main__":
    main()
