"""Download the date of every proposal in the Snapshot spaces matched to a DApp."""
from __future__ import annotations

from urllib.parse import quote

import pandas as pd

from fetch_governance import DATA, RAW, snapshot

PAGE = 1000


def proposals(space: str) -> list[dict]:
    rows, after, page = [], 0, 0
    while True:
        query = (f'{{ proposals(first: {PAGE}, where: {{space: "{space}", created_gt: {after}}}, '
                 f'orderBy: "created", orderDirection: asc) {{ id created }} }}')
        found = snapshot(query, RAW / "snapshot_proposals" / f"{quote(space, safe='')}_{page}.json")
        items = (found or {}).get("proposals") or []
        rows += [dict(space=space, proposal=x["id"], created=x["created"]) for x in items]
        if len(items) < PAGE:
            return rows
        after, page = items[-1]["created"], page + 1


def main() -> None:
    spaces = pd.read_csv(DATA / "source_governance.csv")["space"].dropna().unique()
    rows = [row for space in spaces for row in proposals(space)]
    table = pd.DataFrame(rows)
    table["created"] = pd.to_datetime(table["created"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    table.to_csv(DATA / "source_proposals.csv", index=False)
    print(len(spaces), "spaces,", len(table), "proposals")


if __name__ == "__main__":
    main()
