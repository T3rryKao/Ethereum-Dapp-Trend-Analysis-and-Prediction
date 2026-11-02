"""Find governance for every DApp in dapp_static: DefiLlama IDs first, then a Snapshot name search.

A searched space only counts when its website, Twitter or GitHub matches the DApp.
Raw JSON goes to data/raw, the source table to data/source_governance.csv.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = ROOT / "dapp_data_v2"
DATA = HERE / "data"
RAW = DATA / "raw"

HUB = "https://hub.snapshot.org/graphql"
PAUSE = 1.1
SPACE_FIELDS = "id name website twitter github domain proposalsCount followersCount flagged"

# Read by hand: spaces that copy a DApp's Twitter handle without being its governance.
WRONG_SPACES = {"samogonshik.eth"}
HAND_SPACES = {"stargate-v1": "stgdao.eth", "stargate-v2": "stgdao.eth"}


def snapshot(query: str, path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(5):
        try:
            response = requests.post(HUB, json={"query": query}, timeout=40, headers={"User-Agent": "uci-capstone-research"})
            time.sleep(PAUSE)
            body = response.json() if response.ok else {}
        except (requests.RequestException, ValueError):
            body = {}
        if "data" in body:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body["data"]))
            return body["data"]
        time.sleep(30 * (attempt + 1))
    return None


def get_space(space: str) -> dict:
    found = snapshot(f'{{ space(id: "{space}") {{ {SPACE_FIELDS} }} }}', RAW / "snapshot_space" / f"{quote(space, safe='')}.json")
    return (found or {}).get("space") or {}


def search_spaces(term: str) -> list[dict] | None:
    found = snapshot(f'{{ ranking(first: 20, where: {{search: "{term}"}}) {{ items {{ {SPACE_FIELDS} }} }} }}',
                     RAW / "snapshot_search" / f"{quote(term, safe='')}.json")
    return None if found is None else found["ranking"]["items"]


def first_proposal(space: str) -> int | None:
    found = snapshot(f'{{ proposals(first: 1, where: {{space: "{space}"}}, orderBy: "created", orderDirection: asc) {{ created }} }}',
                     RAW / "snapshot_first" / f"{quote(space, safe='')}.json")
    items = (found or {}).get("proposals") or []
    return items[0]["created"] if items else None


def domain(url) -> str:
    if not isinstance(url, str) or not url:
        return ""
    host = urlparse(url if "//" in url else f"https://{url}").netloc.lower()
    return ".".join(host.split(".")[-2:])


def clean(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower()) if text else ""


def identifiers(hit: dict, parent: dict, owner, url) -> dict:
    return dict(twitter={clean(x) for x in (hit.get("twitter"), parent.get("twitter")) if x},
                github={clean(x) for x in [owner, *(hit.get("github") or []), *(parent.get("github") or [])] if isinstance(x, str) and x},
                domains={d for d in (domain(url), domain(hit.get("url")), domain(parent.get("url"))) if d})


def matches(space: dict, ids: dict) -> str | None:
    if clean(space.get("twitter")) in ids["twitter"]:
        return "twitter"
    if clean(space.get("github")) in ids["github"]:
        return "github"
    if domain(space.get("website")) in ids["domains"] or domain(space.get("domain")) in ids["domains"]:
        return "website"
    return None


def governance_row(row, hit: dict, parent: dict, owner, url) -> dict:
    listed = list(dict.fromkeys((hit.get("governanceID") or []) + (parent.get("governanceID") or [])))
    onchain = [g for g in listed if g.startswith(("eip155:", "compound:"))]
    ids = identifiers(hit, parent, owner, url)
    best, how = None, None
    for g in listed:
        if g.startswith("snapsho"):
            space = get_space(g.split(":", 1)[1])
            if space.get("proposalsCount") and not space.get("flagged") and (best is None or space["proposalsCount"] > best["proposalsCount"]):
                best, how = space, "defillama"
    term = str(row.parent_protocol).replace("-", " ").strip()
    results = search_spaces(term)
    for space in results or []:
        evidence = matches(space, ids)
        if space.get("id") in WRONG_SPACES:
            continue
        if evidence and space.get("proposalsCount") and not space.get("flagged") and (best is None or space["proposalsCount"] > best["proposalsCount"]):
            best, how = space, f"search_{evidence}"
    if row.slug in HAND_SPACES:
        best, how = get_space(HAND_SPACES[row.slug]), "hand"
    return dict(slug=row.slug, in_llama=bool(hit), n_identifiers=sum(len(v) for v in ids.values()), search_ok=results is not None,
                llama_ids=";".join(listed), onchain=bool(onchain), space=(best or {}).get("id"), how=how,
                proposals=(best or {}).get("proposalsCount"), followers=(best or {}).get("followersCount"),
                first_proposal=first_proposal(best["id"]) if best else None)


def main() -> None:
    dapps = pd.read_csv(V2 / "dapp_static.csv")[["slug", "parent_protocol"]]
    index = {p.get("slug"): p for p in json.loads((RAW / "llama_protocols.json").read_text())}
    parents = {p["id"]: p for p in json.loads((RAW / "llama_lite.json").read_text())["parentProtocols"]}
    audits = pd.read_csv(DATA / "source_audits.csv").set_index("slug")
    owners = pd.read_csv(DATA / "source_github.csv").set_index("slug")["owner"]
    urls = pd.read_csv(V2 / "protocols.csv").set_index("slug")["url"]
    rows = []
    for row in dapps.itertuples():
        hit = index.get(row.slug) or {}
        if not hit and audits.loc[row.slug, "llama_match"] == "old_slug":
            path = RAW / "llama_protocol" / f"{row.slug}.json"
            hit = json.loads(path.read_text()) if path.exists() else {}
        parent = parents.get(hit.get("parentProtocol")) or {}
        rows.append(governance_row(row, hit, parent, owners.get(row.slug), urls.get(row.slug)))
        if len(rows) % 50 == 0:
            print(len(rows), flush=True)
    pd.DataFrame(rows).to_csv(DATA / "source_governance.csv", index=False)


if __name__ == "__main__":
    main()
