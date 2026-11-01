"""Fetch three off-chain flags for every DApp in dapp_data_v2/dapp_static.csv.

Raw JSON goes to data/raw and one source table per flag to data/. Calls are cached,
so a rerun only retries what failed. parse_flags.py builds the final table.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = ROOT / "dapp_data_v2"
DATA = HERE / "data"
RAW = DATA / "raw"

MAX_CONTRACTS = 30
ROLE_ORDER = {"factory": 0, "router": 1, "app_token": 2, "app": 3, "pool": 4}
CONTRACT_LANGUAGES = {"Solidity", "Vyper", "Yul", "Cairo", "Move"}
GITHUB_LINK = re.compile(r"github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38})(?=[/\"'\s<)])")
NOT_OWNERS = {"sponsors", "features", "about", "orgs", "login", "topics", "marketplace", "solidity", "ethereum", "ethereum-optimism", "openzeppelin"}

# Same project under another DefiLlama slug.
LLAMA_ALIASES = {"pendle": "pendle-v2", "foundation-nft": "foundation"}

# Checked to exist on GitHub. DefiLlama lists no org for these DApps.
HAND_ORGS = {
    "0x": "0xProject", "base-bridge": "base-org", "blast-bridge": "blast-io", "beta-finance-v1": "beta-finance",
    "fluid-lending": "Instadapp", "foundation-nft": "f8n", "olympus-dao": "OlympusDAO", "opensea-seaport": "ProjectOpenSea",
    "opensea-v2": "ProjectOpenSea", "optimism-bridge": "ethereum-optimism", "paraspace-lending-v1": "para-space",
    "pendle": "pendle-finance", "smardex-amm": "SmarDex-Dev", "starknet-bridge": "starkware-libs", "x2y2": "X2Y2-io",
    "zksync-era-txbridge": "matter-labs", "zircuit": "zircuit-labs", "unstoppable-domains": "unstoppabledomains",
    "loot": "lootproject", "sophon-farm": "sophon-org", "termfinance-vaults": "term-finance", "ora-protocol": "ora-io",
    "eigenpie": "magpiexyz", "art-blocks": "ArtBlocks", "instadapp": "Instadapp", "sorare": "sorare",
    "paraswap": "VeloraDEX", "velora": "VeloraDEX", "sky-lending": "sky-ecosystem", "derive-v3-options": "derivexyz",
}

# Read by hand: auditors, partners or a same-named project, not the DApp itself.
WRONG_OWNERS = {("zerobase-cedefi", "peckshield"), ("kuma-protocol", "code-423n4"), ("vector-reserve", "JacoboLansac"),
                ("alphagrowth", "euler-xyz"), ("parallel", "parallel-finance"), ("parallel", "parallel-protocol"), ("stonkbrokers-desks", "Clutch-L4bs")}


def cached_json(url: str, path: Path, pause: float = .2) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    try:
        response = requests.get(url, timeout=60, headers={"User-Agent": "uci-capstone-research"})
    except requests.RequestException:
        return {"http_error": "connection"}
    time.sleep(pause)
    result = response.json() if response.ok else {"http_error": response.status_code}
    if response.status_code != 429 and response.status_code < 500:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    return result


def llama_table(dapps: pd.DataFrame) -> pd.DataFrame:
    listed = cached_json("https://api.llama.fi/protocols", RAW / "llama_protocols.json")
    lite = cached_json("https://api.llama.fi/lite/protocols2?b=2", RAW / "llama_lite.json")
    index = {p.get("slug"): p for p in listed}
    parents = {p["id"]: p for p in lite["parentProtocols"]}
    rows = []
    for slug in dapps["slug"]:
        name = LLAMA_ALIASES.get(slug, slug)
        hit, match = index.get(name), "listed"
        if hit is None:
            hit, match = cached_json(f"https://api.llama.fi/protocol/{name}", RAW / "llama_protocol" / f"{name}.json"), "old_slug"
            if "http_error" in hit:
                hit, match = {}, "none"
        parent = parents.get(hit.get("parentProtocol")) or {}
        rows.append(dict(slug=slug, llama_match=match, llama_name=hit.get("name"), audits=hit.get("audits"), n_audit_links=len(hit.get("audit_links") or []),
                         llama_orgs=";".join(dict.fromkeys((hit.get("github") or []) + (parent.get("github") or []))), llama_url=hit.get("url")))
    return pd.DataFrame(rows)


def github_user(owner: str) -> dict:
    path = RAW / "github_user" / f"{owner.lower()}.json"
    if path.exists():
        return json.loads(path.read_text())
    done = subprocess.run(["gh", "api", f"users/{owner}"], capture_output=True, text=True)
    result = json.loads(done.stdout) if done.returncode == 0 else {}
    if done.returncode == 0 or "404" in done.stderr:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    return result


def has_contract_code(owner: str) -> bool:
    path = RAW / "github_repos" / f"{owner.lower()}.json"
    if not path.exists():
        done = subprocess.run(["gh", "api", f"users/{owner}/repos?per_page=100&sort=pushed"], capture_output=True, text=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(done.stdout if done.returncode == 0 else "[]")
    return any(r.get("language") in CONTRACT_LANGUAGES and not r.get("fork") for r in json.loads(path.read_text()))


def website_owners(slug: str, url: str) -> list[str]:
    path = RAW / "website_github" / f"{slug}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        page = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"}).text[:2_000_000]
        owners = [o for o in dict.fromkeys(GITHUB_LINK.findall(page)) if o.lower() not in NOT_OWNERS]
    except requests.RequestException:
        owners = []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(owners))
    return owners


def find_github(row) -> dict:
    found = dict(slug=row.slug, owner=None, how=None, public_repos=None)
    candidates = [(o, "defillama") for o in (row.llama_orgs or "").split(";") if o]
    if row.slug in HAND_ORGS:
        candidates.append((HAND_ORGS[row.slug], "hand"))
    if isinstance(row.url, str):
        candidates += [(o, "website") for o in website_owners(row.slug, row.url)]
    parent = str(row.parent_protocol)
    base = parent.replace("-", "")
    guesses = [parent, base, f"{base}-finance", f"{base}-protocol", f"{base}finance", f"{base}protocol", f"{base}-labs", f"{base}labs"]
    candidates += [(o, "login_match") for o in dict.fromkeys(guesses)]
    for owner, how in candidates:
        if (row.slug, owner) in WRONG_OWNERS or (row.slug, owner.lower()) in {(a, b.lower()) for a, b in WRONG_OWNERS}:
            continue
        user = github_user(owner)
        if not user:
            continue
        if how == "login_match" and not (user.get("public_repos") and has_contract_code(user["login"])):
            continue
        repos = user.get("public_repos", 0)
        if found["owner"] is None:
            found.update(owner=user["login"], how=how, public_repos=repos)
        if repos:
            found.update(owner=user["login"], how=how, public_repos=repos)
            break
    return found


def contract_status(contract: str) -> dict:
    blockscout = cached_json(f"https://eth.blockscout.com/api/v2/addresses/{contract}", RAW / "blockscout" / f"{contract}.json", .15)
    sourcify = cached_json(f"https://sourcify.dev/server/v2/contract/1/{contract}", RAW / "sourcify" / f"{contract}.json", .15)
    return dict(is_contract=blockscout.get("is_contract"), blockscout_verified=bool(blockscout.get("is_verified")),
                blockscout_error=blockscout.get("http_error"), sourcify_verified=sourcify.get("match") is not None,
                sourcify_error=sourcify.get("http_error"))


def check_dapp(group: pd.DataFrame) -> list[dict]:
    rows = []
    for r in group.itertuples():
        status = contract_status(r.contract)
        rows.append(dict(slug=r.protocol, contract=r.contract, role=r.role, deployed_date=r.deployed_date, **status))
        if status["blockscout_verified"] or status["sourcify_verified"]:
            break
    return rows


def verified_table(dapps: pd.DataFrame) -> pd.DataFrame:
    contracts = pd.read_csv(V2 / "app_contracts.csv", usecols=["contract", "protocol", "role", "is_token", "deployed_date"])
    contracts = contracts[contracts["protocol"].isin(dapps["slug"])]
    contracts["rank"] = contracts["role"].map(ROLE_ORDER).fillna(5)
    contracts = contracts.sort_values(["protocol", "rank", "deployed_date"]).groupby("protocol").head(MAX_CONTRACTS)
    groups = [g for _, g in contracts.groupby("protocol")]
    with ThreadPoolExecutor(2) as pool:
        rows = [row for result in pool.map(check_dapp, groups) for row in result]
    return pd.DataFrame(rows)


def main() -> None:
    dapps = pd.read_csv(V2 / "dapp_static.csv")[["slug", "name", "parent_protocol"]]
    print(len(dapps), "DApps", flush=True)

    llama = llama_table(dapps)
    llama.to_csv(DATA / "source_audits.csv", index=False)
    print(llama["llama_match"].value_counts().to_dict(), flush=True)

    urls = pd.read_csv(V2 / "protocols.csv").set_index("slug")["url"]
    rows = dapps.merge(llama, on="slug")
    rows["url"] = rows["slug"].map(urls).fillna(rows["llama_url"])
    rows["llama_orgs"] = rows["llama_orgs"].fillna("")
    with ThreadPoolExecutor(6) as pool:
        github = pd.DataFrame(list(pool.map(find_github, rows.itertuples())))
    github.to_csv(DATA / "source_github.csv", index=False)
    print(github["how"].fillna("none").value_counts().to_dict(), flush=True)

    verified = verified_table(dapps)
    verified.to_csv(DATA / "source_verified.csv", index=False)
    print(len(verified), "contracts checked,", int(verified["blockscout_error"].eq(429).sum()), "rate limited", flush=True)


if __name__ == "__main__":
    main()
