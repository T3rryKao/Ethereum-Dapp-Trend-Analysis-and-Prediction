# Off-chain flags

Off-chain data for the DApps in `dapp_data_v2/dapp_static.csv`, ready to merge into the model on `slug` and `k`.

| File | Rows | Use |
|---|---|---|
| `data/dapp_offchain_by_k.csv` | 512 × k = 1-9 | Final table for the model |
| `data/governance_monthly.csv` | 512 × month 0-12 | Governance over time |
| `data/dapp_offchain_flags.csv` | 676 | Flags with no dates |

`split` is `train` for 2020-22 launches and `test` for 2023-24. Window k ends at month k-1, counted in calendar months from the deployment month.

| Column | Meaning | Source | Timing |
|---|---|---|---|
| `is_audited` | Audit status or audit link listed | DefiLlama | Today |
| `is_open_source` | Public repos on the project's GitHub account | DefiLlama, GitHub | Today |
| `is_verified` | At least one contract verified | Blockscout, Sourcify (stand-in for Etherscan) | Today |
| `has_governance` | At least one Snapshot proposal before the cut-off | DefiLlama, Snapshot | By k |
| `n_proposals` | All proposals before the cut-off, including earlier parent history | Snapshot | By k |
| `n_proposals_window` | Proposals from month 0 to the cut-off | Snapshot | By k |

Values are 1, 0 or blank for unknown. A blank is not a 0. `has_governance = 0` means none was found after searching with the DApp's own website, Twitter and GitHub. DApps sharing a parent share its Snapshot space.

Coverage on the 512 study DApps:

| Column | 1 | 0 | Blank |
|---|---|---|---|
| `is_audited` | 357 | 143 | 12 |
| `is_open_source` | 379 | 6 | 127 |
| `is_verified` | 472 | 11 | 29 |
| `has_governance`, k = 3 | 106 | 397 | 9 |

```bash
python fetch_flags.py
python fetch_governance.py
python fetch_proposals.py
python parse_flags.py
```

Needs `gh` signed in. API responses are cached in `data/raw`, which is not in Git.
