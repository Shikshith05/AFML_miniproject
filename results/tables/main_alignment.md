**GQA model through SFT and DPO**

| model | story ppl | pref acc (sum) | pref acc (per-token) | DPO reward acc vs SFT |
|---|---|---|---|---|
| base | 3.778 | 0.785 | 0.910 | - |
| SFT (alpaca) | 4.555 | 0.760 | 0.915 | - |
| DPO (alpaca) | 4.752 | 0.780 | 0.945 | 0.960 |

*Full tables: results/tables/eval.md, plus eval_<dataset>.md per extra SFT dataset.*
