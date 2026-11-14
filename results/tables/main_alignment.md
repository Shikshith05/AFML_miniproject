**GQA model through SFT and DPO**

| model | story ppl | pref acc (sum) | pref acc (per-token) | DPO reward acc vs SFT |
|---|---|---|---|---|
| base | 3.778 | 0.785 | 0.910 | - |
| SFT (alpaca) | 4.555 | 0.760 | 0.915 | - |
| DPO (alpaca) | 4.752 | 0.780 | 0.945 | 0.960 |
| base | 3.778 | 0.785 | 0.910 | - |
| SFT (alpaca_matched) | 4.067 | 0.770 | 0.920 | - |
| DPO (alpaca_matched) | 4.170 | 0.805 | 0.945 | 0.965 |
| base | 3.778 | 0.785 | 0.910 | - |
| SFT (dolly) | 4.120 | 0.750 | 0.900 | - |
| DPO (dolly) | 4.154 | 0.790 | 0.940 | 0.970 |
| base | 3.778 | 0.785 | 0.910 | - |
| SFT (oasst1) | 3.846 | 0.770 | 0.920 | - |
| DPO (oasst1) | 3.934 | 0.800 | 0.945 | 0.980 |

*Full tables: results/tables/eval.md, plus eval_<dataset>.md per extra SFT dataset.*
