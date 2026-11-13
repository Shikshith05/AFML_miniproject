**Evaluation of the GQA chain, SFT data: dolly**

| model | SFT data | story ppl | pref acc (sum) | pref acc (per-token) | DPO reward acc vs SFT | eos rate | distinct-2 | mean tokens |
|---|---|---|---|---|---|---|---|---|
| base | - | 3.778 | 0.785 | 0.910 | - | 0.75 | 0.863 | 105.1 |
| SFT | dolly | 4.120 | 0.750 | 0.900 | - | 0.83 | 0.915 | 62.3 |
| DPO | dolly | 4.154 | 0.790 | 0.940 | 0.970 | 0.58 | 0.825 | 99.1 |

*story ppl: held-out TinyStories validation stories (forgetting check). Preference accuracy: 200 held-out pairs, real ending vs weak checkpoint's. Generation scores: 12 fixed prompts, same seed per prompt.*
