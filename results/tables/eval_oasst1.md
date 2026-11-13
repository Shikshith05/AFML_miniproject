**Evaluation of the GQA chain, SFT data: oasst1**

| model | SFT data | story ppl | pref acc (sum) | pref acc (per-token) | DPO reward acc vs SFT | eos rate | distinct-2 | mean tokens |
|---|---|---|---|---|---|---|---|---|
| base | - | 3.778 | 0.785 | 0.910 | - | 0.75 | 0.863 | 105.1 |
| SFT | oasst1 | 3.846 | 0.770 | 0.920 | - | 1.00 | 0.883 | 62.3 |
| DPO | oasst1 | 3.934 | 0.800 | 0.945 | 0.980 | 0.50 | 0.911 | 93.8 |

*story ppl: held-out TinyStories validation stories (forgetting check). Preference accuracy: 200 held-out pairs, real ending vs weak checkpoint's. Generation scores: 12 fixed prompts, same seed per prompt.*
