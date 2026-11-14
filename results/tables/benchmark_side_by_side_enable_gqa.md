**GQA decode throughput: repeat_interleave (default) path, T4 (results/benchmark) vs enable_gqa (opt-in) path, T4 (benchmark_enable_gqa)**

| batch | context | GQA tok/s (benchmark) | GQA tok/s (benchmark_enable_gqa) | GQA vs MHA (benchmark) | GQA vs MHA (benchmark_enable_gqa) |
|---|---|---|---|---|---|
| 1 | 128 | 83.8 | 78.4 | −5.0% | −17.1% |
| 1 | 256 | 80.4 | 79.6 | +4.8% | −12.5% |
| 1 | 512 | 84.1 | 79.7 | −1.0% | −15.4% |
| 1 | 1024 | 83.5 | 80.8 | −6.3% | −15.3% |
| 1 | 2048 | 88.9 | 78.4 | +2.8% | −15.9% |
| 4 | 128 | 319.0 | 313.1 | +0.3% | −13.6% |
| 4 | 256 | 305.3 | 315.5 | −7.4% | −16.2% |
| 4 | 512 | 331.0 | 314.1 | −7.8% | −9.9% |
| 4 | 1024 | 329.7 | 306.5 | −3.9% | −14.0% |
| 4 | 2048 | 344.7 | 309.1 | −0.5% | −16.0% |
| 16 | 128 | 1,329.6 | 1,273.1 | −2.7% | −14.9% |
| 16 | 256 | 1,265.0 | 1,224.3 | −3.1% | −15.2% |
| 16 | 512 | 1,354.5 | 1,258.9 | −5.8% | −14.6% |
| 16 | 1024 | 1,314.7 | 1,260.8 | −3.3% | −15.9% |
| 16 | 2048 | 1,292.9 | 798.1 | +0.3% | −45.3% |
| 64 | 128 | 4,841.3 | 5,096.2 | −13.4% | −11.1% |
| 64 | 256 | 4,821.8 | 4,904.6 | −8.5% | −13.3% |
| 64 | 512 | 5,026.1 | 3,046.5 | −11.7% | −47.5% |
| 64 | 1024 | 2,820.9 | 1,698.4 | −48.2% | −71.3% |

*results/benchmark: Measured 1 Nov, before the enable_gqa change: GQA still copied K/V on every decode step*
*results/benchmark_enable_gqa: enable_gqa (opt-in) path, T4. Measured 2026-11-12 on Tesla T4, torch 2.11.0+cu130, commit 19f1268.*
*Each ratio is within one session; compare ratios, not absolute tok/s, across sessions. A missing row is a config one run did not complete (e.g. OOM).*
