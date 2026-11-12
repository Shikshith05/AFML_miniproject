**MHA vs GQA (27,795,968 parameters each)**

|  | MHA | GQA |
|---|---|---|
| KV heads | 8 | 2 |
| ffn_hidden (S7: parameter-matched) | 1408 | 1664 |
| final val loss, 12k steps | 1.3557 | 1.3530 |
| final val perplexity | 3.879 | 3.869 |
| KV cache, seq 512, fp16 (MB) | 8.39 | 2.10 |
| training tok/s, same session (1 Nov) | 42,054 | 44,629 |
| decode tok/s, batch 1, context 2048 (1 Nov) | 86.5 | 88.9 |
| decode tok/s, batch 64, context 2048 (1 Nov) | 3,385.9 | 1,631.7 |

*1 Nov: Measured 1 Nov, before the enable_gqa change: GQA still copied K/V on every decode step*
