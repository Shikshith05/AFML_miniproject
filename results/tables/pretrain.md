**Pretraining: MHA vs GQA, parameter-matched (27,795,968 each)**

| model | KV heads | ffn_hidden | steps | tokens | final val loss | final val ppl | KV cache MB (seq 512, fp16) | wall time h (T4) |
|---|---|---|---|---|---|---|---|---|
| MHA (8 KV heads) | 8 | 1408 | 12,000 | 522M | 1.3557 | 3.879 | 8.39 | 3.94 |
| GQA (2 KV heads) | 2 | 1664 | 12,000 | 522M | 1.3530 | 3.869 | 2.10 | 3.18 |

*Final values are the 'done' event of results/<run>/metrics.jsonl.*
*Wall times come from two different Kaggle sessions, so they are not a speed comparison; see the benchmark table for that.*
