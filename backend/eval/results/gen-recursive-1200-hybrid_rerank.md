# gen-recursive-1200-hybrid_rerank

`recursive` parent 1200 / child 300 tokens, `hybrid_rerank`; config `f7e5dd4de317c44e`, git `7f835e0`, 49 questions (41 answerable).

| metric | value |
|---|---|
| hit@5 | 0.927 |
| recall@5 | 0.878 |
| hit@10 | 0.951 |
| recall@10 | 0.927 |
| mrr | 0.740 |
| faithfulness | 0.998 |
| correctness | 0.951 |
| relevance | 0.976 |
| citation_precision | 0.992 |
| refusal_accuracy | 1.000 |
| false_refusal_rate | 0.024 |
| answer_cost_usd_per_query | 0.016 |
| judge_cost_usd_per_query | 0.031 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 26.000 | 30.000 |
| dense | 24.000 | 27.600 |
| sparse | 24.000 | 35.200 |
| rerank | 262.000 | 312.200 |
| retrieve_total | 341.000 | 391.200 |
| generation | 14730.000 | 25676.200 |
