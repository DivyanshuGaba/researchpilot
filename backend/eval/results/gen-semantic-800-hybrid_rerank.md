# gen-semantic-800-hybrid_rerank

`semantic` parent 800 / child 200 tokens, `hybrid_rerank`; config `31fe2091af602170`, git `36ec516`, 49 questions (41 answerable).

| metric | value |
|---|---|
| hit@5 | 0.878 |
| recall@5 | 0.829 |
| hit@10 | 0.927 |
| recall@10 | 0.915 |
| mrr | 0.671 |
| faithfulness | 0.998 |
| correctness | 0.927 |
| relevance | 0.961 |
| citation_precision | 0.984 |
| refusal_accuracy | 1.000 |
| false_refusal_rate | 0.024 |
| answer_cost_usd_per_query | 0.013 |
| judge_cost_usd_per_query | 0.026 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 25.000 | 28.000 |
| dense | 25.000 | 30.600 |
| sparse | 32.000 | 42.200 |
| rerank | 165.000 | 236.200 |
| retrieve_total | 252.000 | 340.800 |
| generation | 2457.000 | 4709.000 |
