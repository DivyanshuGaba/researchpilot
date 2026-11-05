# gen-recursive-800-hybrid_rerank

`recursive` parent 800 / child 200 tokens, `hybrid_rerank`; config `48e15e0889bbaa02`, git `36ec516`, 49 questions (41 answerable).

| metric | value |
|---|---|
| hit@5 | 0.878 |
| recall@5 | 0.842 |
| hit@10 | 0.902 |
| recall@10 | 0.866 |
| mrr | 0.703 |
| faithfulness | 0.998 |
| correctness | 0.951 |
| relevance | 0.971 |
| citation_precision | 1.000 |
| refusal_accuracy | 1.000 |
| false_refusal_rate | 0.024 |
| answer_cost_usd_per_query | 0.012 |
| judge_cost_usd_per_query | 0.024 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 19.000 | 27.600 |
| dense | 18.000 | 25.600 |
| sparse | 17.000 | 26.800 |
| rerank | 174.000 | 218.400 |
| retrieve_total | 237.000 | 285.600 |
| generation | 2402.000 | 4508.800 |
