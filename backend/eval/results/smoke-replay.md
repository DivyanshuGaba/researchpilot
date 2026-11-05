# smoke-replay

`recursive` parent 800 / child 200 tokens, `hybrid_rerank`; config `48e15e0889bbaa02`, git `36ec516`, 10 questions (7 answerable).

| metric | value |
|---|---|
| hit@5 | 0.857 |
| recall@5 | 0.714 |
| hit@10 | 0.857 |
| recall@10 | 0.714 |
| mrr | 0.655 |
| faithfulness | 1.000 |
| correctness | 0.929 |
| relevance | 0.971 |
| citation_precision | 1.000 |
| refusal_accuracy | 1.000 |
| false_refusal_rate | 0.000 |
| answer_cost_usd_per_query | 0.012 |
| judge_cost_usd_per_query | 0.023 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 26.500 | 3882.850 |
| dense | 26.000 | 35.050 |
| sparse | 18.000 | 20.000 |
| rerank | 173.000 | 970.300 |
| retrieve_total | 244.500 | 4908.400 |
| generation | 0.000 | 0.550 |
