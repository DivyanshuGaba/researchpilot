# gen-pilot

`recursive` parent 800 / child 200 tokens, `hybrid_rerank`; config `48e15e0889bbaa02`, git `7f835e0`, 7 questions (4 answerable).

| metric | value |
|---|---|
| hit@5 | 0.750 |
| recall@5 | 0.500 |
| hit@10 | 0.750 |
| recall@10 | 0.500 |
| mrr | 0.583 |
| faithfulness | 1.000 |
| correctness | 0.875 |
| relevance | 0.950 |
| citation_precision | 1.000 |
| refusal_accuracy | 1.000 |
| false_refusal_rate | 0.000 |
| answer_cost_usd_per_query | 0.011 |
| judge_cost_usd_per_query | 0.021 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 29.000 | 4861.500 |
| dense | 26.000 | 46.800 |
| sparse | 18.000 | 21.700 |
| rerank | 183.000 | 987.600 |
| retrieve_total | 265.000 | 5910.400 |
| generation | 2846.000 | 4672.300 |
