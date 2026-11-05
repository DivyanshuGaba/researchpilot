# gen-recursive-800-vector

`recursive` parent 800 / child 200 tokens, `vector`; config `882b47643416b9a0`, git `36ec516`, 49 questions (41 answerable).

| metric | value |
|---|---|
| hit@5 | 0.829 |
| recall@5 | 0.780 |
| hit@10 | 0.902 |
| recall@10 | 0.854 |
| mrr | 0.641 |
| faithfulness | 1.000 |
| correctness | 0.902 |
| relevance | 0.956 |
| citation_precision | 1.000 |
| refusal_accuracy | 1.000 |
| false_refusal_rate | 0.073 |
| answer_cost_usd_per_query | 0.014 |
| judge_cost_usd_per_query | 0.028 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 14.000 | 16.000 |
| dense | 14.000 | 15.600 |
| retrieve_total | 30.000 | 37.200 |
| generation | 2398.000 | 4102.000 |
