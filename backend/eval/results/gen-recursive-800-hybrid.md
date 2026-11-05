# gen-recursive-800-hybrid

`recursive` parent 800 / child 200 tokens, `hybrid`; config `046e17ef4b1ec61b`, git `7f835e0`, 49 questions (41 answerable).

| metric | value |
|---|---|
| hit@5 | 0.805 |
| recall@5 | 0.756 |
| hit@10 | 0.878 |
| recall@10 | 0.842 |
| mrr | 0.666 |
| faithfulness | 1.000 |
| correctness | 0.866 |
| relevance | 0.942 |
| citation_precision | 1.000 |
| refusal_accuracy | 1.000 |
| false_refusal_rate | 0.073 |
| answer_cost_usd_per_query | 0.014 |
| judge_cost_usd_per_query | 0.029 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 15.000 | 16.600 |
| dense | 11.000 | 13.000 |
| sparse | 16.000 | 22.200 |
| retrieve_total | 45.000 | 57.600 |
| generation | 15443.000 | 31199.400 |
