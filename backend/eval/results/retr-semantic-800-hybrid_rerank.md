# retr-semantic-800-hybrid_rerank

`semantic` parent 800 / child 200 tokens, `hybrid_rerank`; config `31fe2091af602170`, git `7f835e0`, 49 questions (41 answerable).

| metric | value |
|---|---|
| hit@5 | 0.878 |
| recall@5 | 0.829 |
| hit@10 | 0.927 |
| recall@10 | 0.915 |
| mrr | 0.671 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 26.000 | 29.000 |
| dense | 27.000 | 34.200 |
| sparse | 35.000 | 47.600 |
| rerank | 171.000 | 238.600 |
| retrieve_total | 266.000 | 359.200 |
