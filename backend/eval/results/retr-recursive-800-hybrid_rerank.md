# retr-recursive-800-hybrid_rerank

`recursive` parent 800 / child 200 tokens, `hybrid_rerank`; config `48e15e0889bbaa02`, git `7f835e0`, 49 questions (41 answerable).

| metric | value |
|---|---|
| hit@5 | 0.878 |
| recall@5 | 0.842 |
| hit@10 | 0.902 |
| recall@10 | 0.866 |
| mrr | 0.703 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 26.000 | 29.000 |
| dense | 25.000 | 28.000 |
| sparse | 30.000 | 39.000 |
| rerank | 178.000 | 220.600 |
| retrieve_total | 267.000 | 313.000 |
