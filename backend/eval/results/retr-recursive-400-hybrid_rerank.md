# retr-recursive-400-hybrid_rerank

`recursive` parent 400 / child 100 tokens, `hybrid_rerank`; config `d1db89d9289aba7b`, git `7f835e0`, 49 questions (41 answerable).

| metric | value |
|---|---|
| hit@5 | 0.854 |
| recall@5 | 0.817 |
| hit@10 | 0.927 |
| recall@10 | 0.915 |
| mrr | 0.667 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 25.000 | 28.000 |
| dense | 26.000 | 30.600 |
| sparse | 36.000 | 48.000 |
| rerank | 110.000 | 139.600 |
| retrieve_total | 197.000 | 238.800 |
