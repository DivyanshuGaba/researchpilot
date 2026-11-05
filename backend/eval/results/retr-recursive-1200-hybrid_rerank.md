# retr-recursive-1200-hybrid_rerank

`recursive` parent 1200 / child 300 tokens, `hybrid_rerank`; config `f7e5dd4de317c44e`, git `7f835e0`, 49 questions (41 answerable).

| metric | value |
|---|---|
| hit@5 | 0.927 |
| recall@5 | 0.878 |
| hit@10 | 0.951 |
| recall@10 | 0.927 |
| mrr | 0.740 |

| stage | p50 ms | p95 ms |
|---|---|---|
| embed | 27.000 | 30.200 |
| dense | 24.000 | 27.000 |
| sparse | 30.000 | 37.600 |
| rerank | 268.000 | 306.400 |
| retrieve_total | 352.000 | 395.200 |
