# Results

*Made by `python rag.py report` from the run files in this folder. Don't edit by hand.*

Scores are per question, then averaged. R@k = Recall@k (of the right passages, how many in the top k); MRR = how high the first right chunk came; All found = every right passage in the top 5; R@~2k tok = Recall when every setup sends about 2,000 tokens; R@50 = in the top 50 at all.

## contoso

### eval

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Changed vs row above |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | structure | semantic | ollama-embed | 17 | 0.58 | 0.61 | 0.69 | 0.58 | 0.47 | 0.89 | 0.89 |  |
| perfect | structure | — | — | 17 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| random | structure | — | ollama-embed | 17 | 0.02 | 0.03 | 0.04 | 0.01 | 0.00 | 0.08 | 0.08 |  |

## dnd-srd

### eval

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Changed vs row above |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | fixed-200 | semantic | bge-small | 55 | 0.44 | 0.45 | 0.58 | 0.44 | 0.24 | 0.51 | 0.63 |  |
| perfect | fixed-200 | — | — | 55 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 |  |
| random | fixed-200 | — | bge-small | 55 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.03 |  |

By level (R@3 / R@5 / MRR):

| Run | easy | medium |
|---|---|---|
| baseline | 0.59 / 0.63 / 0.76 | 0.31 / 0.30 / 0.43 |
| perfect | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| random | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.01 |

