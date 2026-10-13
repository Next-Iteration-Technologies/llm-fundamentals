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

### chunking

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Changed vs row above |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| structure | structure | semantic | ollama-embed | 17 | 0.58 | 0.61 | 0.69 | 0.58 | 0.47 | 0.89 | 0.89 |  |
| naive-500 | naive-500 | semantic | ollama-embed | 17 | 0.45 | 0.47 | 0.49 | 0.42 | 0.35 | 0.73 | 0.93 | won 3, lost 5 |
| fixed-200 | fixed-200 | semantic | ollama-embed | 17 | 0.36 | 0.50 | 0.46 | 0.38 | 0.29 | 0.75 | 0.93 | won 4, lost 3 |
| recursive | recursive | semantic | ollama-embed | 17 | 0.52 | 0.60 | 0.61 | 0.54 | 0.47 | 0.74 | 0.93 | won 4, lost 2 |
| sentence | sentence | semantic | ollama-embed | 17 | 0.40 | 0.42 | 0.54 | 0.40 | 0.29 | 0.86 | 0.91 | won 4, lost 8 |
| semantic | semantic | semantic | ollama-embed | 17 | 0.59 | 0.64 | 0.67 | 0.58 | 0.53 | 0.87 | 0.96 | won 7, lost 2 |
| hierarchical | hierarchical | semantic | ollama-embed | 17 | 0.66 | 0.80 | 0.73 | 0.67 | 0.65 | 0.77 | 1.00 | won 5, lost 0 |

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

### chunking

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Changed vs row above |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed-200 | fixed-200 | semantic | bge-small | 55 | 0.44 | 0.45 | 0.58 | 0.44 | 0.24 | 0.51 | 0.63 |  |
| recursive | recursive | semantic | bge-small | 55 | 0.50 | 0.52 | 0.65 | 0.50 | 0.35 | 0.58 | 0.65 | won 16, lost 11 |
| sentence | sentence | semantic | bge-small | 55 | 0.37 | 0.43 | 0.55 | 0.40 | 0.27 | 0.53 | 0.59 | won 3, lost 16 |
| semantic | semantic | semantic | bge-small | 55 | 0.45 | 0.45 | 0.57 | 0.44 | 0.29 | 0.50 | 0.62 | won 11, lost 8 |
| hierarchical | hierarchical | semantic | bge-small | 55 | 0.38 | 0.41 | 0.52 | 0.39 | 0.25 | 0.48 | 0.65 | won 8, lost 12 |

By level (R@3 / R@5 / MRR):

| Run | easy | medium |
|---|---|---|
| fixed-200 | 0.59 / 0.63 / 0.76 | 0.31 / 0.30 / 0.43 |
| recursive | 0.69 / 0.77 / 0.80 | 0.34 / 0.31 / 0.53 |
| sentence | 0.53 / 0.63 / 0.69 | 0.23 / 0.26 / 0.43 |
| semantic | 0.62 / 0.64 / 0.72 | 0.30 / 0.28 / 0.44 |
| hierarchical | 0.54 / 0.59 / 0.70 | 0.25 / 0.26 / 0.38 |

