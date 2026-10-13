# Results

*Made by `python rag.py report` from the run files in this folder. Don't edit by hand.*

Scores are per question, then averaged. R@k = Recall@k (of the right passages, how many in the top k); MRR = how high the first right chunk came; All found = every right passage in the top 5; R@~2k tok = Recall when every setup sends about 2,000 tokens; R@50 = in the top 50 at all.

## contoso

### eval

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | structure | semantic | ollama-embed | 17 | 0.58 | 0.61 | 0.69 | 0.58 | 0.47 | 0.89 | 0.89 |  |
| perfect | structure | — | — | 17 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| random | structure | — | ollama-embed | 17 | 0.02 | 0.03 | 0.04 | 0.01 | 0.00 | 0.08 | 0.08 |  |

### chunking

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| structure | structure | semantic | ollama-embed | 17 | 0.58 | 0.61 | 0.69 | 0.58 | 0.47 | 0.89 | 0.89 |  |
| naive-500 | naive-500 | semantic | ollama-embed | 17 | 0.45 | 0.47 | 0.49 | 0.42 | 0.35 | 0.73 | 0.93 | won 3, lost 5 |
| fixed-200 | fixed-200 | semantic | ollama-embed | 17 | 0.36 | 0.50 | 0.46 | 0.38 | 0.29 | 0.75 | 0.93 | won 4, lost 3 |
| recursive | recursive | semantic | ollama-embed | 17 | 0.52 | 0.60 | 0.61 | 0.54 | 0.47 | 0.74 | 0.93 | won 4, lost 2 |
| sentence | sentence | semantic | ollama-embed | 17 | 0.40 | 0.42 | 0.54 | 0.40 | 0.29 | 0.86 | 0.91 | won 4, lost 8 |
| semantic | semantic | semantic | ollama-embed | 17 | 0.59 | 0.64 | 0.67 | 0.58 | 0.53 | 0.87 | 0.96 | won 7, lost 2 |
| hierarchical | hierarchical | semantic | ollama-embed | 17 | 0.66 | 0.80 | 0.73 | 0.67 | 0.65 | 0.77 | 1.00 | won 5, lost 0 |

### retrieval

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keyword | hierarchical | keyword | ollama-embed | 17 | 0.41 | 0.60 | 0.56 | 0.49 | 0.47 | 0.54 | 0.94 |  |
| semantic | hierarchical | semantic | ollama-embed | 17 | 0.66 | 0.80 | 0.73 | 0.67 | 0.65 | 0.77 | 1.00 | won 6, lost 1 |
| hybrid | hierarchical | hybrid | ollama-embed | 17 | 0.58 | 0.76 | 0.70 | 0.63 | 0.59 | 0.65 | 1.00 | won 2, lost 2 |
| hybrid-rerank | hierarchical | hybrid + rerank | ollama-embed | 17 | 0.54 | 0.71 | 0.68 | 0.61 | 0.59 | 0.67 | 1.00 | won 2, lost 4 |
| hybrid-rerank-rewrite | hierarchical | hybrid + rerank + rewrite | ollama-embed | 17 | 0.54 | 0.73 | 0.68 | 0.62 | 0.59 | 0.67 | 1.00 | won 1, lost 0 |
| keyword@semantic | semantic | keyword | ollama-embed | 17 | 0.31 | 0.44 | 0.46 | 0.38 | 0.29 | 0.82 | 0.83 |  |
| semantic@semantic | semantic | semantic | ollama-embed | 17 | 0.59 | 0.64 | 0.67 | 0.58 | 0.53 | 0.87 | 0.96 | won 7, lost 1 |
| hybrid@semantic | semantic | hybrid | ollama-embed | 17 | 0.51 | 0.63 | 0.62 | 0.54 | 0.47 | 0.85 | 0.97 | won 2, lost 2 |
| hybrid-rerank@semantic | semantic | hybrid + rerank | ollama-embed | 17 | 0.45 | 0.58 | 0.63 | 0.54 | 0.47 | 0.91 | 0.97 | won 3, lost 3 |
| keyword@fixed-200 | fixed-200 | keyword | ollama-embed | 17 | 0.33 | 0.45 | 0.51 | 0.40 | 0.35 | 0.71 | 0.95 |  |
| semantic@fixed-200 | fixed-200 | semantic | ollama-embed | 17 | 0.36 | 0.50 | 0.46 | 0.38 | 0.29 | 0.75 | 0.93 | won 7, lost 3 |
| hybrid@fixed-200 | fixed-200 | hybrid | ollama-embed | 17 | 0.41 | 0.57 | 0.53 | 0.47 | 0.41 | 0.77 | 1.00 | won 3, lost 1 |
| hybrid-rerank@fixed-200 | fixed-200 | hybrid + rerank | ollama-embed | 17 | 0.47 | 0.61 | 0.64 | 0.54 | 0.47 | 0.80 | 1.00 | won 4, lost 3 |

### embedding

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hash | hierarchical | semantic | hash | 17 | 0.49 | 0.59 | 0.62 | 0.51 | 0.41 | 0.56 | 0.96 |  |
| ollama-embed | hierarchical | semantic | ollama-embed | 17 | 0.66 | 0.80 | 0.73 | 0.67 | 0.65 | 0.77 | 1.00 | won 5, lost 0 |
| bge-small | hierarchical | semantic | bge-small | 17 | 0.56 | 0.73 | 0.77 | 0.67 | 0.59 | 0.70 | 1.00 | won 1, lost 3 |

### filters

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-filters | hierarchical | semantic | ollama-embed | 17 | 0.66 | 0.80 | 0.73 | 0.67 | 0.65 | 0.77 | 1.00 |  |
| asker | hierarchical | semantic, asker=golden | ollama-embed | 17 | 0.69 | 0.77 | 0.67 | 0.64 | 0.65 | 0.71 | 0.94 | won 1, lost 1 |
| platform | hierarchical | semantic, platform=question | ollama-embed | 17 | 0.66 | 0.80 | 0.74 | 0.68 | 0.65 | 0.77 | 1.00 | won 1, lost 1 |
| after-2026 | hierarchical | semantic, after=2026-01-01 | ollama-embed | 17 | 0.66 | 0.77 | 0.70 | 0.65 | 0.59 | 0.74 | 0.97 | won 0, lost 1 |
| prefer-newest | hierarchical | semantic, prefer_newest=True | ollama-embed | 17 | 0.34 | 0.80 | 0.44 | 0.50 | 0.65 | 0.62 | 1.00 | won 1, lost 0 |
| asker+prefer-newest | hierarchical | semantic, asker=golden, prefer_newest=True | ollama-embed | 17 | 0.40 | 0.77 | 0.39 | 0.47 | 0.65 | 0.56 | 0.94 | won 1, lost 1 |

## dnd-srd

### eval

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
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

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
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

### retrieval

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keyword | recursive | keyword | bge-small | 55 | 0.35 | 0.38 | 0.51 | 0.37 | 0.22 | 0.43 | 0.54 |  |
| semantic | recursive | semantic | bge-small | 55 | 0.50 | 0.52 | 0.65 | 0.50 | 0.35 | 0.58 | 0.65 | won 19, lost 3 |
| hybrid | recursive | hybrid | bge-small | 55 | 0.44 | 0.44 | 0.64 | 0.46 | 0.27 | 0.52 | 0.68 | won 1, lost 14 |
| hybrid-rerank | recursive | hybrid + rerank | bge-small | 55 | 0.47 | 0.48 | 0.63 | 0.48 | 0.29 | 0.59 | 0.68 | won 13, lost 6 |
| hybrid-rerank-rewrite | recursive | hybrid + rerank + rewrite | bge-small | 55 | 0.47 | 0.48 | 0.62 | 0.47 | 0.27 | 0.59 | 0.67 | won 5, lost 4 |
| keyword@fixed-200 | fixed-200 | keyword | bge-small | 55 | 0.39 | 0.40 | 0.50 | 0.39 | 0.22 | 0.45 | 0.55 |  |
| semantic@fixed-200 | fixed-200 | semantic | bge-small | 55 | 0.44 | 0.45 | 0.58 | 0.44 | 0.24 | 0.51 | 0.63 | won 18, lost 9 |
| hybrid@fixed-200 | fixed-200 | hybrid | bge-small | 55 | 0.44 | 0.43 | 0.62 | 0.45 | 0.24 | 0.49 | 0.65 | won 9, lost 14 |
| hybrid-rerank@fixed-200 | fixed-200 | hybrid + rerank | bge-small | 55 | 0.45 | 0.48 | 0.64 | 0.49 | 0.25 | 0.53 | 0.65 | won 16, lost 8 |

By level (R@3 / R@5 / MRR):

| Run | easy | medium |
|---|---|---|
| keyword | 0.47 / 0.52 / 0.60 | 0.24 / 0.26 / 0.43 |
| semantic | 0.69 / 0.77 / 0.80 | 0.34 / 0.31 / 0.53 |
| hybrid | 0.63 / 0.63 / 0.76 | 0.28 / 0.28 / 0.55 |
| hybrid-rerank | 0.64 / 0.69 / 0.79 | 0.33 / 0.31 / 0.50 |
| hybrid-rerank-rewrite | 0.63 / 0.70 / 0.75 | 0.33 / 0.31 / 0.50 |
| keyword@fixed-200 | 0.57 / 0.56 / 0.65 | 0.24 / 0.26 / 0.37 |
| semantic@fixed-200 | 0.59 / 0.63 / 0.76 | 0.31 / 0.30 / 0.43 |
| hybrid@fixed-200 | 0.61 / 0.65 / 0.79 | 0.29 / 0.25 / 0.48 |
| hybrid-rerank@fixed-200 | 0.63 / 0.69 / 0.80 | 0.29 / 0.31 / 0.51 |

### embedding

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hash | recursive | semantic | hash | 55 | 0.24 | 0.27 | 0.34 | 0.25 | 0.13 | 0.33 | 0.43 |  |
| ollama-embed | recursive | semantic | ollama-embed | 55 | 0.49 | 0.53 | 0.66 | 0.52 | 0.36 | 0.60 | 0.70 | won 26, lost 2 |
| bge-small | recursive | semantic | bge-small | 55 | 0.50 | 0.52 | 0.65 | 0.50 | 0.35 | 0.58 | 0.65 | won 7, lost 8 |

By level (R@3 / R@5 / MRR):

| Run | easy | medium |
|---|---|---|
| hash | 0.35 / 0.38 / 0.44 | 0.15 / 0.17 / 0.26 |
| ollama-embed | 0.69 / 0.77 / 0.81 | 0.33 / 0.33 / 0.54 |
| bge-small | 0.69 / 0.77 / 0.80 | 0.34 / 0.31 / 0.53 |

### final: RAG vs full context (same model, same judge)

| Model | Row | Answered | Answer rel. | Faithful to right passages | Must say | Refusals right | Cites right file | Tokens / question | Seconds / question | Judge |
|---|---|---|---|---|---|---|---|---|---|---|
| ollama (llama3.2:3b) | RAG | 1 of 1 | 0.75 | 0.50 |   -   |   -   | 0.00 | 750 | 114.70 | ollama (llama3.2:3b) |
| ollama (llama3.2:3b) | full context | 0 of 1 (1 too big for the window) |   -   |   -   |   -   |   -   |   -   | 375,129 |   -   | ollama (llama3.2:3b) |
RAG setup: embedding/ollama-embed.

