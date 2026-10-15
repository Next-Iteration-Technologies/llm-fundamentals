# Results

*Made by `python rag.py report` from the run files in this folder. Don't edit by hand.*

Scores are per question, then averaged. R@k = Recall@k (of the right passages, how many in the top k); MRR = how high the first right chunk came; All found = every right passage in the top 5; R@~2k tok = Recall when every setup sends about 2,000 tokens; R@50 = in the top 50 at all.

## contoso

### eval

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | structure | semantic | ollama-embed | 38 | 0.80 | 0.89 | 0.80 | 0.80 | 0.82 | 0.94 | 0.96 |  |
| perfect | structure | — | — | 38 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| random | structure | — | ollama-embed | 38 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.03 | 0.11 |  |

### chunking

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| structure | structure | semantic | ollama-embed | 38 | 0.80 | 0.89 | 0.80 | 0.80 | 0.82 | 0.94 | 0.96 |  |
| naive-500 | naive-500 | semantic | ollama-embed | 38 | 0.61 | 0.71 | 0.65 | 0.62 | 0.61 | 0.86 | 0.97 | won 3, lost 10 |
| fixed-200 | fixed-200 | semantic | ollama-embed | 38 | 0.65 | 0.73 | 0.63 | 0.63 | 0.68 | 0.76 | 0.93 | won 8, lost 6 |
| recursive | recursive | semantic | ollama-embed | 38 | 0.79 | 0.82 | 0.77 | 0.75 | 0.74 | 0.88 | 0.97 | won 6, lost 4 |
| sentence | sentence | semantic | ollama-embed | 38 | 0.71 | 0.76 | 0.68 | 0.66 | 0.66 | 0.96 | 0.96 | won 3, lost 7 |
| semantic | semantic | semantic | ollama-embed | 38 | 0.60 | 0.66 | 0.57 | 0.56 | 0.58 | 0.86 | 0.91 | won 1, lost 5 |
| hierarchical | hierarchical | semantic | ollama-embed | 38 | 0.78 | 0.84 | 0.74 | 0.75 | 0.76 | 0.87 | 0.98 | won 9, lost 0 |

### retrieval

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keyword | structure | keyword | ollama-embed | 38 | 0.65 | 0.73 | 0.56 | 0.58 | 0.68 | 0.83 | 0.92 |  |
| semantic | structure | semantic | ollama-embed | 38 | 0.80 | 0.89 | 0.80 | 0.80 | 0.82 | 0.94 | 0.96 | won 8, lost 0 |
| hybrid | structure | hybrid | ollama-embed | 38 | 0.77 | 0.85 | 0.75 | 0.76 | 0.76 | 0.92 | 0.96 | won 2, lost 3 |
| hybrid-rerank | structure | hybrid + rerank | ollama-embed | 38 | 0.79 | 0.89 | 0.73 | 0.75 | 0.82 | 0.96 | 0.96 | won 4, lost 3 |
| hybrid-rerank-rewrite | structure | hybrid + rerank + rewrite | ollama-embed | 38 | 0.79 | 0.89 | 0.74 | 0.76 | 0.82 | 0.97 | 0.97 | won 1, lost 1 |
| keyword@hierarchical | hierarchical | keyword | ollama-embed | 38 | 0.61 | 0.70 | 0.60 | 0.60 | 0.68 | 0.76 | 0.89 |  |
| semantic@hierarchical | hierarchical | semantic | ollama-embed | 38 | 0.78 | 0.84 | 0.74 | 0.75 | 0.76 | 0.87 | 0.98 | won 8, lost 0 |
| hybrid@hierarchical | hierarchical | hybrid | ollama-embed | 38 | 0.80 | 0.87 | 0.76 | 0.78 | 0.82 | 0.86 | 0.97 | won 3, lost 3 |
| hybrid-rerank@hierarchical | hierarchical | hybrid + rerank | ollama-embed | 38 | 0.85 | 0.89 | 0.82 | 0.82 | 0.82 | 0.90 | 0.97 | won 2, lost 1 |
| keyword@fixed-200 | fixed-200 | keyword | ollama-embed | 38 | 0.66 | 0.72 | 0.61 | 0.60 | 0.63 | 0.81 | 0.89 |  |
| semantic@fixed-200 | fixed-200 | semantic | ollama-embed | 38 | 0.65 | 0.73 | 0.63 | 0.63 | 0.68 | 0.76 | 0.93 | won 6, lost 3 |
| hybrid@fixed-200 | fixed-200 | hybrid | ollama-embed | 38 | 0.71 | 0.74 | 0.70 | 0.67 | 0.68 | 0.77 | 0.96 | won 2, lost 3 |
| hybrid-rerank@fixed-200 | fixed-200 | hybrid + rerank | ollama-embed | 38 | 0.68 | 0.81 | 0.64 | 0.66 | 0.74 | 0.89 | 0.96 | won 5, lost 2 |

### embedding

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hash | structure | semantic | hash | 38 | 0.49 | 0.63 | 0.42 | 0.44 | 0.58 | 0.81 | 0.94 |  |
| bge-small | structure | semantic | bge-small | 38 | 0.72 | 0.85 | 0.76 | 0.76 | 0.76 | 0.91 | 0.92 | won 11, lost 1 |
| ollama-embed | structure | semantic | ollama-embed | 38 | 0.80 | 0.89 | 0.80 | 0.80 | 0.82 | 0.94 | 0.96 | won 2, lost 1 |
| bge-base | structure | semantic | bge-base | 38 | 0.86 | 0.89 | 0.79 | 0.80 | 0.84 | 0.93 | 0.96 | won 1, lost 1 |

### filters

| Run | Chunking | Search | Embedding | Q | R@3 | R@5 | MRR | nDCG@5 | All found | R@~2k tok | R@50 | Won / lost vs row above (Recall@5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-filters | structure | semantic | ollama-embed | 38 | 0.80 | 0.89 | 0.80 | 0.80 | 0.82 | 0.94 | 0.96 |  |
| asker | structure | semantic, asker=golden | ollama-embed | 38 | 0.85 | 0.89 | 0.82 | 0.82 | 0.82 | 0.96 | 0.96 | won 0, lost 0 |
| country | structure | semantic, country=question | ollama-embed | 38 | 0.80 | 0.89 | 0.80 | 0.80 | 0.82 | 0.94 | 0.96 | won 0, lost 0 |
| after-2026 | structure | semantic, after=2026-01-01 | ollama-embed | 38 | 0.51 | 0.61 | 0.53 | 0.53 | 0.53 | 0.63 | 0.66 | won 0, lost 12 |
| prefer-newest | structure | semantic, prefer_newest=True | ollama-embed | 38 | 0.58 | 0.89 | 0.52 | 0.60 | 0.82 | 0.94 | 0.96 | won 12, lost 0 |
| asker+prefer-newest | structure | semantic, asker=golden, prefer_newest=True | ollama-embed | 38 | 0.61 | 0.89 | 0.54 | 0.61 | 0.82 | 0.96 | 0.96 | won 0, lost 0 |

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

### answers (RAG triad)

| Run | Chat | Judge | Q | Context rel. | Faithfulness | Answer rel. | Must say | Refusals right |
|---|---|---|---|---|---|---|---|---|
| embedding/ollama-embed | ollama (llama3.2:3b) | ollama (llama3.2:3b) | 10 | 0.54 | 0.69 | 0.75 |   -   | 1 of 1 |
| eval/baseline | ollama (llama3.2:3b) | ollama (llama3.2:3b) | 10 | 0.46 | 0.68 | 0.68 |   -   | 1 of 1 |

### final: RAG vs full context (same model, same judge)

| Model | Row | Answered | Answer rel. | Faithful to right passages | Must say | Refusals right | Cites right file | Tokens / question | Seconds / question | Judge |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic (claude-opus-5-5) | RAG | 5 of 5 | 0.75 | 0.52 |   -   | 1.00 | 0.75 | 1,243 | 8.84 | anthropic (claude-opus-5-5) |
| anthropic (claude-opus-5-5) | full context | 5 of 5 | 1.00 | 0.59 |   -   | 1.00 | 1.00 | 627,711 | 21.92 | anthropic (claude-opus-5-5) |
| ollama (llama3.2:3b) | RAG | 1 of 1 | 0.75 | 0.50 |   -   |   -   | 0.00 | 750 | 114.70 | ollama (llama3.2:3b) |
| ollama (llama3.2:3b) | full context | 0 of 1 (1 too big for the window) |   -   |   -   |   -   |   -   |   -   | 375,129 |   -   | ollama (llama3.2:3b) |
RAG setup: embedding/ollama-embed.

