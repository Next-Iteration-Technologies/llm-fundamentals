"""The evaluation lab (LAB.md): datasets, chunkings, retrievers, scores and saved runs, all in memory.

rag.py stays the entry point; lab/cli.py adds the lab's commands to it. HOST is the running rag.py module, so the lab
reuses its Contoso loaders, prompt and policy instead of copying them.
"""
from pathlib import Path

HOST = None
PY = Path(__file__).resolve().parents[1]      # rag-exercises/python
ROOT = PY.parent                              # rag-exercises
