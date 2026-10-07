"""
Step 2 of the toy language model: embeddings + ONE attention head.

Every line of the attention math is written out explicitly so you can
match it to the party analogy:
    query  = the question a token asks
    key    = the badge a token wears
    value  = what a token actually says when listened to
    score  = query . key   (how well a badge matches a question)
    softmax turns scores into shares of one unit of attention
    output = weighted blend of values

Run it as-is to see the shapes and the attention weights on random input:
    uv run attention.py

Requires torch, installed into this folder's .venv by `uv sync` (see pyproject.toml).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EMBEDDING: token ID -> vector, plus position -> vector, added together
# ---------------------------------------------------------------------------

class Embedding(nn.Module):
    def __init__(self, vocab_size, dim, max_positions):
        super().__init__()
        # Both tables start random. Training fills them with meaning.
        self.token_table = nn.Embedding(vocab_size, dim)        # vocab_size rows, dim columns
        self.position_table = nn.Embedding(max_positions, dim)  # one row per position slot

    def forward(self, ids):
        # ids: (batch, T) integers from the tokenizer
        batch, T = ids.shape
        tok = self.token_table(ids)                              # (batch, T, dim)  row lookup
        pos = self.position_table(torch.arange(T))               # (T, dim)         rows 0..T-1
        return tok + pos                                         # same shape as tok


# ---------------------------------------------------------------------------
# ONE ATTENTION HEAD
# ---------------------------------------------------------------------------

class AttentionHead(nn.Module):
    def __init__(self, dim, head_dim, max_positions):
        super().__init__()
        # The three learned matrices. Each turns a token's vector into one role.
        self.Wq = nn.Linear(dim, head_dim, bias=False)   # vector -> query  (the question)
        self.Wk = nn.Linear(dim, head_dim, bias=False)   # vector -> key    (the badge)
        self.Wv = nn.Linear(dim, head_dim, bias=False)   # vector -> value  (what it says)
        self.head_dim = head_dim
        # Mask so a token can only look at earlier tokens, never ahead.
        # Lower-triangular matrix of ones: row i has ones in columns 0..i.
        self.register_buffer("mask", torch.tril(torch.ones(max_positions, max_positions)))

    def forward(self, x, return_weights=False):
        # x: (batch, T, dim) -- the stack of vectors from the embedding step
        batch, T, dim = x.shape

        # 1. Each token, ALONE, computes its own query, key, value.
        q = self.Wq(x)                                   # (batch, T, head_dim)
        k = self.Wk(x)                                   # (batch, T, head_dim)
        v = self.Wv(x)                                   # (batch, T, head_dim)

        # 2. Every query meets every key: dot products give raw scores.
        #    scores[i][j] = how much token i's question matches token j's badge.
        scores = q @ k.transpose(-2, -1)                 # (batch, T, T)
        scores = scores / (self.head_dim ** 0.5)         # keep numbers tame

        # 3. Hide the future: token i must not see tokens j > i.
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # 4. Softmax: each row becomes shares of one unit of attention.
        weights = F.softmax(scores, dim=-1)              # (batch, T, T), each row sums to 1

        # 5. Blend: each token takes a weighted average of everyone's values.
        out = weights @ v                                # (batch, T, head_dim)

        if return_weights:
            return out, weights
        return out


# ---------------------------------------------------------------------------
# DEMO: random tokens, random weights, just to see the shapes and the numbers
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    VOCAB = 556          # 256 bytes + 300 merges from your tokenizer
    DIM = 32             # number of "facets" per token
    HEAD_DIM = 16        # size of q/k/v inside this one head
    MAX_POS = 64         # context window

    embed = Embedding(VOCAB, DIM, MAX_POS)
    head = AttentionHead(DIM, HEAD_DIM, MAX_POS)

    # Pretend tokenizer output for a 6-token sentence (batch of 1).
    ids = torch.tensor([[300, 412, 275, 301, 389, 412]])

    x = embed(ids)
    print("after embedding:", tuple(x.shape), "  (batch, tokens, dim)")

    out, weights = head(x, return_weights=True)
    print("after attention:", tuple(out.shape), "  (batch, tokens, head_dim)")

    torch.set_printoptions(precision=2, sci_mode=False)
    print("\nattention weights (row = which token is asking, col = who it listens to):")
    print(weights[0])
    print("\neach row sums to:", weights[0].sum(dim=-1))
    print("\nNote the zeros above the diagonal: no token can see the future.")
    print("The numbers are meaningless until training; only the shapes matter now.")
