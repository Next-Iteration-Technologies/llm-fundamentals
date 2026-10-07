"""
Step 3 of the toy language model: the full transformer.

Builds on attention.py:
    Embedding  -> N x [ attention (several heads) + feed-forward ] -> prediction head

One "layer" = attention then feed-forward, each wrapped with a residual
connection (add the input back) and layer norm (keep numbers tame).

Nothing here trains; see train.py for that.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Embedding, AttentionHead


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, each asking a different kind of question."""

    def __init__(self, dim, n_heads, max_positions):
        super().__init__()
        head_dim = dim // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(dim, head_dim, max_positions) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(dim, dim)   # stitch the heads' outputs back together

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (batch, T, dim)
        return self.proj(out)


class FeedForward(nn.Module):
    """
    The 'neuron nodes'. Works on each token's vector ALONE, no neighbours.
    Expand to 4x width, cut negatives (ReLU), shrink back.
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),   # 32 -> 128 neurons
            nn.ReLU(),
            nn.Linear(4 * dim, dim),   # 128 -> 32
        )

    def forward(self, x):
        return self.net(x)


class Layer(nn.Module):
    """One transformer layer: attention, then feed-forward, with residuals."""

    def __init__(self, dim, n_heads, max_positions):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, max_positions)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # gather from the room, add back to self
        x = x + self.ff(self.ln2(x))     # think about what was gathered, add back
        return x


class ToyGPT(nn.Module):
    def __init__(self, vocab_size, dim=32, n_heads=4, n_layers=2, max_positions=64):
        super().__init__()
        self.max_positions = max_positions
        self.embed = Embedding(vocab_size, dim, max_positions)
        self.layers = nn.ModuleList(
            [Layer(dim, n_heads, max_positions) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(dim)
        # Prediction head: vector (dim) -> one score per token in the vocabulary.
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, ids, targets=None):
        x = self.embed(ids)                 # (batch, T, dim)
        for layer in self.layers:
            x = layer(x)                    # same shape, a little wiser each time
        x = self.ln_final(x)
        logits = self.head(x)               # (batch, T, vocab)  raw scores per position

        loss = None
        if targets is not None:
            # Cross-entropy = -log(probability given to the correct next token),
            # averaged over every position in the batch.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (batch*T, vocab)
                targets.view(-1),                   # (batch*T,)
            )
        return logits, loss

    @torch.no_grad()
    def next_token_probs(self, ids):
        """Probabilities for the token after the last one in `ids`."""
        ids = ids[:, -self.max_positions:]          # never exceed the context window
        logits, _ = self(ids)
        return F.softmax(logits[:, -1, :], dim=-1)  # (batch, vocab)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=50, temperature=1.0):
        for _ in range(max_new_tokens):
            probs = self.next_token_probs(ids)
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            nxt = torch.multinomial(probs, num_samples=1)   # sample one token
            ids = torch.cat([ids, nxt], dim=1)              # append and go again
        return ids


if __name__ == "__main__":
    model = ToyGPT(vocab_size=556)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ToyGPT built: {n_params:,} parameters")
    ids = torch.randint(0, 556, (1, 10))
    logits, _ = model(ids)
    print("logits shape:", tuple(logits.shape), "(batch, tokens, vocab)")
