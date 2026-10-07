"""
Step 4 of the toy language model: train it on stories.txt.

    uv run train.py                 # uses stories.txt, 3000 steps
    uv run train.py 10000           # more steps

Needs tokenizer.json from bpe_tokenizer.py (it will train one if missing).
Saves model.pt when done.

The loop:
    1. grab a random 64-token window from the stories
    2. input  = tokens 0..63, target = tokens 1..64   (the next token, every position)
    3. forward pass -> loss
    4. backward pass -> gradients for every weight, embedding rows included
    5. optimizer step -> nudge every weight a little
    6. repeat
"""

import os
import sys
import time

import torch

from bpe_tokenizer import train as train_tokenizer, encode, save, load
from model import ToyGPT

# ---------------------------------------------------------------------------
# Settings. Small on purpose. Bump DIM / N_LAYERS / STEPS once it works.
# ---------------------------------------------------------------------------
SOURCE = "stories.txt"
DIM = 32
N_HEADS = 4
N_LAYERS = 2
CONTEXT = 64            # tokens the model can see at once
BATCH = 32              # windows per step
LR = 3e-3
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
EVAL_EVERY = 200

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. Tokenizer: load or train
# ---------------------------------------------------------------------------
with open(SOURCE, encoding="utf-8") as f:
    text = f.read()

if os.path.exists("tokenizer.json"):
    merges, vocab = load("tokenizer.json")
else:
    print("No tokenizer.json found, training one...")
    merges, vocab = train_tokenizer(text, 300, verbose=False)
    save("tokenizer.json", merges, vocab)
VOCAB = len(vocab)

# ---------------------------------------------------------------------------
# 2. Encode the stories once and cache (the pure-Python encoder is slow)
# ---------------------------------------------------------------------------
if os.path.exists("data.pt"):
    data = torch.load("data.pt")
else:
    print("Encoding stories.txt (one-time, may take a minute)...")
    data = torch.tensor(encode(text, merges), dtype=torch.long)
    torch.save(data, "data.pt")
print(f"{len(data):,} tokens, vocabulary {VOCAB}")

# Hold out the last 10% to check we learn language, not just memorise.
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


def get_batch(src):
    """BATCH random windows: inputs (0..63) and targets (1..64)."""
    starts = torch.randint(0, len(src) - CONTEXT - 1, (BATCH,))
    x = torch.stack([src[s: s + CONTEXT] for s in starts])
    y = torch.stack([src[s + 1: s + CONTEXT + 1] for s in starts])
    return x, y


@torch.no_grad()
def estimate_loss(model, src, n=20):
    model.eval()
    losses = [model(*get_batch(src))[1].item() for _ in range(n)]
    model.train()
    return sum(losses) / n


# ---------------------------------------------------------------------------
# 3. Build model and optimizer
# ---------------------------------------------------------------------------
model = ToyGPT(VOCAB, DIM, N_HEADS, N_LAYERS, CONTEXT)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"random-guess loss would be ln({VOCAB}) = {torch.log(torch.tensor(float(VOCAB))):.2f}\n")

# ---------------------------------------------------------------------------
# 4. The loop
# ---------------------------------------------------------------------------
t0 = time.time()
for step in range(STEPS + 1):
    if step % EVAL_EVERY == 0:
        tr, va = estimate_loss(model, train_data), estimate_loss(model, val_data)
        print(f"step {step:5d}  train loss {tr:.3f}  val loss {va:.3f}  ({time.time() - t0:.0f}s)")

    x, y = get_batch(train_data)
    _, loss = model(x, y)           # forward
    optimizer.zero_grad()
    loss.backward()                 # backward: gradients for every weight
    optimizer.step()                # nudge

torch.save(
    {"state": model.state_dict(),
     "config": dict(vocab_size=VOCAB, dim=DIM, n_heads=N_HEADS,
                    n_layers=N_LAYERS, max_positions=CONTEXT)},
    "model.pt",
)
print("\nsaved model.pt  -> now run: uv run generate.py \"Once upon a\"")
