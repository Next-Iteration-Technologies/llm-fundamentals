"""
Step 5 of the toy language model: the match / no-match test.

    uv run generate.py "Once upon a"        # expect a sharp spike on " time"
    uv run generate.py "The quantum"        # expect a flat, unsure spread
    uv run generate.py                      # runs both of the above

Prints the top-5 next-token probabilities, then lets the model keep writing.
A language model never 'fails to predict' -- it always produces a spread.
A match looks like one tall bar; no match looks like many small ones.
"""

import sys

import torch

from bpe_tokenizer import load, encode, decode
from model import ToyGPT

merges, vocab = load("tokenizer.json")
ckpt = torch.load("model.pt")
model = ToyGPT(**ckpt["config"])
model.load_state_dict(ckpt["state"])
model.eval()


def show(prompt, n_continue=40):
    ids = torch.tensor([encode(prompt, merges)])
    probs = model.next_token_probs(ids)[0]
    top = torch.topk(probs, 5)

    print(f'\nPrompt: "{prompt}"')
    print("Top-5 next tokens:")
    for p, i in zip(top.values, top.indices):
        print(f"   {p:.3f}  {decode([i.item()], vocab)!r}")
    print(f"   (confidence of best guess: {top.values[0]:.0%})")

    out = model.generate(ids, max_new_tokens=n_continue, temperature=0.8)
    print("Continuation:", repr(decode(out[0].tolist(), vocab)))


if len(sys.argv) > 1:
    show(" ".join(sys.argv[1:]))
else:
    show("Once upon a")      # should match
    show("The quantum")      # should not
