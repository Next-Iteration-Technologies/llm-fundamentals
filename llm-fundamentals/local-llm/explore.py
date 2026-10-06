"""
Extra experiments on the toy model: what changed when we trained it?

    uv run explore.py untrained ["prompt"]     # a model with random weights (also saves model_untrained.pt)
    uv run explore.py compare ["prompt"]       # untrained vs model_3k.pt vs model.pt, same prompt
    uv run explore.py temperature ["prompt"]   # same prompt continued at temperature 0.5, 0.8, 1.2
    uv run explore.py embeddings               # nearest tokens in the embedding table, untrained vs trained
    uv run explore.py attention ["sentence"]   # who listens to whom in the trained model

Add --model PATH to use a different trained model (default model.pt).
"""

import sys

import torch
import torch.nn.functional as F

from bpe_tokenizer import load, encode, decode
from model import ToyGPT

# The same settings train.py uses, so the untrained model is train.py's starting point.
CONFIG = dict(vocab_size=556, dim=32, n_heads=4, n_layers=2, max_positions=64)

merges, vocab = load("tokenizer.json")


def load_model(path):
    ckpt = torch.load(path)
    model = ToyGPT(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def untrained_model():
    """Random weights, built exactly like train.py does before step 0."""
    torch.manual_seed(0)
    model = ToyGPT(**CONFIG)
    model.eval()
    torch.save({"state": model.state_dict(), "config": CONFIG}, "model_untrained.pt")
    return model


def show(model, prompt, label="", temperature=0.8, n_continue=40):
    """Top-5 next tokens and a continuation, like generate.py, but with a fixed seed."""
    ids = torch.tensor([encode(prompt, merges)])
    probs = model.next_token_probs(ids)[0]
    top = torch.topk(probs, 5)

    print(f'\n{label}Prompt: "{prompt}"')
    print("Top-5 next tokens:")
    for p, i in zip(top.values, top.indices):
        print(f"   {p:.3f}  {decode([i.item()], vocab)!r}")
    print(f"   (confidence of best guess: {top.values[0]:.0%})")

    torch.manual_seed(0)                    # same dice every run, so the text is repeatable
    out = model.generate(ids, max_new_tokens=n_continue, temperature=temperature)
    print("Continuation:", repr(decode(out[0].tolist(), vocab)))


# ---------------------------------------------------------------------------
# SUBCOMMANDS
# ---------------------------------------------------------------------------

def cmd_untrained(prompts, model_path):
    model = untrained_model()
    print("Untrained model (random weights, seed 0). Saved as model_untrained.pt")
    for prompt in prompts:
        show(model, prompt)


def cmd_compare(prompts, model_path):
    models = [("UNTRAINED", untrained_model()),
              ("3,000 STEPS", load_model("model_3k.pt")),
              (f"{model_path}", load_model(model_path))]
    for prompt in prompts:
        for label, model in models:
            show(model, prompt, label=f"[{label}]  ")


def cmd_temperature(prompts, model_path):
    model = load_model(model_path)
    for prompt in prompts:
        for t in (0.5, 0.8, 1.2):
            ids = torch.tensor([encode(prompt, merges)])
            torch.manual_seed(0)
            out = model.generate(ids, max_new_tokens=40, temperature=t)
            print(f"\ntemperature {t}:", repr(decode(out[0].tolist(), vocab)))


def nearest(table, token_id, k=5):
    """The k rows most similar (cosine) to row `token_id`, excluding itself."""
    sims = F.cosine_similarity(table[token_id].unsqueeze(0), table, dim=-1)
    sims[token_id] = -1.0
    best = torch.topk(sims, k)
    return [(decode([i.item()], vocab), s.item()) for s, i in zip(best.values, best.indices)]


def cmd_embeddings(prompts, model_path):
    untrained = untrained_model().state_dict()
    trained = torch.load(model_path)["state"]
    table = trained["embed.token_table.weight"]
    positions = trained["embed.position_table.weight"]
    print(f"Stored in {model_path} -> ['state']['embed.token_table.weight']     shape {tuple(table.shape)}")
    print(f"Stored in {model_path} -> ['state']['embed.position_table.weight']  shape {tuple(positions.shape)}")
    print("\nNearest tokens by cosine similarity (1.0 = same direction, 0 = unrelated):")

    for word in [" him", "she ", "king", "down", "when"]:
        ids = encode(word, merges)
        tid = ids[0]
        print(f"\n{decode([tid], vocab)!r}  (id {tid})  first 4 numbers: "
              f"{[round(x, 2) for x in table[tid][:4].tolist()]}")
        for label, state in (("untrained", untrained), ("trained  ", trained)):
            near = nearest(state["embed.token_table.weight"], tid)
            print(f"   {label}: " + ", ".join(f"{t!r} {s:.2f}" for t, s in near))


@torch.no_grad()
def cmd_attention(prompts, model_path):
    model = load_model(model_path)
    sentence = prompts[0]
    ids = torch.tensor([encode(sentence, merges)])
    labels = [decode([i], vocab) for i in ids[0].tolist()]

    layer = model.layers[0]
    x = layer.ln1(model.embed(ids))         # exactly what layer 0's heads receive

    torch.set_printoptions(precision=2, sci_mode=False)
    for h, head in enumerate(layer.attn.heads):
        _, w = head(x, return_weights=True)
        w = w[0]
        if h == 0:
            print(f'Layer 0, head 0 attention weights for "{sentence}"')
            print("(row = token asking, column = token listened to)\n")
            print(" " * 11 + "".join(f"{l!r:>10}" for l in labels))
            for label, row in zip(labels, w):
                print(f"{label!r:>11}" + "".join(f"{v:10.2f}" for v in row.tolist()))
        print(f"\nhead {h}: who each token listens to most")
        print("   " + ",  ".join(f"{labels[i]!r}->{labels[w[i].argmax()]!r}"
                                 for i in range(len(labels))))


# ---------------------------------------------------------------------------

COMMANDS = {"untrained": cmd_untrained, "compare": cmd_compare,
            "temperature": cmd_temperature, "embeddings": cmd_embeddings,
            "attention": cmd_attention}

if __name__ == "__main__":
    args = sys.argv[1:]
    model_path = "model.pt"
    if "--model" in args:
        i = args.index("--model")
        model_path = args[i + 1]
        del args[i:i + 2]
    if not args or args[0] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    cmd, rest = args[0], args[1:]
    if rest:
        prompts = [" ".join(rest)]
    elif cmd == "attention":
        prompts = ["Once upon a time there was a little girl."]
    else:
        prompts = ["Once upon a", "The quantum"]
    COMMANDS[cmd](prompts, model_path)
