"""
The debugger: follow one prompt through the trained toy GPT, stage by stage.
Every number is printed, and next to it what it means in tokens and words.

    uv run debug.py overview "Little Red"          # one line per stage
    uv run debug.py trace "Little Red"             # every number, for the token making the guess
    uv run debug.py full "Little Red"              # every number, for every token
    uv run debug.py compare "Little Red"           # untrained vs trained, stage by stage
    uv run debug.py off "Little Red"               # switch off each head and neuron: which ones matter?
    uv run debug.py off "Little Red" --off 1.2     # switch off layer 1 head 2 (1.ff57 = layer 1 neuron 57)
    uv run debug.py train-step "Little Red"        # one training step, in slow motion

Options:
    --model PATH    the model to inspect (default model.pt)
    --token N       trace: zoom on token N instead of the last one
    --layer L       trace, full: only layer L
    --head H        trace, full: only head H
    --html PATH     also write an interactive page with everything for this prompt: trained and
                    untrained traces, the switch-off ranking and the training step

Layers, heads, neurons and tokens are counted from 0, as in explore.py.

The vectors have no names, so each one is read through what it matches or what it
pushes the prediction towards:
    query    "looks for"    the tokens whose key it matches best
    key      "answers"      the tokens whose query matches it best
    value    "passes on"    the next tokens it pushes up when someone listens to it
    neuron   "fires for"    the places in the stories where it fires hardest
             "pushes"       the next tokens its output pushes up
    guess    "if it stopped here"   ln_final + prediction head, applied early (the "logit lens")
"""

import copy
import json
import sys

import torch
import torch.nn.functional as F

from bpe_tokenizer import encode, decode
from explore import load_model, merges, vocab

DEFAULT_PROMPT = "Once upon a time there was a little girl. Her name was Little Red"
TOP = 5            # how many tokens each reading lists
TOP_NEURONS = 6    # strongest neurons shown per token and layer
BLOCKS = " ▁▂▃▄▅▆▇█"


# Readings only list tokens that turn up at least 20 times in the stories (364 of 556).
# The others, mostly lone bytes of rarely used characters, were hardly trained, so
# their numbers are leftovers from the random start and would drown out the rest.
COUNTS = torch.bincount(torch.load("data.pt"), minlength=len(vocab))
RARE = COUNTS < 20


def tk(i):
    """A token as readable text: spaces as ·, new lines as ⏎ (a lone carriage return as ␍), a lone byte as ⟨e2⟩."""
    try:
        text = vocab[i].decode("utf-8")
    except UnicodeDecodeError:
        return "⟨" + " ".join(f"{b:02x}" for b in vocab[i]) + "⟩"
    text = text.replace("\r\n", "⏎").replace("\n", "⏎").replace("\r", "␍").replace(" ", "·")
    return "'" + "".join(f"⟨{ord(c):02x}⟩" if ord(c) < 32 or ord(c) == 127 else c for c in text) + "'"


def best_tokens(scores, k=TOP, lowest=False):
    """Top k common tokens by score: [['·she·', 0.77], ...]"""
    s = (-scores if lowest else scores).masked_fill(RARE, float("-inf"))
    v, i = torch.topk(s, k)
    return [[tk(j), round(-x if lowest else x, 2)] for x, j in zip(v.tolist(), i.tolist())]


def r(t, n=3):
    """Tensor -> plain list of rounded floats (for printing and for the page)."""
    return [round(x, n) for x in t.tolist()]


# ---------------------------------------------------------------------------
# THE FORWARD PASS, ONE STAGE AT A TIME
# Same submodules and the same order as ToyGPT.forward in model.py, but every
# intermediate value is kept. `off` switches one part off:
#     ("head", L, H)  -> head H of layer L says nothing
#     ("ff", L, N)    -> neuron N of layer L never fires
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward(model, ids, off=None):
    T = len(ids)
    x_ids = torch.tensor(ids)
    tok = model.embed.token_table(x_ids)                    # (T, 32) the token cards
    pos = model.embed.position_table(torch.arange(T))       # (T, 32) the seat cards
    x = tok + pos
    raw = {"ids": ids, "tok": tok, "pos": pos, "embed": x, "layers": []}

    for L, layer in enumerate(model.layers):
        st = {"x_in": x}
        a_in = layer.ln1(x)                                 # steadied before the heads
        st["a_in"] = a_in
        st["heads"], outs = [], []
        for H, head in enumerate(layer.attn.heads):
            q, k, v = head.Wq(a_in), head.Wk(a_in), head.Wv(a_in)       # (T, 8) each
            scores = q @ k.T / head.head_dim ** 0.5                     # (T, T)
            scores = scores.masked_fill(head.mask[:T, :T] == 0, float("-inf"))
            w = F.softmax(scores, dim=-1)
            out = w @ v                                                 # (T, 8)
            if off == ("head", L, H):
                out = torch.zeros_like(out)
            st["heads"].append({"q": q, "k": k, "v": v, "scores": scores, "w": w, "out": out})
            outs.append(out)
        st["attn"] = layer.attn.proj(torch.cat(outs, dim=-1))   # 4 x 8 stitched -> 32
        x = x + st["attn"]                                      # residual: add back
        st["mid"] = x

        f_in = layer.ln2(x)
        act = F.relu(layer.ff.net[0](f_in))                     # widen 32 -> 128, cut negatives
        if off is not None and off[0] == "ff" and off[1] == L:
            act = act.clone()
            act[:, off[2]] = 0
        st["f_in"], st["act"] = f_in, act
        st["ff"] = layer.ff.net[2](act)                         # shrink 128 -> 32
        x = x + st["ff"]                                        # residual: add back
        st["out"] = x
        raw["layers"].append(st)

    raw["logits"] = model.head(model.ln_final(x))           # (T, 556)
    raw["probs"] = F.softmax(raw["logits"], dim=-1)
    if off is None:
        # Proof that this trace IS the model: same answer as model(ids).
        real, _ = model(x_ids.unsqueeze(0))
        raw["check"] = (real[0] - raw["logits"]).abs().max().item()
        assert raw["check"] < 1e-4, f"trace differs from the model by {raw['check']}"
    return raw


# ---------------------------------------------------------------------------
# READERS: numbers -> tokens and words
# ---------------------------------------------------------------------------

@torch.no_grad()
def guess(model, x, k=TOP):
    """If the model stopped here: ln_final + prediction head on the vector so far."""
    p = F.softmax(model.head(model.ln_final(x)), dim=-1)
    v, i = torch.topk(p, k)
    return [[tk(j), round(q, 4)] for q, j in zip(v.tolist(), i.tolist())]


@torch.no_grad()
def push(model, d, k=TOP):
    """Which next tokens a change `d` (32 numbers) pushes up and down.
    A straight-line reading: centre d, scale it like ln_final, multiply by the
    prediction head. It ignores ln_final's overall resizing, so it's a direction, not exact %."""
    s = scores_of(model, d)
    return {"up": best_tokens(s, k), "down": best_tokens(s, 3, lowest=True)}


def scores_of(model, d):
    """How much a change d adds to each token's raw score (the straight-line reading used by push)."""
    return model.head.weight @ (model.ln_final.weight * (d - d.mean()))


@torch.no_grad()
def nearest_cards(model, tid, k=TOP):
    table = model.embed.token_table.weight
    sims = F.cosine_similarity(table[tid].unsqueeze(0), table, dim=-1)
    sims[tid] = -1
    return best_tokens(sims, k)


class VocabReader:
    """What every one of the 556 tokens would send into a layer's heads, standing
    alone at a given position. Exact for layer 0 (it only sees the embedding);
    for layer 1 the token has no neighbours, so it's an approximation."""

    def __init__(self, model):
        self.model, self.cache = model, {}

    @torch.no_grad()
    def inputs(self, L, position):
        if (L, position) not in self.cache:
            m = self.model
            x = (m.embed.token_table.weight + m.embed.position_table.weight[position]).unsqueeze(1)
            for layer in m.layers[:L]:
                x = layer(x)
            self.cache[L, position] = m.layers[L].ln1(x)[:, 0]          # (556, 32)
        return self.cache[L, position]

    def looks_for(self, L, H, position, q):
        """Query: which tokens' keys it matches best."""
        keys = self.model.layers[L].attn.heads[H].Wk(self.inputs(L, position))
        return best_tokens(keys @ q / q.numel() ** 0.5)

    def answers(self, L, H, position, k):
        """Key: which tokens' queries match it best."""
        queries = self.model.layers[L].attn.heads[H].Wq(self.inputs(L, position))
        return best_tokens(queries @ k / k.numel() ** 0.5)


@torch.no_grad()
def neuron_examples(model, n_tokens=20000):
    """One pass over the held-out stories, recording every neuron's firing.
    Returns the token ids and, per layer, a (tokens, 128) table of activations."""
    data = torch.load("data.pt")
    val = data[int(0.9 * len(data)):][:n_tokens]
    T = model.max_positions
    ids = val[: len(val) // T * T].view(-1, T)
    x, acts = model.embed(ids), []
    for layer in model.layers:
        x = x + layer.attn(layer.ln1(x))
        a = F.relu(layer.ff.net[0](layer.ln2(x)))
        acts.append(a.reshape(-1, a.shape[-1]))
        x = x + layer.ff.net[2](a)
    return {"ids": ids.reshape(-1).tolist(), "acts": acts, "T": T}


def fires_for(ex, L, n, k=3):
    """The k places in the stories where neuron n of layer L fires hardest, as '…Little [Red]'."""
    top = torch.topk(ex["acts"][L][:, n], k)
    out = []
    for a, p in zip(top.values.tolist(), top.indices.tolist()):
        start = max(p - 3, p - p % ex["T"])
        before = decode(ex["ids"][start:p], vocab).replace("\r", "").replace("\n", "⏎")
        here = decode([ex["ids"][p]], vocab).replace("\r", "").replace("\n", "⏎")
        out.append(f"…{before}[{here}]")
    return out


# ---------------------------------------------------------------------------
# THE REPORT: one plain dict with every number and every reading.
# The terminal printouts and the HTML page both read from it.
# ---------------------------------------------------------------------------

@torch.no_grad()
def report(model, prompt, label, examples):
    ids = encode(prompt, merges)
    if not 0 < len(ids) <= model.max_positions:
        sys.exit(f"The prompt must be 1 to {model.max_positions} tokens; this one is {len(ids)}.")
    raw = forward(model, ids)
    reader = VocabReader(model)
    n_heads = len(model.layers[0].attn.heads)
    hd = model.layers[0].attn.heads[0].head_dim
    T = len(ids)

    positions = []
    for i in range(T):
        win = int(raw["probs"][i].argmax())             # the token this position ends up predicting

        def lift(d):
            """How much change d raises the winning token's raw score."""
            return round(scores_of(model, d)[win].item(), 2)

        e = {"tok": r(raw["tok"][i]), "pos": r(raw["pos"][i]), "sum": r(raw["embed"][i]),
             "near": nearest_cards(model, ids[i]), "guess": guess(model, raw["embed"][i])}
        layers = []
        for L, st in enumerate(raw["layers"]):
            layer = model.layers[L]
            proj = layer.attn.proj.weight
            heads = []
            for H, h in enumerate(st["heads"]):
                cols = proj[:, H * hd:(H + 1) * hd]          # where this head's 8 numbers land in the 32
                heads.append({
                    "q": r(h["q"][i]), "k": r(h["k"][i]), "v": r(h["v"][i]),
                    "looks_for": reader.looks_for(L, H, i, h["q"][i]),
                    "answers": reader.answers(L, H, i, h["k"][i]),
                    "passes_on": push(model, cols @ h["v"][i]),
                    "scores": r(h["scores"][i, : i + 1], 2),
                    "weights": r(h["w"][i, : i + 1], 3),
                    "out": r(h["out"][i]),
                    "out_push": push(model, cols @ h["out"][i]),
                    "lift": lift(cols @ h["out"][i]),
                })
            # Neurons ranked by how much they lift the winning token (firing x its output column).
            act = st["act"][i]
            w2 = layer.ff.net[2].weight                          # (32, 128): neuron n's output is w2[:, n]
            lifts = torch.tensor([scores_of(model, w2[:, n] * act[n])[win].item() for n in range(act.numel())])
            order = torch.topk(lifts.abs(), TOP_NEURONS).indices.tolist()
            neurons = [{"n": n, "act": round(act[n].item(), 2), "lift": round(lifts[n].item(), 2),
                        "fires_for": fires_for(examples, L, n),
                        "pushes": push(model, w2[:, n], 3)}
                       for n in order if act[n] > 0]
            layers.append({
                "ln1": {"before": [round(st["x_in"][i].mean().item(), 2), round(st["x_in"][i].std().item(), 2)],
                        "after": [round(st["a_in"][i].mean().item(), 2), round(st["a_in"][i].std().item(), 2)],
                        "vec": r(st["a_in"][i])},
                "heads": heads,
                "attn": r(st["attn"][i]), "attn_push": push(model, st["attn"][i]),
                "attn_lift": lift(st["attn"][i]),
                "guess_attn": guess(model, st["mid"][i]),
                "ff_in": r(st["f_in"][i]),
                "wide": r(act, 2), "firing": int((act > 0).sum()),
                "neurons": neurons,
                "ff": r(st["ff"][i]), "ff_push": push(model, st["ff"][i]), "ff_lift": lift(st["ff"][i]),
                "guess_ff": guess(model, st["out"][i]),
            })
        p = raw["probs"][i]
        top10 = torch.topk(p, 10)
        positions.append({
            "embed": e, "layers": layers, "win": tk(win),
            "logits": [[tk(j), round(raw["logits"][i, j].item(), 2)] for j in top10.indices.tolist()],
            "final": [[tk(j), round(q, 4)] for q, j in zip(top10.values.tolist(), top10.indices.tolist())],
            "rest": round(1 - top10.values.sum().item(), 4),
        })

    return {"label": label, "prompt": prompt, "check": raw["check"],
            "tokens": [{"id": t, "text": tk(t)} for t in ids],
            "n_layers": len(model.layers), "n_heads": n_heads, "positions": positions}


# ---------------------------------------------------------------------------
# PRINTING
# ---------------------------------------------------------------------------

def g(items, pct=True):
    """[['n', 0.84], ...] -> 'n' 84%  'e·' 4%"""
    return "  ".join(f"{t} {v:.0%}" if pct else f"{t} {v:+.2f}" for t, v in items)


def vec(name, v, per_line=16):
    lines = [f"{name:<12}" + " ".join(f"{x:+.2f}" for x in v[:per_line])]
    for s in range(per_line, len(v), per_line):
        lines.append(" " * 12 + " ".join(f"{x:+.2f}" for x in v[s:s + per_line]))
    return "\n".join(lines)


def spark(values):
    """128 neuron firings as one line of blocks: blank = silent, █ = strongest."""
    top = max(values) or 1
    return "".join(BLOCKS[min(8, round(8 * x / top))] for x in values)


def listened(R, i, L, H):
    w = R["positions"][i]["layers"][L]["heads"][H]["weights"]
    j = max(range(len(w)), key=w.__getitem__)
    who = "itself" if j == i else R["tokens"][j]["text"]
    return who, w[j]


def overview_lines(R, i=None):
    """[(stage, text)] for one token: one line per stage, readings only."""
    i = len(R["tokens"]) - 1 if i is None else i
    P = R["positions"][i]
    tok = R["tokens"][i]["text"]
    out = [("tokens", " ".join(t["text"] for t in R["tokens"])
            + f"   ids {[t['id'] for t in R['tokens']]}"),
           ("embedding", f"{tok} = card #{R['tokens'][i]['id']} + seat {i}.  Nearest cards: {g(P['embed']['near'], False)}"),
           ("", f"   if it stopped here: {g(P['embed']['guess'])}")]
    for L, Ly in enumerate(P["layers"]):
        heads = " · ".join(f"h{H} → {w} {s:.2f}" for H, (w, s) in
                           enumerate(listened(R, i, L, H) for H in range(R["n_heads"])))
        out.append((f"L{L} attention", f"{tok} listens: {heads}"))
        out.append(("", f"   if it stopped here: {g(Ly['guess_attn'])}"))
        n = max(Ly["neurons"], key=lambda n: n["lift"], default=None)     # the one helping the winner most
        key = (f"key neuron #{n['n']} ({n['lift']:+.1f} for {P['win']}, fires for {n['fires_for'][0]})"
               if n else "none fire")
        out.append((f"L{L} feed-fwd", f"{Ly['firing']} of 128 neurons fire; {key}"))
        out.append(("", f"   if it stopped here: {g(Ly['guess_ff'])}"))
    out.append(("prediction", g(P["final"][:5]) + f"   (other {556 - 5} tokens share "
                f"{1 - sum(v for _, v in P['final'][:5]):.0%})"))
    return out


def print_header(R):
    print(f'Prompt "{R["prompt"]}"   model: {R["label"]}')
    print(f"Check: this trace and model() agree to within {R['check']:.1e}\n")


def print_overview(R):
    print_header(R)
    for stage, text in overview_lines(R):
        print(f"{stage.upper():<14}{text}")


def print_trace(R, i, layers=None, heads=None):
    P = R["positions"][i]
    T = len(R["tokens"])
    tok = R["tokens"][i]["text"]
    print("=" * 100)
    print(f"TOKEN {i}: {tok}   (id {R['tokens'][i]['id']}){'   <- makes the prediction' if i == T - 1 else ''}")
    print("=" * 100)

    e = P["embed"]
    print("\n--- EMBEDDING: look up the token card and the seat card, add them ---")
    print(vec("token card", e["tok"]))
    print(vec(f"seat {i}", e["pos"]))
    print(vec("sum", e["sum"]))
    print(f"meaning: nearest cards to {tok} (cosine, 1 = same direction): {g(e['near'], False)}")
    print(f"if it stopped here: {g(e['guess'])}")

    for L, Ly in enumerate(P["layers"]):
        if layers is not None and L not in layers:
            continue
        print(f"\n--- LAYER {L} · STEADYING (ln1): mean {Ly['ln1']['before'][0]:+.2f}, spread "
              f"{Ly['ln1']['before'][1]:.2f}  ->  mean {Ly['ln1']['after'][0]:+.2f}, spread "
              f"{Ly['ln1']['after'][1]:.2f} ---")
        print(vec("steadied", Ly["ln1"]["vec"]))
        approx = "" if L == 0 else "   (approx: each token alone, without neighbours)"
        for H, h in enumerate(Ly["heads"]):
            if heads is not None and H not in heads:
                continue
            print(f"\n--- LAYER {L} · HEAD {H} ---{approx}")
            print(vec("query", h["q"]) + f"   looks for: {g(h['looks_for'], False)}")
            print(vec("key", h["k"]) + f"   answers:   {g(h['answers'], False)}")
            print(vec("value", h["v"]) + f"   passes on: {g(h['passes_on']['up'], False)}")
            print("Q·K score (÷√8) and share of attention for each token so far:")
            for j, (s, w) in enumerate(zip(h["scores"], h["weights"])):
                bar = "█" * round(w * 30)
                print(f"   {R['tokens'][j]['text']:>12}  score {s:+6.2f}  ->  {w:5.3f} {bar}")
            for j in range(i + 1, T):
                print(f"   {R['tokens'][j]['text']:>12}  —  future, hidden by the mask")
            print(vec("blend", h["out"]) + "   (the values, mixed by those shares)")
            print(f"this head pushes: up {g(h['out_push']['up'], False)}   down {g(h['out_push']['down'], False)}")
            print(f"for the winner {P['win']}: {h['lift']:+.2f}")
        print(f"\n--- LAYER {L} · STITCH 4 heads x 8 -> 32 (proj), then ADD to the input ---")
        print(vec("change", Ly["attn"]))
        print(f"pushes: up {g(Ly['attn_push']['up'], False)}   down {g(Ly['attn_push']['down'], False)}")
        print(f"for the winner {P['win']}: {Ly['attn_lift']:+.2f}")
        print(f"if it stopped here: {g(Ly['guess_attn'])}")

        print(f"\n--- LAYER {L} · FEED-FORWARD: steady (ln2), widen 32 -> 128, cut negatives, shrink 128 -> 32 ---")
        print(vec("in (32)", Ly["ff_in"]))
        print(f"widened (128 neurons, after cutting negatives): {Ly['firing']} fire")
        print(f"   |{spark(Ly['wide'])}|")
        print(f"the neurons that move the winner {P['win']} most (firing x output):")
        for n in Ly["neurons"]:
            print(f"   #{n['n']:<3} fires {n['act']:4.2f}  -> {n['lift']:+5.2f} for {P['win']}   "
                  f"fires hardest at: {'  '.join(n['fires_for'])}")
            print(f"              pushes: {g(n['pushes']['up'], False)}")
        print(vec("shrunk (32)", Ly["ff"]))
        print(f"pushes: up {g(Ly['ff_push']['up'], False)}   down {g(Ly['ff_push']['down'], False)}")
        print(f"for the winner {P['win']}: {Ly['ff_lift']:+.2f}")
        print(f"if it stopped here: {g(Ly['guess_ff'])}")

    print("\n--- PREDICTION: steady (ln_final), 556 raw scores, softmax -> chances ---")
    print("highest raw scores: " + "  ".join(f"{t} {s:+.2f}" for t, s in P["logits"][:5]))
    for t, p in P["final"]:
        print(f"   {t:>12}  {p:6.1%} {'█' * round(p * 40)}")
    print(f"   {'other 546':>12}  {P['rest']:6.1%}")
    print()


def print_attention_table(R, L, H):
    toks = [t["text"] for t in R["tokens"]]
    print(f"\nLayer {L}, head {H}: share of attention (row = token asking, column = token listened to)")
    print(" " * 12 + "".join(f"{t[:9]:>10}" for t in toks))
    for i, t in enumerate(toks):
        w = R["positions"][i]["layers"][L]["heads"][H]["weights"]
        print(f"{t[:11]:>12}" + "".join(f"{v:10.2f}" for v in w) + "".join(f"{'—':>10}" for _ in toks[i + 1:]))


# ---------------------------------------------------------------------------
# EXTRAS: switch parts off, one training step
# ---------------------------------------------------------------------------

@torch.no_grad()
def switch_off(model, prompt, spec=None):
    ids = encode(prompt, merges)
    base = forward(model, ids)["probs"][-1]
    best = int(base.argmax())

    def effect(off):
        p = forward(model, ids, off)["probs"][-1]
        top = torch.topk(p, 5)
        return {"drop": round((base[best] - p[best]).item(), 4), "best_now": round(p[best].item(), 4),
                "top": [[tk(j), round(q, 4)] for q, j in zip(top.values.tolist(), top.indices.tolist())]}

    def name(off):
        return f"layer {off[1]} head {off[2]}" if off[0] == "head" else f"layer {off[1]} neuron #{off[2]}"

    top = torch.topk(base, 5)
    result = {"best": tk(best), "best_p": round(base[best].item(), 4),
              "base": [[tk(j), round(q, 4)] for q, j in zip(top.values.tolist(), top.indices.tolist())]}
    if spec:
        result["single"] = {"name": name(spec), **effect(spec)}
        return result
    heads = [("head", L, H) for L in range(len(model.layers)) for H in range(len(model.layers[L].attn.heads))]
    result["heads"] = [{"name": name(o), **effect(o)} for o in heads]
    n_ff = model.layers[0].ff.net[0].out_features
    neurons = [{"name": name(o), **effect(o)} for o in
               (("ff", L, n) for L in range(len(model.layers)) for n in range(n_ff))]
    result["neurons"] = sorted(neurons, key=lambda d: -d["drop"])[:8]
    return result


def parse_off(text, model):
    try:
        L, part = text.split(".")
        L = int(L)
        off = ("ff", L, int(part[2:])) if part.startswith("ff") else ("head", L, int(part))
        limit = model.layers[L].ff.net[0].out_features if off[0] == "ff" else len(model.layers[L].attn.heads)
        assert 0 <= off[2] < limit
        return off
    except (ValueError, IndexError, AssertionError):
        sys.exit(f"--off {text!r}: use LAYER.HEAD like 1.2, or LAYER.ffNEURON like 1.ff57 "
                 f"(layers 0-{len(model.layers) - 1}, heads 0-3, neurons 0-127)")


def print_off(res):
    print(f"Normal best guess: {res['best']} {res['best_p']:.1%}   top 5: {g(res['base'])}\n")
    if "single" in res:
        s = res["single"]
        print(f"With {s['name']} switched off:")
        print(f"   {res['best']} goes from {res['best_p']:.1%} to {s['best_now']:.1%}")
        print(f"   top 5 now: {g(s['top'])}")
        return
    print(f"Switch off one head at a time. How much does {res['best']} lose?")
    for h in res["heads"]:
        bar = "█" * max(0, round(h["drop"] * 60))
        print(f"   {h['name']:<18} {res['best_p']:5.1%} -> {h['best_now']:5.1%}  {bar}   now: {g(h['top'][:3])}")
    print(f"\nThe 8 neurons (of 256) whose silence hurts {res['best']} most:")
    for n in res["neurons"]:
        print(f"   {n['name']:<22} {res['best_p']:5.1%} -> {n['best_now']:5.1%}   now: {g(n['top'][:3])}")


FRIENDLY = {"embed.token_table": "token cards", "embed.position_table": "seat cards",
            "ln1": "steadying before attention", "ln2": "steadying before feed-forward",
            "attn.proj": "stitch (proj)", "ff.net.0": "feed-forward widen", "ff.net.2": "feed-forward shrink",
            "ln_final": "final steadying", "head": "prediction head"}


def friendly(name):
    """'layers.0.attn.heads.2.Wq.weight' -> 'layer 0 head 2 query'"""
    parts = name.split(".")
    if parts[0] == "layers":
        rest = ".".join(parts[2:-1])
        if rest.startswith("attn.heads"):
            role = {"Wq": "query", "Wk": "key", "Wv": "value"}[parts[5]]
            return f"layer {parts[1]} head {parts[4]} {role}"
        return f"layer {parts[1]} {FRIENDLY.get(rest, rest)}"
    return FRIENDLY.get(".".join(parts[:-1]), name)


def train_step(model, prompt):
    """One step of train.py's loop on a copy of the model. model.pt is never written."""
    model = copy.deepcopy(model)
    data = torch.load("data.pt")
    train_data = data[: int(0.9 * len(data))]
    T, BATCH, LR = model.max_positions, 32, 3e-3          # train.py's settings
    torch.manual_seed(0)
    starts = torch.randint(0, len(train_data) - T - 1, (BATCH,))
    x = torch.stack([train_data[s: s + T] for s in starts])
    y = torch.stack([train_data[s + 1: s + T + 1] for s in starts])

    prompt_ids = torch.tensor([encode(prompt, merges)])
    before_prompt = model.next_token_probs(prompt_ids)[0]
    before_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    logits, loss = model(x, y)
    p_right = F.softmax(logits[0], dim=-1)[torch.arange(T), y[0]].detach()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    optimizer.zero_grad()
    loss.backward()
    grads = {n: p.grad.norm().item() for n, p in model.named_parameters()}
    table_grad = model.embed.token_table.weight.grad.norm(dim=1)
    optimizer.step()
    model.eval()

    with torch.no_grad():
        loss_after = model(x, y)[1].item()
        after_prompt = model.next_token_probs(prompt_ids)[0]
    moved = {n: (p - before_state[n]).norm().item() for n, p in model.named_parameters()}
    rows = torch.topk(table_grad, 8)

    def top5(p):
        v, i = torch.topk(p, 5)
        return [[tk(j), round(q, 4)] for q, j in zip(v.tolist(), i.tolist())]

    best = int(before_prompt.argmax())
    return {
        "loss": round(loss.item(), 3), "loss_after": round(loss_after, 3),
        "window": [{"in": tk(a), "next": tk(b), "p": round(q, 4)}
                   for a, b, q in zip(x[0].tolist()[:20], y[0].tolist()[:20], p_right.tolist()[:20])],
        "grads": sorted(([friendly(n), round(v, 4), round(moved[n], 4)] for n, v in grads.items()),
                        key=lambda t: -t[1])[:10],
        "cards": [[tk(j), round(v, 4)] for v, j in zip(rows.values.tolist(), rows.indices.tolist())],
        "n_cards": int((table_grad > 0).sum()),
        "before": top5(before_prompt), "after": top5(after_prompt),
        "best": tk(best), "best_before": round(before_prompt[best].item(), 4),
        "best_after": round(after_prompt[best].item(), 4),
    }


def print_train(t, prompt):
    print("One training step, exactly like one turn of train.py's loop (on a copy: model.pt is not changed).")
    print("Batch: 32 random windows of 64 tokens from the training stories (seed 0).\n")
    print(f"1. FORWARD. Loss {t['loss']:.3f}: on average the right next token got "
          f"e^-{t['loss']:.3f} = {2.71828 ** -t['loss']:.1%} of the chance.")
    print("   First window, token by token (the chance it gave to the real next token):")
    for w in t["window"]:
        surprise = -torch.log(torch.tensor(max(w["p"], 1e-9))).item()
        print(f"   {w['in']:>12} -> {w['next']:<12} {w['p']:6.1%}   surprise {surprise:5.2f}  "
              f"{'█' * round(w['p'] * 30)}")
    print("\n2. BACKWARD. Every knob gets a gradient: which way to turn to lower the loss.")
    print("   The 10 groups with the biggest gradient, and how far one AdamW step moved them:")
    for name, gr, mv in t["grads"]:
        print(f"   {name:<34} gradient {gr:7.4f}   moved {mv:7.4f}")
    print(f"\n   Token cards with a gradient: {t['n_cards']} of 556 (only tokens that appear in the batch).")
    print("   The cards pulled hardest: " + "  ".join(f"{c} {v:.3f}" for c, v in t["cards"]))
    print(f"\n3. STEP. Loss on the same batch: {t['loss']:.3f} -> {t['loss_after']:.3f}")
    print(f'   Prompt "{prompt}":  {t["best"]} {t["best_before"]:.1%} -> {t["best_after"]:.1%}')
    print(f"   top 5 before: {g(t['before'])}")
    print(f"   top 5 after:  {g(t['after'])}")
    print("\n   (train.py did not save its optimizer's memory, so this step starts AdamW fresh."
          "\n   A fresh AdamW moves every knob it touches by about the learning rate, 0.003, which is a"
          "\n   bigger jump than a step late in training. The loss is measured on the same batch the step"
          "\n   was made for, so it falls more than it would on new text.)")


# ---------------------------------------------------------------------------
# HTML: the same report, as an interactive page (debug_page.html is the template)
# ---------------------------------------------------------------------------

def write_html(path, page):
    with open("debug_page.html", encoding="utf-8") as f:
        template = f.read()
    blob = json.dumps(page, ensure_ascii=False).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(template.replace("/*DATA*/null", blob))
    print(f"\nwrote {path}  (open it in a browser)")


# ---------------------------------------------------------------------------

def take(args, flag, convert=str):
    if flag not in args:
        return None
    i = args.index(flag)
    value = convert(args[i + 1])
    del args[i:i + 2]
    return value


if __name__ == "__main__":
    args = sys.argv[1:]
    model_path = take(args, "--model") or "model.pt"
    html_path = take(args, "--html")
    token = take(args, "--token", int)
    only_layer = take(args, "--layer", int)
    only_head = take(args, "--head", int)
    off_text = take(args, "--off")
    COMMANDS = ("overview", "trace", "full", "compare", "off", "train-step")
    if not args or args[0] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    cmd = args[0]
    prompt = " ".join(args[1:]) or DEFAULT_PROMPT

    model = load_model(model_path)
    torch.set_grad_enabled(cmd == "train-step")
    page = {"command": cmd, "prompt": prompt, "reports": []}

    def make(m, label):
        return report(m, prompt, label, neuron_examples(m))

    if cmd in ("overview", "trace", "full"):
        R = make(model, model_path)
        page["reports"].append(R)
        T = len(R["tokens"])
        if cmd == "overview":
            print_overview(R)
        else:
            print_header(R)
            layers = None if only_layer is None else [only_layer]
            heads = None if only_head is None else [only_head]
            if cmd == "trace":
                i = T - 1 if token is None else token
                if not 0 <= i < T:
                    sys.exit(f"--token must be 0 to {T - 1}")
                print_trace(R, i, layers, heads)
            else:
                for i in range(T):
                    print_trace(R, i, layers, heads)
                for L in layers or range(R["n_layers"]):
                    for H in heads or range(R["n_heads"]):
                        print_attention_table(R, L, H)

    elif cmd == "compare":
        before = make(load_model("model_untrained.pt"), "model_untrained.pt")
        after = make(model, model_path)
        page["reports"] += [after, before]
        print(f'Prompt "{prompt}": untrained (random knobs) vs {model_path}, stage by stage\n')
        for (stage, a), (_, b) in zip(overview_lines(before), overview_lines(after)):
            if stage:
                print(stage.upper())
            print(f"   untrained  {a.strip()}")
            print(f"   trained    {b.strip()}")

    elif cmd == "off":
        spec = parse_off(off_text, model) if off_text else None
        res = switch_off(model, prompt, spec)
        page["off"] = res
        page["reports"].append(make(model, model_path))
        print(f'Prompt "{prompt}"   model: {model_path}')
        print_off(res)

    elif cmd == "train-step":
        t = train_step(model, prompt)
        page["train"] = t
        print_train(t, prompt)

    if html_path:
        # The page always holds everything for this prompt, whichever command made it.
        if "train" not in page:
            torch.set_grad_enabled(True)
            page["train"] = train_step(model, prompt)
        torch.set_grad_enabled(False)
        labels = [R["label"] for R in page["reports"]]
        if model_path not in labels:
            page["reports"].insert(0, make(model, model_path))
        if "model_untrained.pt" not in labels and model_path != "model_untrained.pt":
            page["reports"].append(make(load_model("model_untrained.pt"), "model_untrained.pt"))
        if "off" not in page or "single" in page["off"]:
            page["off"] = switch_off(model, prompt)
        write_html(html_path, page)
