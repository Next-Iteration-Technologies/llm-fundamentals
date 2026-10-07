"""
Byte-level BPE tokenizer, written from scratch.
Step 1 of the toy language model: tokenizer -> embeddings -> attention -> layers.

Usage:
    python bpe_tokenizer.py stories.txt          # trains on the file, saves tokenizer.json
    python bpe_tokenizer.py stories.txt 500      # optional: number of merges (default 300)

Put a couple of children's stories (plain text, UTF-8) into stories.txt first.
Project Gutenberg's Aesop's Fables or Grimm's Fairy Tales work well.
"""

import json
import re
import sys
from collections import Counter


# Before any merging, the text is cut into pieces: words, numbers, punctuation, spaces.
# Merges only happen inside a piece, so a token never spans two words.
# A space can only sit at the start of a token (" the"), never at the end ("e ").
SPLIT = re.compile(r"""[’'](?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


# ---------------------------------------------------------------------------
# TRAINING: learn the vocabulary and the ordered merge rules
# ---------------------------------------------------------------------------

def get_pair_counts(ids):
    """Count every adjacent pair of token IDs in the sequence."""
    counts = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def apply_merge(ids, pair, new_id):
    """Replace every occurrence of `pair` (a, b) in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text, num_merges=300, verbose=True):
    """
    Learn BPE merges from `text`.

    Starting alphabet = the 256 possible byte values.
    The text is cut into pieces (SPLIT), and each merge glues the most frequent
    adjacent pair *inside a piece* into one new token.
    Returns (merges, vocab):
        merges: ordered list of ((a, b), new_id) - the recipe for encoding later
        vocab:  dict id -> bytes, so we can decode and print tokens
    """
    # Each distinct piece is stored once, with how often it occurs: " the" -> 9000.
    piece_counts = Counter(SPLIT.findall(text))
    pieces = [list(p.encode("utf-8")) for p in piece_counts]   # piece -> UTF-8 bytes -> ints 0..255
    freqs = list(piece_counts.values())
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    for step in range(num_merges):
        counts = Counter()
        for ids, freq in zip(pieces, freqs):
            for pair, n in get_pair_counts(ids).items():
                counts[pair] += n * freq
        if not counts:
            break
        pair, freq = counts.most_common(1)[0]
        if freq < 2:
            break                          # nothing repeats any more; stop early
        new_id = 256 + step
        pieces = [apply_merge(ids, pair, new_id) for ids in pieces]
        merges.append((pair, new_id))
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        if verbose and step < 20:
            print(f"merge {step + 1:3d}: {vocab[pair[0]]!r} + {vocab[pair[1]]!r} "
                  f"-> {vocab[new_id]!r}   (id {new_id}, seen {freq} times)")

    if verbose:
        n_tokens = sum(len(ids) * f for ids, f in zip(pieces, freqs))
        print(f"\nTraining done. Vocabulary size: {len(vocab)} tokens. "
              f"Text went from {len(text.encode('utf-8'))} bytes to {n_tokens} tokens.")
    return merges, vocab


# ---------------------------------------------------------------------------
# USING THE TOKENIZER: encode replays the merges, decode reverses them
# ---------------------------------------------------------------------------

def encode(text, merges):
    """Text -> list of token IDs: cut into pieces, then replay the learned merges on each piece."""
    cache = {}                             # the same word is only worked out once
    ids = []
    for piece in SPLIT.findall(text):
        if piece not in cache:
            piece_ids = list(piece.encode("utf-8"))
            for pair, new_id in merges:
                piece_ids = apply_merge(piece_ids, pair, new_id)
            cache[piece] = piece_ids
        ids.extend(cache[piece])
    return ids


def decode(ids, vocab):
    """List of token IDs -> text, by concatenating each token's bytes."""
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# SAVE / LOAD so the model (next step) can use the same tokenizer
# ---------------------------------------------------------------------------

def save(path, merges, vocab):
    data = {
        "merges": [[list(pair), new_id] for pair, new_id in merges],
        "vocab": {str(i): list(b) for i, b in vocab.items()},   # bytes as int lists
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load(path):
    with open(path) as f:
        data = json.load(f)
    merges = [(tuple(pair), new_id) for pair, new_id in data["merges"]]
    vocab = {int(i): bytes(b) for i, b in data["vocab"].items()}
    return merges, vocab


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source = sys.argv[1]
    num_merges = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    with open(source, encoding="utf-8") as f:
        text = f.read()

    merges, vocab = train(text, num_merges)
    save("tokenizer.json", merges, vocab)

    # Sanity checks
    sample = "Once upon a time there was a little girl."
    ids = encode(sample, merges)
    print("\nSample:   ", sample)
    print("Token IDs:", ids)
    print("Tokens:   ", [decode([i], vocab) for i in ids])
    print("Round trip OK:", decode(ids, vocab) == sample)
