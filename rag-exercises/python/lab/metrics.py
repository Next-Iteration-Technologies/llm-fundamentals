"""The scores. Every number is calculated per question, then averaged (SPEC "How we measure").

A chunk counts as "the right one" for a gold passage when they share at least half of the smaller of the two
(same file, overlapping positions). One rule for every score.
"""
import math

K_FAIR = 2000   # the fair-size view: every setup gets about this many tokens of text, not the same number of chunks


def overlap(c, g):
    if c["file"] != g.file:
        return 0
    return max(0, min(c["end"], g.end) - max(c["start"], g.start))


def matches(c, g):
    """At least half of the smaller of chunk and gold passage is shared."""
    ov = overlap(c, g)
    return ov > 0 and ov >= 0.5 * min(c["end"] - c["start"], g.end - g.start)


def gold_ranks(ranked, gold):
    """For each gold passage, the rank (1-based) of the first chunk that matches it, or None."""
    out = []
    for g in gold:
        out.append(next((r for r, c in enumerate(ranked, 1) if matches(c, g)), None))
    return out


def recall(ranks, k):
    """Of the gold passages, how many are in the top k? Out of k at most, so a perfect top k always scores 1."""
    if not ranks:
        return None
    # capped at 1: gold passages can overlap, so one chunk may stand for two of them
    return min(1.0, sum(1 for r in ranks if r and r <= k) / min(k, len(ranks)))


def score(ranks, grades, sizes=None):
    """All scores for one question. ranks: per gold passage, its first rank or None; grades: per gold passage (3/2/1);
    sizes: tokens of each ranked chunk, for the fair-size view."""
    if not ranks:
        return {}
    hit = [r for r in ranks if r]
    out = {"R@3": recall(ranks, 3), "R@5": recall(ranks, 5), "R@50": recall(ranks, 50),
           "MRR": 1 / min(hit) if hit else 0.0,
           # every right passage in the top 5 (when there are more than 5, the top 5 all right)
           "all@5": 1.0 if sum(1 for r in ranks if r and r <= 5) >= min(5, len(ranks)) else 0.0}
    dcg = sum(g / math.log2(r + 1) for r, g in zip(ranks, grades) if r and r <= 5)  # gain x 1/log2(rank+1)
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(grades, reverse=True)[:5]))
    out["nDCG@5"] = min(1.0, dcg / ideal) if ideal else 0.0   # two passages in one chunk could pass 1: capped
    if sizes is not None:
        n, total = 0, 0
        for s in sizes:
            if n and total + s > K_FAIR:
                break
            n, total = n + 1, total + s
        out["R@2k"] = sum(1 for r in ranks if r and r <= n) / len(ranks)
        out["chunks@2k"] = n
    return out


NAMES = ["R@3", "R@5", "MRR", "nDCG@5", "all@5", "R@2k", "R@50"]
EXPLAIN = {
    "R@3": "Recall@3: of the right passages, how many were in the top 3? (Counted out of 3 at most: the best 3 chunks can do.)",
    "R@5": "Recall@5: of the right passages, how many were in the top 5, the chunks sent? (Out of 5 at most.)",
    "MRR": "MRR: how high was the first right chunk? 1st = 1.0, 2nd = 0.5, 3rd = 0.33",
    "nDCG@5": "nDCG@5: like MRR, but counts every right chunk in the top 5, weighted by how important it is",
    "all@5": "All found: share of questions where every right passage was in the top 5",
    "R@2k": "Recall@~2k tokens: the same, when every setup sends about 2,000 tokens instead of 5 chunks",
    "R@50": "Recall@50: in the top 50 at all? Rerank can only reorder these",
}


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None
