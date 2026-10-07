"""
Rebuild stories.txt from Project Gutenberg.

    uv run build_stories.py              # downloads the three books, writes stories.txt

What goes in (licence headers and footers are cut off):
    Aesop's Fables            pg21     lines 877-6152
    The Tale of Peter Rabbit  pg14838  lines 65-252, [Illustration] lines removed
    Grimms' Fairy Tales       pg2591   from line 125, cut at the last story title
                                       that keeps the whole file under 450,000 bytes
The books are downloaded into memory only; nothing else is written.
"""

import re
import urllib.request

BOOKS = {  # Gutenberg id -> (first line, last line), 1-based and inclusive
    21: (877, 6152),
    14838: (65, 252),
    2591: (125, 9241),
}
BUDGET = 450_000


def fetch_lines(book_id, first, last):
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    with urllib.request.urlopen(url) as r:
        text = r.read().decode("utf-8-sig")
    lines = text.replace("\r\n", "\n").split("\n")[first - 1:last]
    return [l for l in lines if l.strip() != "[Illustration]"]


def clean(lines):
    """Join lines and squeeze runs of blank lines down to one."""
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


if __name__ == "__main__":
    aesop, peter, grimm = (clean(fetch_lines(b, *rng)) for b, rng in BOOKS.items())
    head = aesop + "\n" + peter + "\n"
    budget = BUDGET - len(head.encode())

    # Grimm story titles are all-caps lines; cut at the last one that fits the budget.
    starts = [m.start() for m in re.finditer(r"(?m)^[A-Z][A-Z ,’'\-\[\].]+$\n\n", grimm)]
    if len(grimm.encode()) > budget:
        cut = max(s for s in starts if len(grimm[:s].encode()) <= budget)
    else:
        cut = len(grimm)

    with open("stories.txt", "w", encoding="utf-8") as f:
        f.write(head + grimm[:cut].rstrip() + "\n")
    print("Grimm stops before:", grimm[cut:cut + 60].splitlines()[0] if cut < len(grimm) else "(all)")
