"""
Pattern 03 · Parallel Fan-Out

DAS scenario (matches the slide): Priya asks to find invoice 4711, an older
record that might be filed under any of three archives. A dispatcher fans
out to all three at once; total latency is the slowest one, not the sum.

    Dispatcher -> SQL Archive, Document Store, Email Archive (parallel)
               -> Merger deduplicates, returns the best match

Run it:
    uv run labs/03_parallel_fanout.py
"""

import time
from concurrent.futures import ThreadPoolExecutor

import shared


def sql_archive(doc_id: str) -> dict | None:
    shared.step("SQL Archive", f"doc_id={doc_id} (structured DB)")
    return shared.lookup_invoice(doc_id)


def document_store(doc_id: str) -> dict | None:
    shared.step("Document Store", f"doc_id={doc_id} (scanned PDFs, slower)")
    time.sleep(0.5)  # scanned documents take longer to search
    record = shared.lookup_invoice(doc_id)
    return record if record and record.get("platform") == "Classic" else None


def email_archive(doc_id: str) -> dict | None:
    shared.step("Email Archive", f"doc_id={doc_id} (invoice emails)")
    return None  # this invoice was never emailed


def merge(results: list[dict | None]) -> dict | None:
    shared.step("Merger", "deduplicating results from all three archives")
    found = [record for record in results if record is not None]
    return found[0] if found else None


def dispatcher(doc_id: str) -> dict | None:
    shared.step("Dispatcher", f"fanning out doc_id={doc_id} to three archives")
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda fn: fn(doc_id), [sql_archive, document_store, email_archive]))
    return merge(results)


def main() -> None:
    record = dispatcher(doc_id="4711")
    if record is None:
        print("\nPriya sees: No match found in any archive.")
        return
    answer = shared.ask(
        "Write one short, friendly sentence for a Finance user summarising "
        "this archived document record, naming which platform it came from.",
        str(record),
    )
    print(f"\nPriya sees: {answer}")


if __name__ == "__main__":
    main()
