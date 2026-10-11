# Contoso Archive Services: the RAG day data pack

**Everything in this folder is fictional.** Contoso Archive Services, its platforms (Legacy Archive, Content Hub, Capture), its people (Sam, Priya, Lee, Noor, Jonas), its tenants (Fabrikam Legal, Northwind Finance, Tailspin HR, Woodgrove Procurement), vendors A, B and C, every document ID, ticket, log line and price are made up for the workshop. Mail addresses use `.example` domains. Don't replace them with real ones in the room.

The trainer's deck and its browser demos (`corpus.js` in the deck repo) and these exercises all use this pack. Each of the 31 demo chunks (c01 to c31) appears word for word in one file here; `corpus-index.json` says which.

## What's in it

| Folder / file | What it is | Used in flows / questions |
|---|---|---|
| `minutes/` | Four approved POC minutes (2026-09-03 kick-off, 09-17 vendor A demo, 09-24 vendor B demo, 10-01 decision) in one business format: Attendees, Agenda, topic sections, Decisions, Asks, Actions. Access `team, vendor`; the 09-24 **Internal note** section is `team` only (c31). | 1, 4.5, 5, 6.4, 7, 9.1–9.2; "Why did we drop vendor A?" |
| `mail/` | 30 `.eml` files (RFC 822, plain text) and `index.json` (from, to, date, subject, file). The vendor A thread "Re: Re: Re: on-prem parser?" with stacked quoted replies (c09, c10); the vendor B pricing thread (answer 2026-10-07); the forwarded vendor C transcript with a hidden instruction (c11); 10 alert mails (`ALERT ... doc-id ... stage=transfer result=TIMEOUT`); routine team mail. | 2.2 (strip quotes), 6.3 echoing thread, 9.3 injection, F4 alert |
| `contenthub/` | The Content Hub library: SOP-CH-03 *Search issue triage* (c12–c15), SOP-CH-07 *Alert mail handling* (c16), SOP-OPS-02 *Log retention* (c17, says 14 days), SD-LOG-01 log platform design (c19), POL-RET-01 *Archive retention policy* (c26, Public), plus SOP-LA-02, SOP-CAP-01, SD-CH-02 for realism. Each has front matter: id, title, owner, revised, access, class. `config/contenthub-cleanup.xml` keeps logs 30 days (c18): **the drift against SOP-OPS-02**. | 2.3 chunking, 2.6 contextual, 6.8 out of date; "Search returns nothing for one tenant" |
| `onedrive/priya-notes/` | Priya's private notes (`access: priya`). `index-worker-fix.md` (c25) is the only written copy of "clear the worker's temp folder". | 3, 9.2 permissions |
| `known-issues.csv`, `known-issues.xlsx` | One engineer's sheet: 15 rows (KI-014, KI-022, KI-031, KI-040, KI-047 and ten more). Columns ID, Platform, Symptom, Cause, Check, Fix, Added. | 2.2 row-per-chunk, 6.3 split row |
| `known-issues.as-text.txt` | The same rows rendered one row per chunk with column names written in. The five demo rows keep the demos' short form (c20–c24). | 2.2 |
| `logs/transfer.log`, `logs/index.log` | ~60 lines each from server CH-APP-02, night of 2026-10-05/06. Contains c27 (`02:14:07 ... 7f3a-91c2 ... result=TIMEOUT`), c28 and look-alike TIMEOUT lines for other document IDs. | 4.1 keyword vs semantic, 6.2 look-alike logs |
| `jira/issues.json` | Live Jira (mock): epic POC-1 and POC-10..POC-18. POC-12 *Ask vendor B: pricing* is **Done, updated 2026-10-07**. Served by `../jira-mock/server.py` (Python rung) or the `jira-mock` n8n workflow. | 8 live tool |
| `jira/snapshot-2026-10-06.json` | The nightly index snapshot: POC-12 still **Open** (c29). Indexed on purpose to show the stale-status trap. | 6.9, 8.2 |
| `tickets/tickets.json` | 50 tickets (TCK-44xx/45xx): tenant, platform, opened, text, resolution, as_text. Three from Fabrikam Legal about search returning nothing: TCK-4433 (2026-04-14), TCK-4471 (2026-07-09, c30), TCK-4512 (2026-09-21). | 8.1 agentic |
| `golden-set.json` | 18 questions: the five from flow 7.2 plus 13 more, each with expected chunk ids (graded), files, must-say facts and the asker. g05 and g14 must be refused. | 7 evaluation; `rag.py eval` |
| `corpus-index.json` | The 31 demo chunks with their file here, so deck, demos, lab and golden set agree. | all |
| `manifest.json` | Every text file of the pack with a coarse access tag, for an ingest that fetches over HTTP (the hosted n8n rung). Regenerate after adding files. | 3 n8n rung |

Note: the deck and demos cite `priya-notes/index-worker-fix.docx` and `known-issues.xlsx`. In the pack the note is `.md` so it can be read and diffed; on the test tenant upload it as a Word file with the same text (open the `.md` in Word and save as `.docx`).

## Load it into a Microsoft 365 test tenant (Copilot rungs)

Use a **test tenant** or a sandbox site only; this is training data.

1. **SharePoint:** create a team site, document library **"Contoso Archive Ops"**. Upload `contenthub/` (all SOPs, designs, the policy and `config/`) and `minutes/` as folders. Upload `known-issues.xlsx` to the library root. Optionally set the 2026-09-24 minutes to a restricted folder to show permissions.
2. **OneDrive:** sign in as a test user "Priya" and upload `onedrive/priya-notes/` to her OneDrive (as `.docx`, see note above). Do not share it: the point is that Sam's Copilot does not find it.
3. **Mail:** in a test mailbox ("Sam"), import the `.eml` files: in Outlook (classic) drag the `.eml` files into a folder "Contoso POC", or in new Outlook / OWA open each and move it. Alert mails can go to a folder "Alerts".
4. **Copilot chat:** wait for indexing (can take hours), then ask Sam's Copilot the two questions of the day. Note which sources it cites.
5. **Agent builder (Microsoft 365 Copilot):** create an agent "Contoso archive assistant", knowledge = the "Contoso Archive Ops" library (and only it), instructions = the grounding rules from flow 5.2. Ask the golden questions and compare.
6. Jira and the tickets are not in Graph: in Copilot they can only come in through a connector. That gap is the reason for the n8n and Python rungs.

## How the lab uses it

- **Python rung:** `../python/rag.py ingest` reads this folder directly (structure-aware chunking with section-level access, see `../README.md`). Set `CONTOSO_PACK=` if the pack lives elsewhere.
- **Hosted n8n rung:** the workflows cannot read a laptop's disk, so they fetch the pack over HTTP from `PACK_URL`: first `manifest.json` (every text file with a coarse access tag), then each file. The pack is published for that in the public training repo.
- **Jira mock:** `jira/issues.json` is served by `../jira-mock/server.py` for the Python rung and by the `jira-mock` n8n workflow for the hosted rung.
- The optional `../docker-compose.yml` runs n8n and Ollama locally; its workflows fetch the pack the same way.

Nothing in the pack is changed by the lab.
