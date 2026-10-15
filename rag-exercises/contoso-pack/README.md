# Contoso Water: the RAG day data pack

**Everything in this folder is fictional.** Contoso Water Ltd, its people (Maria, Priya, Emma, Jonas, Sam, Noor, Lee),
its clients (Fabrikam, Northwind, Tailspin, Woodgrove and a few small ones), its suppliers (ClearSpring, BlueRiver,
Alpenquell), every order, ticket, price and policy are made up for the workshop. Mail addresses use `.example` domains.
Don't replace them with real ones in the room.

## The company in one paragraph

Contoso Water rents water coolers to offices and refills them. A bottle cooler has a 19-litre bottle on top that a driver
swaps every week; a plumbed-in cooler is connected to the office's water pipe and has a filter. About 120 people work
in three offices: London (head office), Utrecht (Netherlands) and Cologne (Germany). The pack is what their staff would
search: the HR handbook, the client onboarding rules, the service rules, meeting minutes, mail, a price list, the
delivery log, service tickets and Jira.

The big story of autumn 2026: **Northwind**, a new client in Utrecht, is being onboarded (7 coolers, 36 months), and
Contoso needs a supplier for the extra bottles in the Netherlands.

## What's in it

| Folder / file | What it is | Lab traps |
|---|---|---|
| `policies/HR-*` | The HR handbook: annual leave per country (UK, Netherlands, Germany), sick leave, remote work, hours, expenses, parental leave, code of conduct, reviews, joining, leaving, pay, salary bands (HR only, Confidential), health and safety, laptops. Front matter: id, title, owner, revised, access, class, country. | Three leave policies that look alike (country filter, g17). Salary bands only for HR (g41). |
| `policies/archive/HR-REM-01_2024.md` | The **old** remote-work policy, still in the library: up to **2 days** a week. The current `HR-REM-01` (2026-03-01) says **3 days**. | Two versions disagree (prefer-newest, after-2026, g10). |
| `policies/CL-*` | Client onboarding: the steps, documents, credit checks, pricing and discount approval, site survey, welcome call, client data (Public), ending a contract. | |
| `policies/SV-*` | Service: delivery areas and times, service levels, complaints and credits (these three Public, readable by clients), missed deliveries, hygiene, suppliers, installing coolers. | |
| `minutes/` | Four approved Northwind onboarding minutes (2026-09-03 kick-off, 09-17 site survey and ClearSpring, 09-24 pricing and BlueRiver, 10-01 supplier decision). Access `team, client`; the 09-24 **Internal note** section (the discount limit) is `team` only. | Decision in the minutes, reason in a mail (g01). Section-level permission (g14). |
| `onedrive/hr-notes/` | Priya's private notes (`access: hr`). `payslip-fix.md` is the only written copy of the quick payslip fix. | Private document (g13). |
| `mail/` | 37 `.eml` files and `index.json`. The ClearSpring thread "Re: Re: Re: Bottles for Utrecht?" with stacked quoted replies; the BlueRiver price thread (answered 2026-10-07); the forwarded Alpenquell call notes with a **hidden instruction**; 7 "MISSED delivery" notifications; routine staff mail. | Quoted replies (E1), injection (g15). |
| `price-list.csv`, `price-list.xlsx` | 12 products and services, WC-100 … WC-470, prices in GBP and EUR. | Row per chunk (E1); bare id WC-300 (g08) vs "fizzy water" (g09). |
| `logs/deliveries.log` | One line per delivery stop, 5 and 6 October 2026. Holds `ORD-7F3A` (Fabrikam, MISSED, no-access) and a look-alike `ORD-7F3B`. Confidential. | Keyword vs semantic (g03, g07). |
| `tickets/tickets.json` | 50 service tickets (TKT-54xx/55xx): client, country, opened, text, resolution. Fabrikam's taste complaints: TKT-5433, TKT-5471 ("smells funny"), TKT-5512. | Vocabulary gap (g06), agent with find_tickets (g12). |
| `jira/issues.json` | Live Jira (mock), project ONB: the Northwind onboarding. ONB-15 *Ask BlueRiver for the price at 550 bottles a week* is **Done, updated 2026-10-07**. Served by `../jira-mock/server.py` or the `jira-mock` n8n workflow. | Live tool (E5). |
| `jira/snapshot-2026-10-06.json` | The nightly index snapshot: ONB-15 still **Open**. Indexed on purpose. | Stale index (g11). |
| `golden-set.json` | 42 questions with graded expected chunk ids, files, must-say and must-not-say facts and the asker (employee, hr, client). g05, g14, g40, g41, g42 must be refused. | LAB chapter 2 onwards; `rag.py eval`. |
| `corpus-index.json` | The 51 marked passages (c01 … c51) with their file and quote, so lab, golden set and any demo agree. | all |
| `manifest.json` | Every text file of the pack with a coarse access tag, for an ingest that fetches over HTTP (the hosted n8n rung). Regenerate after adding files. | n8n rung |

Roles (who asks): **employee** sees `team` documents; **hr** sees `team` and `hr`; **client** sees only what is shared
with clients (`client`): the Public client and service policies, the price list and the Northwind minutes without
their internal note.

Data classes: **Public** (price list, client data policy, delivery, service levels, complaints), **Internal** (most
policies, minutes, mail, tickets, Jira), **Confidential** (salary bands, Priya's notes, the delivery log with
addresses).

## Load it into a Microsoft 365 test tenant (Copilot rungs)

Use a **test tenant** or a sandbox site only; this is training data.

1. **SharePoint:** create a team site, document library **"Contoso Water Handbook"**. Upload `policies/` (with
   `archive/`) and `minutes/` as folders, and `price-list.xlsx` to the library root. Optionally put the 2026-09-24
   minutes in a restricted folder to show permissions.
2. **OneDrive:** sign in as a test user "Priya" and upload `onedrive/hr-notes/` (open each `.md` in Word and save as
   `.docx`). Do not share it: the point is that Sam's Copilot does not find it.
3. **Mail:** in a test mailbox ("Sam"), import the `.eml` files: in Outlook (classic) drag them into a folder
   "Northwind", or in new Outlook / OWA open each and move it.
4. **Copilot chat:** wait for indexing (can take hours), then ask Sam's Copilot the questions of the day. Note which
   sources it cites.
5. **Agent builder (Microsoft 365 Copilot):** create an agent "Contoso Water assistant", knowledge = the "Contoso Water
   Handbook" library (and only it), instructions = the grounding rules from flow 5.2. Ask the golden questions and
   compare.
6. Jira and the tickets are not in Graph: in Copilot they can only come in through a connector. That gap is the reason
   for the n8n and Python rungs.

## How the lab uses it

- **Python rung:** `../python/rag.py ingest` reads this folder directly (structure-aware chunking with section-level
  access, see `../README.md`). Set `CONTOSO_PACK=` if the pack lives elsewhere.
- **Hosted n8n rung:** the workflows cannot read a laptop's disk, so they fetch the pack over HTTP from `PACK_URL`:
  first `manifest.json`, then each file.
- **Jira mock:** `jira/issues.json` is served by `../jira-mock/server.py` for the Python rung and by the `jira-mock`
  n8n workflow for the hosted rung.

Nothing in the pack is changed by the lab.
