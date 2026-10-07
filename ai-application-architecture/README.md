# AI Application Architecture: the LoDA chatbot, one step at a time

Small Python programs that build a chatbot for the LoDA archive. It answers the basic questions department users ask every day ("Where is invoice 1234?", "Did my upload get archived?") and hands everything else to the developer team.

Every step is one file. Each file imports the previous step and adds one thing, so you can open two neighbouring files and see exactly what changed. After every call to the model, the program prints **what was sent**, so you can see where the answer comes from.

You can run every step in two ways:
- **Live:** your laptop calls a real model. This needs a key, which the trainer will give you.
- **Replay:** the program plays back the trainer's recorded answers. No key needed.

## Setup

1. Install `uv`, which installs Python and the packages for you:
   - **Windows (PowerShell):** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - **Mac or Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

   Then open a new terminal.
2. Get the code and switch to this folder:

   ```bash
   git clone https://github.com/Next-Iteration-Technologies/llm-fundamentals.git
   cd ai-application-architecture
   ```
3. Check your laptop:

   ```bash
   uv run check_setup.py
   ```

   It ends with **LIVE** (you can call the model), **REPLAY** (add `--replay`) or **WATCH** (follow the trainer's screen).

**To go live**, copy `providers.example.toml` to `providers.toml` and fill in the block the trainer names, then copy `.env.example` to `.env` and put your key there. `.env` stays on your laptop: never commit it.

## The steps

| Step | Run | What it adds |
|---|---|---|
| 1. A bare model call | `uv run step1_bare_model_call.py --demo` | A system prompt and one call. The model can only guess |
| 2. Agent loop and tools | `uv run step2_agent_loop.py --demo` | Five archive tools and the loop that runs them |
| 3. Retrieval with a sample file | `uv run step3_retrieval.py --demo` | A sixth tool that searches `data/known_issues.json` |

Add `--replay` to play back the trainer's answers instead of calling a model. Without `--demo` you type the questions yourself; type `exit` to stop.

## Browser UI

The same steps 1 to 3, with a chat window on the left and the request/tool-call printout on the right, live, instead of reading the terminal.

```bash
uv run web_app.py
```

Then open `http://127.0.0.1:8000`. Use the tabs at the top to switch steps; each keeps its own conversation. "Reset" clears the current step's conversation.

## What each step teaches

**Step 1.** Ask *"Hi, I'm Priya. I can't find my invoice 1234."* The answer sounds helpful but nothing was checked. Ask *"Is it in the 2024 folder?"*: the printout shows one message. The model has forgotten the invoice.

**Step 2.** The same question now runs `search_document`. In the printout, find the model's request (`-> please run ...`) and the result we sent back. Count the calls for one question. Who ran the tool, the model or this laptop?

**Step 3.** Ask *"My upload failed, it says the file is too large."* The answer quotes KI-031 from the sample file. Then describe a problem no known issue covers: the model hands it over instead of guessing.

## Exercises

1. **Bad descriptions.** In `step2_agent_loop.py`, change the description of `check_upload_status` to `"Upload stuff."` Ask about yesterday's upload. Does the model still pick the right tool? Change it back.
2. **The brake.** Set `MAX_CALLS_PER_QUESTION = 1` and ask a question that needs two tools. What does the user see?
3. **A known issue of your own.** Add an entry to `data/known_issues.json` for a problem your team sees often, then ask about it in step 3.

## Files

| File | What it is | Colour |
|---|---|---|
| `step1_…`, `step2_…`, `step3_…` | The steps. Open these | |
| `archive.py` | The mock archive: CSP, Classic and ServiceNow, played by `data/` | Blue |
| `data/` | Documents, uploads, users and known issues. All invented | |
| `request_view.py` | The printout | Blue |
| `llm_client.py` | Which model is used, and how recording and replay work | Blue |
| `check_setup.py` | The laptop check | |
| `tests/` | Tests that run without a model: `uv run pytest` | |
| `web_app.py`, `web/`, `web_static/` | The browser UI: `uv run web_app.py` | |

Blue is plain software engineering; the green parts are the system prompt, the tool descriptions and the model's decisions.

The next steps (conversation store, MCP, authentication, authorization, guardrails, model gateway, PII redaction, logging) are on the branch `ai-application-architecture-handson`.
