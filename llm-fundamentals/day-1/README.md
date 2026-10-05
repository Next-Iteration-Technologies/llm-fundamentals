# LLM Fundamentals, day 1: what a model is, and where memory lives

Small Python programs that talk to a language model, one round at a time. After every call they print **what was sent to the model**, so you can see where the answer comes from.

You can run every round in two ways:
- **Live:** your laptop calls a real model. This needs a key, which the trainer will give you.
- **Replay:** the program plays back the trainer's recorded answers. No key needed. Use this if live doesn't work on your laptop.

## Setup

1. Install `uv`, which installs Python and the packages for you:
   - **Windows (PowerShell):** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - **Mac or Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

   Then open a new terminal.
2. Get this folder: `git clone https://github.com/recursivewonder/benz-loda-training.git`, or download the ZIP from GitHub. Then:

   ```bash
   cd benz-loda-training/llm-fundamentals/day-1
   ```
3. Check your laptop:

   ```bash
   uv run check_setup.py
   ```

   The check ends with one of three results:
   - **LIVE:** you can call the model.
   - **REPLAY:** use `--replay`.
   - **WATCH:** follow along on the trainer's screen.

   The first run needs internet access, because `uv` downloads Python and the packages.

**To go live** (optional), copy the two example files:
- `providers.example.toml` → `providers.toml`. Fill in the block the trainer names.
- `.env.example` → `.env`. Put your key there.

Then run `uv run check_setup.py` again. `.env` stays on your laptop. Never commit it or paste the key anywhere.

## The rounds

| Round | Run live | Run as replay | What it shows |
|---|---|---|---|
| R0 | `uv run r0_continue_text.py` | `uv run r0_continue_text.py --replay` | A plain text goes to a base model, with no chat around it |
| R1 | `uv run r1_chat_stateless.py` | `uv run r1_chat_stateless.py --replay` | A chat that sends only your new question |
| R2 | `uv run r2_chat_with_history.py` | `uv run r2_chat_with_history.py --replay` | A chat that sends the whole conversation |
| R2, long | `uv run r2_chat_with_history.py --long --demo` | `uv run r2_chat_with_history.py --long --replay` | 20 turns, then what they cost |
| R2b | `uv run r2b_chat_keep_last.py --demo` | `uv run r2b_chat_keep_last.py --replay` | A chat that sends only the last 5 messages |

Without `--demo` you type the lines yourself; with `--demo` the program types the prepared lines for you. Type `exit` to stop.

R0 needs a local base model (Ollama), so most of you will run it with `--replay`.

## The printout

After every call you see what went to the model:

```
--- what we sent to the model | call 2 | claude-opus-5-5 ---
system prompt   (none)
conversation    3 messages
   [0] user       My name is Priya and I look after invoices.
   [1] assistant  Nice to meet you, Priya! ...
   [2] user       What is my name?
input tokens    58
```

Read it every time:
- Which messages were sent?
- Which were not?
- How many tokens did the request use?

## Exercise: break the memory

1. Open `r2_chat_with_history.py` and find the line marked `THE MEMORY`.
2. Change `chat_history` to `chat_history[-1:]`. Now only the newest message is sent.
3. Run it again, live or with `--replay`. What happens, and what does the printout look like now?
4. Change it back.

## Files

| File | What it is |
|---|---|
| `r0_…`, `r1_…`, `r2_…`, `r2b_…` | The rounds. Open these |
| `request_view.py` | The printout |
| `llm_client.py` | Which model is used, and how recording and replay work. You don't need to open it |
| `check_setup.py` | The laptop check |
| `providers.example.toml` | Model settings (Anthropic, an OpenAI-compatible service, Ollama) |
| `recordings/` | The trainer's recorded answers, used by `--replay` |

Every round also accepts `--provider NAME`, which picks a block from `providers.toml`.

Replay finds an answer by the exact request. If you change a round so that it sends something new, replay says there is no recording for it. That is expected. Run it live to see the answer.
