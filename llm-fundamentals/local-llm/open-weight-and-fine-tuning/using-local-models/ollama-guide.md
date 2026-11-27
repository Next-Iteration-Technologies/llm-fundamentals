# Ollama guide

Ollama is a model runner and a local HTTP server. It downloads open-weight models, loads them into memory, and answers requests on port `11434`. Agent tools such as Claude Code, OpenCode, or a short Python loop sit on top of it and call it as their model endpoint.

Based on November 2026 internet data, checked against Ollama v0.40.18 and the official docs.

This page covers install, first chat, model tags, APIs, loading and unloading models from RAM, serving a team, and shutting the server down. For Claude Code, Codex, Hermes Agent, and the other official launch targets, see [Chat vs agent harnesses](chat-vs-agent-harnesses.md). For the GUI alternative, see [LM Studio](lm-studio-guide.md). For sizing a model to your machine, see [Hardware and VRAM](hardware-and-vram-guide.md).

## Quick answers

- **Is Ollama an agent?** No. It serves models. The agent is the program that loops, calls tools, and edits files.
- **Can Ollama edit files, run terminal commands, or read PDFs on its own?** No. It takes text or images in and sends text or tool-call requests back. Your script or agent does the work on disk, then sends the result back.
- **How do you use different models?** `ollama pull qwen3.5:9b`, then `ollama run qwen3.5:9b`, or pass a different `"model"` name in the API. Idle models unload after 5 minutes by default.
- **Can you run this for a client that forbids external LLMs?** Yes. Local models stay on your laptop or internal server. Set `OLLAMA_NO_CLOUD=1` so cloud models and cloud web search stay off.
- **Can you point tools at an existing server by URL?** Yes. Set `OLLAMA_HOST=http://server-ip:11434`, or pass that URL to the Python client or the OpenAI SDK (`base_url="http://server-ip:11434/v1"`). Put a reverse proxy or VPN in front. Ollama has no built-in login check.

## What Ollama is

Think of Ollama as a database server, except it serves language models instead of tables. Three layers are easy to mix up:

1. **The model file.** A large file of learned weights, for example `qwen3.5:9b`. On disk it sits idle until a runner loads it. Given some input text, it predicts the next tokens.
2. **The runner and server (Ollama).** The engine that loads those weights onto your GPU or CPU, manages the conversation cache, and listens for HTTP requests.
3. **The agent (your app or coding tool).** A program that wraps the model in a loop. It hands the model a list of tools, runs whatever tool the model asks for, feeds the output back, and repeats until the task is finished.

If you ask Ollama to "fix the bug in `app.py`", nothing changes on disk. The model can print code you copy, or it can return a structured request to call `read_file`. Something has to sit between you and Ollama to execute that request.

What turns a chat model into a working agent:

- **Tool calling.** You send a JSON list of functions. Models such as `qwen3.5`, `qwen3.8`, `gemma4`, and `gpt-oss` can emit structured tool-call requests.
- **An execution loop.** Your code, or a tool such as Claude Code, keeps calling the model, running tools, and appending results.
- **Real tools.** Small functions that read files, apply diffs, run shell commands, extract text from PDFs, or query a database.
- **Memory.** Short-term memory is the message list you resend on every call. Longer memory comes from files or from searching documents with embeddings (RAG).

## Install and first run

=== "macOS"

    Download `Ollama.dmg` from [ollama.com](https://ollama.com), drag `Ollama.app` into Applications, and open it once. macOS Sonoma (v14) or newer is required. Apple Silicon (M1 to M5) uses Metal and MLX. Intel Macs run on CPU only. On first launch, Ollama asks to add the `ollama` command to `/usr/local/bin`.

    The desktop app puts an icon in the menu bar, opens a local chat window, and keeps the HTTP server running at `http://localhost:11434` while the app is open.

=== "Linux"

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

    This installs the binary and a `systemd` service named `ollama`. NVIDIA (CUDA driver 550+, or 570+ for older Maxwell and Pascal cards), AMD ROCm v7, and Vulkan are supported. For headless or air-gapped machines, Ollama also publishes `.tar.zst` archives on GitHub Releases.

=== "Windows"

    Download and run `OllamaSetup.exe` from [ollama.com](https://ollama.com). Windows 10 22H2 or newer is required. The installer goes into your user folder without Admin rights and adds `ollama` to your user `PATH`. NVIDIA (driver 551.61+), AMD ROCm v7, and Vulkan are supported. The tray app keeps the local server running while it is open.

The desktop app and the `ollama` CLI talk to the same engine and share the same model files.

### First run

```bash
ollama -v
curl http://localhost:11434/api/version
ollama pull qwen3.5:4b
ollama run qwen3.5:4b
```

The version check should return a short JSON body with a `version` field. If `curl` fails, start the desktop app, or run `ollama serve` in a terminal.

At the `>>>` prompt, ask something short, for example `Explain what an API gateway does in three short sentences.` Type `/bye` or press `Ctrl+D` to leave. Then run `ollama ps` to confirm the model is loaded, and to see whether it is on the GPU or the CPU.

### Where models are stored

Models take several gigabytes each. Defaults:

=== "macOS"

    `~/.ollama/models`

=== "Linux"

    `/usr/share/ollama/.ollama/models` when the `systemd` service is running, or `~/.ollama/models` when you start `ollama serve` as your own user.

=== "Windows"

    `%HOMEPATH%\.ollama\models`, which is `C:\Users\<you>\.ollama\models`.

To move storage to a larger drive, set `OLLAMA_MODELS` to the new folder and restart Ollama. On Linux with `systemd`, the `ollama` user must own that folder: `sudo chown -R ollama:ollama /path/to/models`.

## Using different models

The public catalog is at [ollama.com/library](https://ollama.com/library). A name looks like `family:tag`, for example `qwen3.5:9b` or `gemma4:12b-mlx`.

- **Family.** The model line, such as `qwen3.5`, `qwen3.8`, `gemma4`, `gpt-oss`, or `embeddinggemma-2`.
- **Parameter count.** `4b`, `9b`, or `27b` means 4, 9, or 27 billion weights. Larger models usually reason and code better. They also run slower and need more memory.
- **`:latest`.** If you skip the tag, Ollama pulls `:latest`. For `qwen3.5` that is `9b`. For `qwen3.8` that is `27b`. Write the exact tag in scripts so everyone gets the same file.
- **Quantisation.** Raw 16-bit weights make a 9B model about 18 GB on disk. 4-bit or 8-bit quantisation shrinks the file so it fits on a laptop, with a small quality trade-off. Common tags include `Q4_K_M` (standard 4-bit GGUF), `q8_0` (8-bit), `MXFP4` (used by `gpt-oss`), and `-mlx` or `NVFP4` builds for Apple Silicon and newer NVIDIA cards.

A rough sizing rule: a 4-bit model takes about 0.6 GB per billion parameters on disk, plus extra memory for the KV cache, which grows with prompt length and chat history. Walk the numbers in the [VRAM guide](hardware-and-vram-guide.md) before you pull a large model.

### Core CLI commands

| Command | What it does | Example |
| --- | --- | --- |
| `ollama pull` | Downloads a model without starting a chat. | `ollama pull qwen3.5:9b` |
| `ollama run` | Downloads if needed, then starts a terminal chat, or runs one prompt. | `ollama run qwen3.5:9b "Summarise REST vs GraphQL"` |
| `ollama list` (or `ls`) | Lists models on disk, with size and update time. | `ollama ls` |
| `ollama ps` | Shows models loaded in RAM or VRAM, GPU vs CPU split, context size, and the unload timer. | `ollama ps` |
| `ollama show` | Architecture, context length, capabilities (`tools`, `vision`, `thinking`), or the Modelfile. | `ollama show qwen3.8:27b` |
| `ollama stop` | Unloads a model from memory right away. | `ollama stop qwen3.8:27b` |
| `ollama rm` | Deletes a downloaded model from disk. | `ollama rm qwen3.5:4b` |
| `ollama cp` | Creates an alias that points at the same weights. Instant, no extra disk. | `ollama cp qwen3.5:9b gpt-4o` |

Inside `ollama run`, useful slash commands include `/set parameter num_ctx 32768`, `/show info`, `/clear` (wipes history), and `/bye`. Wrap multiline prompts in triple quotes (`"""`).

### Picking a model by RAM

On Apple Silicon, macOS shares unified memory between the CPU and GPU. The GPU typically gets about 65% to 75% of total RAM. A 16 GB Mac has about 10 to 11 GB for the model plus its context cache. That is why the VRAM calculator in this workshop shows 12 GB usable out of 16 GB total on an M4.

| Machine memory | Default pick | File size | Notes |
| --- | --- | --- | --- |
| 8 GB | `qwen3.5:4b` | 3.3 to 4.0 GB | `qwen3.5:2b` if you keep a heavy browser open. |
| 16 GB | `qwen3.5:9b` | 6.6 to 7.6 GB | `gemma4:12b` also fits. Avoid `gpt-oss:20b` (14 GB) on a 16 GB Mac. |
| 32 GB | `qwen3.8:27b` | about 18 GB | `gemma4:26b` or `gemma4:31b` are close alternatives. |
| 64 GB+ | `qwen3.8:27b` with a long context | 18 GB plus cache | `gpt-oss:120b` (65 GB) wants a 128 GB machine. |
| Embeddings | `embeddinggemma-2:270m` | 304 to 378 MB | Use this for the PDF example in this folder. |

These families were listed in the public library in late November 2026, and they use the Apache 2.0 licence.

The runnable examples and the fine-tune demo default to `qwen3.5:0.8b` so a live walkthrough stays fast. Swap the tag to `qwen3.5:4b` when you want a stronger reply and still have room on a 16 GB laptop.

### Cloud models, and turning cloud off

Ollama also offers `:cloud` tags that run on Ollama's hosted servers after `ollama signin` or an `OLLAMA_API_KEY`. Those calls leave your network.

To keep a private client offline:

- Set `OLLAMA_NO_CLOUD=1`, or
- Put `{"disable_ollama_cloud": true}` in `~/.ollama/server.json` and restart Ollama.

The server log then prints `Ollama cloud disabled: true`.

## Loading and unloading models from RAM

A pulled model sits on disk until something calls it. The first request loads the weights into RAM (and VRAM, or unified memory on Apple Silicon). That load can take several seconds and several gigabytes. After the reply, Ollama keeps the model warm so the next prompt is fast, then unloads it when the idle timer expires.

This matters on a 16 GB laptop. If you leave a 9B or 27B model loaded, Chrome, your IDE, and the model will fight for the same memory.

### See what is loaded: `ollama ps`

```bash
ollama ps
```

A typical row looks like this:

```text
NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL
qwen3.5:9b     3d1b0f8a1c2d    7.2 GB   100% GPU     4096       4 minutes from now
```

- **PROCESSOR.** `100% GPU` is what you want. `CPU`, or a split such as `45%/55% CPU/GPU`, means the model spilled onto system RAM and replies will crawl.
- **CONTEXT.** The context window currently reserved for this model. If a coding agent ignores tools, check this first. A 4K window will truncate a long tool list.
- **UNTIL.** When the idle timer will unload the model. After a fresh request this is usually about 5 minutes.

You can also call `GET http://localhost:11434/api/ps`.

### Unload a model right away: `ollama stop`

```bash
ollama stop qwen3.5:9b
ollama ps
```

After `stop`, `ollama ps` should show an empty list. The model stays on disk (`ollama ls` still lists it). Only the RAM is freed. The next `ollama run` or API call will load it again.

To free memory after a demo, stop every loaded model:

```bash
ollama ps
# then, for each NAME:
ollama stop <model>
```

### The keep-alive timer

The default idle timeout is **5 minutes**. You can change it per request, or as a server default.

On `/api/generate` and `/api/chat` (and the matching Python client), pass `keep_alive`:

| Value | Meaning |
| --- | --- |
| omitted | Use the server default (5 minutes, unless you changed `OLLAMA_KEEP_ALIVE`). |
| `"5m"`, `"30m"`, `"1h"` | Stay loaded for that duration after the response. |
| `60` (a number) | Stay loaded for that many seconds. |
| `-1` or a negative duration | Stay loaded until you stop it, or the process exits. |
| `0` or `"0"` | Unload as soon as this response finishes. |

Unload after one reply:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:9b",
  "messages": [{"role": "user", "content": "Ping"}],
  "stream": false,
  "keep_alive": 0
}'
```

Pin a model for a workshop or a shared server:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3.5:9b",
  "keep_alive": -1
}'
```

An empty generate or `ollama run qwen3.5:9b ""` is the official way to preload a model without producing a long reply, so the first live prompt has no load delay.

A per-request `keep_alive` overrides the server default for that call only.

### Server defaults: `OLLAMA_KEEP_ALIVE` and `OLLAMA_MAX_LOADED_MODELS`

| Variable | Default | What it does |
| --- | --- | --- |
| `OLLAMA_KEEP_ALIVE` | `5m` | How long an idle model stays loaded. Use `30m` or `-1` on a dedicated box that serves one team model. |
| `OLLAMA_MAX_LOADED_MODELS` | `3 * number of GPUs`, or `3` on CPU | How many models may stay loaded at once, if they all fit. Set `1` or `2` on a tight laptop so Ollama does not try to keep three models warm. |

When memory is tight, Ollama waits for in-flight requests to finish, unloads an idle model, and then loads the one you asked for.

Set these the same way you set `OLLAMA_HOST`. The new value applies after a restart.

=== "macOS (desktop app)"

    ```bash
    launchctl setenv OLLAMA_KEEP_ALIVE 5m
    launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
    ```

    Then quit Ollama from the menu bar and reopen the app. `launchctl setenv` lasts until logout. For a permanent value, use a LaunchAgent, or set the variable in the environment and relaunch.

=== "Linux (systemd)"

    ```bash
    sudo systemctl edit ollama.service
    ```

    Add:

    ```ini
    [Service]
    Environment="OLLAMA_KEEP_ALIVE=5m"
    Environment="OLLAMA_MAX_LOADED_MODELS=2"
    ```

    Then:

    ```bash
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    ```

=== "Windows"

    Quit Ollama from the system tray. Open **Settings > System > About > Advanced system settings > Environment Variables**, add the variables under your user account, and relaunch the app.

=== "Foreground `ollama serve`"

    Export the variables in the same shell before you start the server:

    ```bash
    export OLLAMA_KEEP_ALIVE=5m
    export OLLAMA_MAX_LOADED_MODELS=1
    ollama serve
    ```

### Confirm that memory is free again

1. Run `ollama ps`. The list should be empty.
2. On a Mac, open Activity Monitor and check the `Ollama` process. Memory should drop after the unload (it will not go to zero; the server process itself stays running).
3. On NVIDIA Linux, `nvidia-smi` should show the process using much less VRAM once the model is gone.
4. On Windows, Task Manager will show the same drop for the Ollama process.

If `ollama ps` is empty and the process is still holding a large allocation, quit the server completely (next section) and start it again.

### Quit or stop the server

Unloading a model frees the model weights. The Ollama process itself keeps running until you quit it.

=== "macOS"

    Click the llama icon in the menu bar and choose **Quit**. If it relaunches at login, remove it from **System Settings > General > Login Items**. After it quits, `curl http://localhost:11434/api/version` should fail, and the memory should return to the system.

=== "Linux"

    If the install script set up the service:

    ```bash
    sudo systemctl stop ollama
    ```

    If you started it yourself, press `Ctrl+C` in the `ollama serve` terminal, or `pkill ollama` if you know that is the only Ollama process you have running. Confirm with `curl http://localhost:11434/api/version` and `systemctl status ollama`.

=== "Windows"

    Right-click the tray icon and quit. If it starts again at login, turn it off under **Settings > Apps > Startup**. After it exits, `curl http://localhost:11434/api/version` should fail.

Starting the app, or running `ollama serve` / `sudo systemctl start ollama`, brings the server back. Pulled models stay on disk.

## What the APIs can do

Ollama exposes three styles on the same port.

1. **Native REST API** at `http://localhost:11434/api`: `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/ps`, `/api/show`, `/api/create`, `/api/copy`, `/api/pull`, `/api/push`, `/api/delete`, and `/api/version`.
2. **OpenAI-compatible API** at `http://localhost:11434/v1`: `/v1/chat/completions`, stateless `/v1/responses`, `/v1/embeddings`, `/v1/models`, and `/v1/audio/speech`. Set `api_key="ollama"` (the SDK requires a string; it is ignored locally). `/v1/chat/completions` has no per-request `num_ctx` field, so set context length with `OLLAMA_CONTEXT_LENGTH` or a `Modelfile`.
3. **Anthropic-compatible API** at `http://localhost:11434/v1/messages`. Set `ANTHROPIC_BASE_URL=http://localhost:11434` and `ANTHROPIC_AUTH_TOKEN=ollama`. This is what Claude Code uses. Some Anthropic features are unsupported, including `/v1/messages/count_tokens`, forced `tool_choice`, `cache_control`, PDF `document` blocks, and the Batches API.

Other native features worth knowing:

- **Streaming.** `/api/chat` and `/api/generate` stream NDJSON by default. Set `"stream": false` when a script wants one complete JSON body.
- **Structured outputs.** Pass `"format": "json"` or a JSON schema on `/api/chat` or `/api/generate`. Pair with `temperature: 0` and a validator such as Pydantic. Local models support this. Ollama's hosted cloud API does not.
- **Tool calling.** Pass a `tools` array. When the model needs an action, `message.tool_calls` holds the name and arguments. Check `ollama show <model>` for a `tools` capability.
- **Thinking.** Models such as `qwen3.5`, `qwen3.8`, `gemma4`, and `gpt-oss` accept `"think": true`, `false`, or named levels. Thinking tokens arrive in `message.thinking`.
- **Vision.** Multimodal models accept screenshots and photos. In the CLI: `ollama run gemma4 ./diagram.png "Explain this flow"`.
- **Embeddings.** `POST /api/embed` returns L2-normalised vectors. Keep the same embedding model for indexing and for queries.

Context length, if you do not set `OLLAMA_CONTEXT_LENGTH`, follows available VRAM (from 2026 onwards):

- Under 24 GiB: **4,096** tokens
- 24 to 48 GiB: **32,768** tokens
- 48 GiB or more: **262,144** tokens

Override it globally with `OLLAMA_CONTEXT_LENGTH=64000`, in a `Modelfile` (`PARAMETER num_ctx 32768`), in the desktop slider, or per request on `/api/chat` (`"options": {"num_ctx": 32768}`). Coding harnesses need 32K or 64K. The 4K default will truncate tool lists.

### Custom models with a Modelfile

```dockerfile
FROM qwen3.8:27b
PARAMETER temperature 0.2
PARAMETER num_ctx 65536
SYSTEM """You are a senior backend engineer. Follow Go standard style and keep answers brief."""
```

```bash
ollama create go-reviewer -f Modelfile
ollama run go-reviewer
```

`FROM` can also point at a local `.gguf` file, a shard wildcard (`FROM ./model-*.gguf`), or a Safetensors directory.

Ollama v0.34.1 (September 2026) removed the `ADAPTER` keyword for LoRA files, and it no longer auto-quantises on import. Merge and quantise a fine-tuned model first, then import the GGUF. See the [fine-tuning guide](../fine-tuning/README.md).

On Apple Silicon, Ollama can run MLX tags (for example `gemma4:12b-mlx`) alongside GGUF. v0.40.17 enabled concurrent request serving on Apple Silicon.

## Python and the examples in this folder

A minimal chat, once the server is up and the model is pulled:

```python
from ollama import Client

client = Client(host="http://localhost:11434")
reply = client.chat(
    model="qwen3.5:0.8b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(reply.message.content)
```

The same call with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
reply = client.chat.completions.create(
    model="qwen3.5:0.8b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(reply.choices[0].message.content)
```

Worked scripts live in [examples](examples/README.md):

- `01_chat_api.py` for chat and structured JSON.
- `02_tool_calling_agent.py` for a small file-reading loop.
- `03_local_pdf_rag.py` for embeddings plus a grounded answer.

Each script starts with a Prerequisites block, and it exits with a short message if the server is down or the model is missing. Set `OLLAMA_HOST` if the server is not on localhost.

Ollama does not host MCP servers. Your client (LM Studio, Claude Desktop, OpenCode, and others) is the MCP host, and it can still use Ollama as the model.

## Serving Ollama for a team

By default Ollama binds to `127.0.0.1:11434`. To let other machines on a LAN or VPN reach it, set `OLLAMA_HOST=0.0.0.0:11434`.

=== "macOS (desktop app)"

    ```bash
    launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
    ```

    Then quit and reopen the app.

=== "Linux (systemd)"

    ```bash
    sudo systemctl edit ollama.service
    ```

    Add:

    ```ini
    [Service]
    Environment="OLLAMA_HOST=0.0.0.0:11434"
    ```

    Then `sudo systemctl daemon-reload && sudo systemctl restart ollama`.

=== "Windows"

    Quit from the tray, add a user environment variable `OLLAMA_HOST` = `0.0.0.0:11434`, and relaunch.

If a browser app talks to Ollama from another origin, also set `OLLAMA_ORIGINS`, for example `OLLAMA_ORIGINS="https://chat.internal.company,chrome-extension://*"`.

### Connecting from another machine

Once the server is at `http://10.0.20.15:11434`:

```bash
OLLAMA_HOST=http://10.0.20.15:11434 ollama list
OLLAMA_HOST=http://10.0.20.15:11434 ollama run qwen3.8:27b

curl http://10.0.20.15:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8:27b", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

```python
from ollama import Client
client = Client(host="http://10.0.20.15:11434")

from openai import OpenAI
oai = OpenAI(base_url="http://10.0.20.15:11434/v1", api_key="ollama")
```

### Security

Ollama does not validate API keys or user logins. Anyone who can reach port `11434` can chat, pull large files, or delete models.

Do not expose port `11434` to the public internet. For a shared office server:

1. Keep Ollama bound to `127.0.0.1:11434` on the server itself.
2. Put Nginx or Caddy in front with HTTPS and a required `Authorization` header, or restrict access over a VPN, WireGuard, or Tailscale.
3. Set `OLLAMA_NO_CLOUD=1` so the box cannot make outbound cloud calls.

Minimal Nginx fragment:

```nginx
server {
    listen 443 ssl;
    server_name llm.internal.company;

    location / {
        if ($http_authorization != "Bearer YOUR_LONG_INTERNAL_TOKEN") {
            return 401;
        }
        proxy_pass http://127.0.0.1:11434;
        proxy_set_header Host localhost:11434;
        proxy_buffering off;
    }
}
```

### Useful server environment variables

| Variable | What it controls | Default and a practical setting |
| --- | --- | --- |
| `OLLAMA_CONTEXT_LENGTH` | Default context window when a request omits `num_ctx`. | VRAM-based (`4096`, `32768`, or `262144`). Set `32768` or `65536` for coding agents. |
| `OLLAMA_NUM_PARALLEL` | Concurrent requests one loaded model handles. | Unset lets MLX on Metal pick; other backends default to `1`. Setting `4` multiplies KV cache memory by 4. |
| `OLLAMA_MAX_LOADED_MODELS` | Models kept in memory at once. | `3 * GPUs`, or `3` on CPU. Set `1` or `2` on a tight machine. |
| `OLLAMA_KEEP_ALIVE` | Idle unload timeout. | `5m`. Use `30m` or `-1` on a dedicated server. |
| `OLLAMA_MAX_QUEUE` | Requests queued before HTTP `503`. | `512`. |
| `OLLAMA_KV_CACHE_TYPE` | Precision of the KV cache: `f16`, `q8_0`, `q4_0`. | `f16`. `q8_0` halves cache RAM when Flash Attention is active, with a small quality trade-off. |
| `OLLAMA_FLASH_ATTENTION` | Reduces memory growth at long context. | On automatically where supported (`1` to force, `0` to disable). |
| `CUDA_VISIBLE_DEVICES` | Pin NVIDIA GPUs. | Use UUIDs from `nvidia-smi -L` on multi-GPU boxes. |

If a model is larger than one GPU, Ollama splits layers across cards. If it fits on one GPU, it stays on one card.

### Docker, and when to leave Ollama

On Linux with the NVIDIA Container Toolkit:

```bash
docker run -d --gpus=all \
  -v ollama:/root/.ollama \
  -p 127.0.0.1:11434:11434 \
  -e OLLAMA_CONTEXT_LENGTH=32768 \
  -e OLLAMA_NO_CLOUD=1 \
  --name ollama ollama/ollama
```

Docker Desktop on macOS cannot pass Apple Silicon GPUs into a Linux container. Run Ollama natively on a Mac.

Ollama is the right default for laptops, workshops, rapid model switching, and a small team. Move to vLLM when you have data-centre NVIDIA GPUs serving dozens of concurrent users. Move to `llama-server` (llama.cpp) when you want direct control over every GGUF flag for a single model, without Ollama's daemon.

## Common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Replies crawl at 2 to 4 tokens/sec | Model plus KV cache spilled onto CPU RAM. | Run `ollama ps`. If `PROCESSOR` is `CPU` or a split, use a smaller model, lower `num_ctx`, or set `OLLAMA_KV_CACHE_TYPE=q8_0`. |
| Coding agent ignores tools or loops | The 4K default context truncated the tool list. | Confirm `CONTEXT` in `ollama ps`. Set `OLLAMA_CONTEXT_LENGTH=65536` (or `32768`). |
| Remote client gets `Connection refused` | Bound only to `127.0.0.1`, or a firewall. | Set `OLLAMA_HOST=0.0.0.0:11434`, restart, and allow the port (or, better, proxy 443). |
| `Error: LoRA adapters are no longer supported` | An old `Modelfile` still has `ADAPTER`. | Merge the adapter, export GGUF (`Q4_K_M`), and use `FROM ./merged.gguf`. |
| Browser app fails with a CORS error | Origin blocked. | Set `OLLAMA_ORIGINS` and restart. |
| Need raw errors or prompt text | Server logs. | See the log paths below. Set `OLLAMA_DEBUG=1` temporarily. |

Logs:

=== "macOS"

    `~/.ollama/logs/server.log`

=== "Linux"

    `journalctl -u ollama -n 100 --no-pager`

=== "Windows"

    `%LOCALAPPDATA%\Ollama\server.log`

=== "Docker"

    `docker logs ollama`

## Cheat sheet

```text
# CLI
ollama pull qwen3.5:9b
ollama run qwen3.5:9b
ollama ls
ollama ps
ollama show qwen3.8:27b
ollama create my-bot -f Modelfile
ollama stop qwen3.5:9b
ollama rm qwen3.5:4b
ollama launch claude --model qwen3.8:27b

# Check the server, then free memory
curl http://localhost:11434/api/version
ollama ps
ollama stop qwen3.5:9b

# Useful env
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_MODELS=/mnt/fast-ssd/ollama
OLLAMA_CONTEXT_LENGTH=65536
OLLAMA_KEEP_ALIVE=30m
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_NO_CLOUD=1
OLLAMA_KV_CACHE_TYPE=q8_0

# HTTP (base http://localhost:11434)
POST /api/chat
POST /api/generate     # also used to preload or to unload with keep_alive 0
POST /api/embed
GET  /api/tags
GET  /api/ps
GET  /api/version
POST /v1/chat/completions
POST /v1/messages
```

## Sources

- [Ollama CLI](https://docs.ollama.com/cli), [FAQ](https://docs.ollama.com/faq), [context length](https://docs.ollama.com/context-length), [Modelfile](https://docs.ollama.com/modelfile), [import](https://docs.ollama.com/import)
- Platform: [macOS](https://docs.ollama.com/macos), [Linux](https://docs.ollama.com/linux), [Windows](https://docs.ollama.com/windows), [Docker](https://docs.ollama.com/docker), [GPU](https://docs.ollama.com/gpu)
- API: [introduction](https://docs.ollama.com/api/introduction), [chat](https://docs.ollama.com/api/chat), [generate](https://docs.ollama.com/api/generate), [embed](https://docs.ollama.com/api/embed), [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [Anthropic compatibility](https://docs.ollama.com/api/anthropic-compatibility)
- [Tool calling](https://docs.ollama.com/capabilities/tool-calling), [structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [thinking](https://docs.ollama.com/capabilities/thinking), [vision](https://docs.ollama.com/capabilities/vision)
- Integrations: [Claude Code](https://docs.ollama.com/integrations/claude-code), [Codex](https://docs.ollama.com/integrations/codex), [OpenCode](https://docs.ollama.com/integrations/opencode), [Cline](https://docs.ollama.com/integrations/cline), [Hermes](https://docs.ollama.com/integrations/hermes)
- [Ollama Python library](https://github.com/ollama/ollama-python) and [releases](https://github.com/ollama/ollama/releases)
