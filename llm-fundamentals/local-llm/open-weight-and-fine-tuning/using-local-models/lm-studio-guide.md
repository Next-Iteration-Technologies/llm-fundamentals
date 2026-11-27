# LM Studio

LM Studio is a desktop app that downloads local models, runs them, and serves an HTTP API. It is the usual GUI alternative to Ollama. Use it when you want a chat window, a model catalogue you can browse, MCP tools, or a local server you start with a toggle.

Based on November 2026 internet data, covering the 0.4.x / Bionic line.

Ollama remains the default in this workshop because the CLI, `ollama launch`, and the Python examples all assume port `11434`. You can use both apps on one machine, but do not run both servers on the same port at the same time.

## What you get

- A desktop chat UI, with model download and a loaded-model indicator.
- GGUF models on every platform, and MLX models on Apple Silicon.
- A local server that speaks OpenAI-compatible endpoints, and Anthropic-compatible endpoints in current 0.4.x builds.
- **MCP** via an `mcp.json` file, so the chat UI can call local tools.
- **Projects and Sessions**, which keep a working set of files and a conversation history together.
- A CLI, `lms`, plus `llmster` for terminal use.
- **LM Link**, for sharing or connecting a running local server.

Official docs live at [lmstudio.ai/docs](https://lmstudio.ai/docs).

## Install

Download the app from [lmstudio.ai](https://lmstudio.ai) for macOS, Windows, or Linux. On Apple Silicon, pick the Mac build so you can use MLX as well as GGUF.

After install, open the app once and, if you want the CLI:

```bash
# The app can install the lms command for you.
lms --help
```

The first useful check is the same idea as Ollama: start the local server in the app (Developer / Server, or the equivalent toggle in your build) and hit the health endpoint the UI shows. The default OpenAI-compatible base is usually `http://localhost:1234/v1`. That is **1234**, not 11434. Point the OpenAI SDK at that base and use `api_key="lm-studio"` or any non-empty string the client requires.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
reply = client.chat.completions.create(
    model="qwen3.5-0.8b",  # use the identifier shown in LM Studio
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(reply.choices[0].message.content)
```

If something else is already bound to 1234, change the port in the server settings and update `base_url`.

## Download and load a model

1. Open the catalogue, search for Qwen 3.5, and download a size that fits. For a 16 GB Mac, a 4-bit 0.8B or 4B build is the easy start. Check the [VRAM guide](hardware-and-vram-guide.md) before you download a 27B or 32B.
2. Load the model in the chat UI. Watch the memory indicator. On a Mac this comes out of unified memory, the same pool as Ollama.
3. Unload it when you are done, from the same UI. Leaving a large model loaded will crowd your IDE the same way Ollama does.

`lms` can list, load, and unload models from a terminal. Run `lms --help` and `lms load --help` on your install for the current flags, because they have moved between minor releases.

## Chat, Projects, and Sessions

A fresh chat is enough for a single question. **Projects** (0.4.x) give you a place to pin files, instructions, and a set of sessions. Use a Project when the work is "this repo" or "this design doc," so each Session does not start from a blank system prompt.

Current builds also support **stateful** `/v1/chat` behaviour, where the server can keep a session rather than requiring you to resend the full history on every call. For scripts in this workshop, stick to stateless calls you fully control, the same way the Ollama examples do.

## MCP tools

LM Studio can act as an MCP host. You declare servers in an `mcp.json` file (the app documents the exact path for your OS). Each entry is a command the app starts, plus any env it needs. Once a server is enabled, the chat UI can call its tools.

Ollama does not host MCP servers itself. If you want MCP with an Ollama-backed model, use a client that is an MCP host (LM Studio, Claude Desktop, OpenCode, Continue) and point that client at Ollama, or use LM Studio as both the model runner and the host.

Do not paste secrets into `mcp.json` if the file lives in a shared repo. Use environment variables, and keep the file local.

## When to use LM Studio, and when to use Ollama

Use **Ollama** when you want a single CLI, `ollama launch` into coding harnesses, a systemd or menu-bar server, and the scripts in this folder as written.

Use **LM Studio** when you want a visual model browser, a friendly unload button, MCP in the same app as the chat, or MLX in a GUI on a Mac.

Use **both** only with a clear port split: Ollama on `11434`, LM Studio on `1234`. Unload the model in the app you are not using, so 16 GB machines stay usable. See [Loading and unloading models](ollama-guide.md#loading-and-unloading-models-from-ram) for the Ollama side.

The examples in `using-local-models/examples/` talk to Ollama by default. To point one at LM Studio, start LM Studio's server, then run the script with a matching base URL if you adapt it, or use the OpenAI SDK snippet above.

## Sources

- [LM Studio docs](https://lmstudio.ai/docs)
- [LM Studio app](https://lmstudio.ai/docs/app)
- [MCP in LM Studio](https://lmstudio.ai/docs/mcp)
- [Projects and Sessions](https://lmstudio.ai/docs/projects-and-sessions)
