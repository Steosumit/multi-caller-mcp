# multi-caller-mcp

Imagine your agent makes `n` calls for doing a `n` step workflow. And during each call it wastes tokens in repititive reasoning and token generation. We can do the same workflow with 2 generations instead of `2 + n`

Idea is the open source implementation of cloudflare codemode concept: https://blog.cloudflare.com/code-mode-mcp/

A local agent that connects to one or more **MCP stdio servers** and supports two
interaction modes:

- **`tools`** — classic function calling: each MCP tool is exposed to the model.
- **`code` (Code Mode)** — MCP tools are converted into a nested **TypeScript API**;
  the model writes one script that calls `codemode.<server>.<tool>({...})`, and the
  script runs in a Bun sandbox.

Both modes use `ChatOpenAI` against a local OpenAI-compatible server (LM Studio at
`http://127.0.0.1:1234/v1` by default).

## Why code mode

Direct tool calling sends every intermediate result back through the model. Code
mode keeps intermediate values in the sandbox, so the model can chain several MCP
calls in one script and only read the final `console.log` output. The model also
gets a typed API instead of a list of tool definitions.

## Requirements

- Python 3.11+ and the dependencies in `requirements.txt`
- [Bun](https://bun.sh/) for code mode
- LM Studio (or any OpenAI-compatible endpoint) with a tool-calling model
- Linux + `bubblewrap` (`bwrap`) for the code-mode sandbox

## Setup

```bash
.venv/bin/pip install -r requirements.txt
```

Start LM Studio, load a model such as `qwen/qwen3-8b`, and enable the local server
on port `1234`. Verify connectivity:

```bash
.venv/bin/python llm.py
```

Configuration is read from `.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen/qwen3-8b` | Model id |
| `LLM_API_KEY` | `lm-studio` | API key (ignored by LM Studio) |
| `LLM_TEMPERATURE` | `0` | Sampling temperature |
| `CODE_TIMEOUT` | `20` | Code-mode execution timeout (seconds) |
| `CODE_MAX_CALLS` | `25` | Max MCP calls per generated script |
| `MCP_SERVER_COMMAND` | unset | Default single MCP server command |

## Usage

### Local tools only

```bash
.venv/bin/python agent.py "What is (17 * 23) + 9?"
.venv/bin/python agent.py            # interactive REPL
```

### Direct tool mode

```bash
.venv/bin/python agent.py --mode tools \
  --mcp ".venv/bin/python tests/fake_mcp_server.py" \
  "Use the add tool to compute 19 + 23."
```

With multiple servers, tool names are prefixed as `server__tool`:

```bash
.venv/bin/python agent.py --mode tools \
  --mcp alpha=".venv/bin/python tests/fake_mcp_server.py --label alpha" \
  --mcp beta=".venv/bin/python tests/fake_mcp_server.py --label beta" \
  "Call alpha__add and beta__add, then compare the results."
```

### Code mode

Passing `--mcp` without `--mode` selects code mode automatically.

```bash
.venv/bin/python agent.py --mode code \
  --mcp alpha=".venv/bin/python tests/fake_mcp_server.py --label alpha" \
  --mcp beta=".venv/bin/python tests/fake_mcp_server.py --label beta" \
  "In one script, call alpha.add(19,23) and beta.add(2,3), then print both sums."
```

The model sees a generated API like:

```ts
interface AlphaAddInput {
  /** first number */
  a: number;
  /** second number */
  b: number;
}
type AlphaAddOutput = any;

interface AlphaApi {
  /** Add two numbers and return a JSON result. */
  add: (input: AlphaAddInput) => Promise<AlphaAddOutput>;
}

interface CodeModeRootApi {
  alpha: AlphaApi;
  beta: BetaApi;
}

declare const codemode: CodeModeRootApi;
```

and submits TypeScript through the single `execute_typescript` tool:

```ts
const a = await codemode.alpha.add({ a: 19, b: 23 });
const b = await codemode.beta.add({ a: 2, b: 3 });
console.log(a, b);
```

The parser accepts top-level `await`, or an async `main()` that is awaited (and
gives unawaited `main()` calls a short settle window).

Other useful flags:

```bash
--sandbox auto|bwrap|none   # auto = bwrap on Linux; none is unsafe
--timeout 20                # code execution timeout
--max-calls 25              # MCP calls per script
--bun-path /path/to/bun
--stats                     # MCP calls per server and elapsed time
--model qwen/qwen3-8b
--base-url http://127.0.0.1:1234/v1
```

## How code mode works

```
model ── execute_typescript(code) ──► Python
                                        │  writes runner.ts + main.ts
                                        ▼
                              bwrap: Bun sandbox
                                        │  codemode.alpha.add({...})
                                        ▼
                              JSON lines on stdout
                                        │
                                        ▼
                              Python MCP client
                                        │  tools/call
                                        ▼
                                   MCP server
```

1. Python starts every MCP server and reads `tools/list`.
2. `codemode.py` converts each `inputSchema` into a TypeScript input interface and
   renders nested `codemode.<server>.<tool>` declarations.
3. The model writes a script; `CodeModeExecutor` writes it to a temp directory.
4. `codemode_runner.ts` installs the `codemode` proxy, captures `console` output,
   and emits namespaced call events.
5. Python resolves `(server, tool)` to the right `StdioMCPClient`, sends
   `tools/call`, and returns the result to Bun.
6. The final logs are returned to the model as the tool result.

### Sandbox

On Linux, `--sandbox auto` runs Bun inside `bwrap` with:

- a minimal root (`/usr`, `/etc`, `/proc`, `/dev`, a writable `/tmp`),
- the Bun binary mounted read-only at `/bun`,
- the temp workdir mounted read-only at `/work`,
- `--unshare-all` (no network), `--die-with-parent`, and `--new-session`,
- scrubbed environment (no API keys), plus CPU/memory/file-size limits.

The sandbox protects the host from generated code. It does **not** make an untrusted
MCP server safe, prevent prompt injection in tool descriptions, or replace human
approval for destructive tools. Use `--sandbox none` only for debugging.

## Tests

The offline suite uses a configurable fake MCP server and does not call the LLM:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

It covers multi-server routing, namespaced direct tools, TypeScript generation,
markdown-fence stripping, tool errors, call limits, timeouts, output truncation,
and bwrap isolation (network, files, secrets).

## Files

| File | Purpose |
| --- | --- |
| `agent.py` | CLI, REPL, mode selection, prompt construction |
| `codemode.py` | TS API generation, executor, JSON-lines bridge |
| `codemode_runner.ts` | Bun harness and `codemode` proxy |
| `mcp_converter.py` | MCP stdio client, LangChain tools, multi-server loading |
| `llm.py` | `ChatOpenAI` factory and model discovery |
| `sandbox.py` | Local `calculator` / `run_python` tools for direct mode |
| `tests/fake_mcp_server.py` | Fake MCP server used by the suite |
| `tests/test_codemode.py` | Offline tests |

## Limitations

- Bun strips TypeScript types; it does not typecheck. The declarations guide the
  model, but a separate `tsc --noEmit` would be needed to reject type errors.
- The bwrap path is Linux-only. On other platforms, code mode requires an explicit
  `--sandbox none`.
- Output types default to `any` unless an MCP tool provides `outputSchema`.
- Very large APIs are truncated in the system prompt; add a search step if you
  connect many servers.
- A local 8B model may need a retry or two to produce correct TypeScript.
