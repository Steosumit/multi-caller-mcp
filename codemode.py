"""Code mode: expose MCP tools as a TypeScript API and execute generated TS.

The model never sees the individual MCP tools. It sees one ``execute_typescript``
tool plus a generated TypeScript declaration, writes a script that calls
``codemode.<server>.<tool>({...})``, and receives the script's ``console.log``
output.

The script runs in a Bun subprocess. A tiny JSON-lines protocol bridges its
``codemode`` calls back to the Python MCP clients. On Linux the Bun process runs
inside a bwrap sandbox with no network, no project files, and no credentials.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

from mcp_converter import StdioMCPClient, _format_tool_result, sanitize_identifier

_CODE_LIMIT = 20_000
_RUNNER_NAME = "codemode_runner.ts"
_MAIN_NAME = "main.ts"
_TYPE_LIMIT = 12_000


class CodeModeError(RuntimeError):
    """Raised when code-mode setup or execution fails."""


# ---------------------------------------------------------------------------
# Tool inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolInfo:
    server: str
    original_name: str
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None


@dataclass
class CodeModeStats:
    calls: int = 0
    servers: Counter[str] = field(default_factory=Counter)
    duration: float = 0.0

    def describe(self) -> str:
        detail = ", ".join(
            f"{name}: {count}" for name, count in sorted(self.servers.items())
        )
        suffix = f" ({detail})" if detail else ""
        return f"code-mode: {self.calls} MCP call(s){suffix}, {self.duration:.2f}s"


def collect_tools(clients: dict[str, StdioMCPClient]) -> list[ToolInfo]:
    """Read every client's tool list and assign unique TypeScript method names."""
    tools: list[ToolInfo] = []
    for server_name, client in clients.items():
        try:
            specs = client.list_tools()
        except Exception as exc:  # noqa: BLE001 - add server context
            raise CodeModeError(
                f"MCP server {server_name!r} tools/list failed: {exc}"
            ) from exc

        used: set[str] = set()
        for spec in specs:
            original = spec.get("name") or "mcp_tool"
            method = _unique_identifier(
                sanitize_identifier(original, fallback="mcp_tool", limit=64),
                used,
            )
            tools.append(
                ToolInfo(
                    server=server_name,
                    original_name=original,
                    name=method,
                    description=spec.get("description")
                    or spec.get("title")
                    or f"MCP tool {original}",
                    input_schema=spec.get("inputSchema") or {},
                    output_schema=spec.get("outputSchema"),
                )
            )
    tools.sort(key=lambda item: (item.server, item.name))
    return tools


def _unique_identifier(base: str, used: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# JSON Schema -> TypeScript declarations
# ---------------------------------------------------------------------------


def _pascal(value: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", value)
    result = "".join(part[:1].upper() + part[1:] for part in parts if part)
    if not result:
        result = "Tool"
    if result[0].isdigit():
        result = f"T{result}"
    return result


def _property_name(name: str) -> str:
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name):
        return name
    return json.dumps(name, ensure_ascii=False)


def _jsdoc(text: str, indent: str = "") -> list[str]:
    clean = str(text).replace("*/", "* /").strip()
    lines = clean.splitlines() or [""]
    if len(lines) == 1:
        return [f"{indent}/** {lines[0]} */"]
    body = [f"{indent} * {line}".rstrip() for line in lines]
    return [f"{indent}/**", *body, f"{indent} */"]


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        return {}
    node: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return {}
        node = node[part]
    return node if isinstance(node, dict) else {}


def _ts_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False)


def _ts_type(
    schema: Any,
    root: dict[str, Any],
    *,
    depth: int = 0,
) -> str:
    if not isinstance(schema, dict) or depth > 8:
        return "unknown"

    if "$ref" in schema:
        resolved = _resolve_ref(str(schema["$ref"]), root)
        return _ts_type(resolved, root, depth=depth + 1) if resolved else "unknown"

    if "allOf" in schema and isinstance(schema["allOf"], list):
        parts = [_ts_type(item, root, depth=depth + 1) for item in schema["allOf"]]
        parts = [part for part in parts if part != "unknown"]
        return " & ".join(dict.fromkeys(parts)) if parts else "unknown"

    for key in ("anyOf", "oneOf"):
        if key in schema and isinstance(schema[key], list):
            parts = [_ts_type(item, root, depth=depth + 1) for item in schema[key]]
            return " | ".join(dict.fromkeys(parts)) or "unknown"

    if "const" in schema:
        return _ts_literal(schema["const"])

    if "enum" in schema and isinstance(schema["enum"], list):
        values = [_ts_literal(value) for value in schema["enum"]]
        return " | ".join(dict.fromkeys(values)) or "unknown"

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        parts = [
            _ts_type({**schema, "type": item}, root, depth=depth + 1)
            for item in schema_type
        ]
        return " | ".join(dict.fromkeys(parts)) or "unknown"

    if schema_type == "array" or "items" in schema:
        item_type = _ts_type(schema.get("items") or {}, root, depth=depth + 1)
        if "|" in item_type or "&" in item_type:
            item_type = f"({item_type})"
        return f"{item_type}[]"

    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        additional = schema.get("additionalProperties")
        if not isinstance(properties, dict):
            properties = {}

        members: list[str] = []
        for raw_name, prop in properties.items():
            prop = prop if isinstance(prop, dict) else {}
            optional = "" if raw_name in required else "?"
            description = prop.get("description")
            if description:
                members.append(
                    "/** "
                    + str(description).replace("*/", "* /").replace("\n", " ")
                    + " */"
                )
            members.append(
                f"{_property_name(raw_name)}{optional}: "
                f"{_ts_type(prop, root, depth=depth + 1)};"
            )

        if additional is True:
            members.append("[key: string]: unknown;")
        elif isinstance(additional, dict):
            members.append(
                f"[key: string]: {_ts_type(additional, root, depth=depth + 1)};"
            )

        if not members:
            return "Record<string, unknown>"
        return "{ " + " ".join(members) + " }"

    if schema_type == "string":
        return "string"
    if schema_type in {"integer", "number"}:
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "null":
        return "null"
    return "unknown"


def _type_name(
    server: str,
    tool: str,
    suffix: str,
    used: set[str],
) -> str:
    base = f"{_pascal(server)}{_pascal(tool)}{suffix}"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}{index}"
        index += 1
    used.add(candidate)
    return candidate


def _interface_for(tool: ToolInfo, input_name: str, output_name: str) -> list[str]:
    schema = tool.input_schema
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    lines: list[str] = []

    if not isinstance(properties, dict) or not properties:
        lines.append(f"type {input_name} = Record<string, unknown>;")
    else:
        lines.append(f"interface {input_name} {{")
        for raw_name, prop in properties.items():
            prop = prop if isinstance(prop, dict) else {}
            if prop.get("description"):
                lines.extend(_jsdoc(str(prop["description"]), indent="  "))
            optional = "" if raw_name in required else "?"
            lines.append(
                f"  {_property_name(raw_name)}{optional}: "
                f"{_ts_type(prop, schema)};"
            )
        lines.append("}")

    if tool.output_schema:
        lines.append(
            f"type {output_name} = {_ts_type(tool.output_schema, tool.output_schema)};"
        )
    else:
        lines.append(f"type {output_name} = any;")
    return lines


def generate_typescript_api(tools: list[ToolInfo]) -> str:
    """Render the nested ``codemode`` TypeScript API for the prompt."""
    used_types: set[str] = set()
    definitions: list[str] = []
    methods_by_server: dict[str, list[ToolInfo]] = {}
    type_names: dict[tuple[str, str], tuple[str, str]] = {}

    for tool in tools:
        input_name = _type_name(tool.server, tool.name, "Input", used_types)
        output_name = _type_name(tool.server, tool.name, "Output", used_types)
        type_names[(tool.server, tool.name)] = (input_name, output_name)
        definitions.extend(_interface_for(tool, input_name, output_name))
        definitions.append("")
        methods_by_server.setdefault(tool.server, []).append(tool)

    root_members: list[str] = []
    used_server_interfaces: set[str] = {"CodeModeRootApi"}
    for server_name in sorted(methods_by_server):
        interface_name = _unique_identifier(
            f"{_pascal(server_name)}Api", used_server_interfaces
        )
        definitions.append(f"interface {interface_name} {{")
        for tool in methods_by_server[server_name]:
            input_name, output_name = type_names[(tool.server, tool.name)]
            definitions.extend(_jsdoc(tool.description, indent="  "))
            definitions.append(
                f"  {tool.name}: (input: {input_name}) => Promise<{output_name}>;"
            )
        definitions.append("}")
        definitions.append("")
        root_members.append(f"  {server_name}: {interface_name};")

    api_lines = ["interface CodeModeRootApi {"]
    api_lines.extend(root_members)
    api_lines.append("}")
    api_lines.append("")
    api_lines.append("declare const codemode: CodeModeRootApi;")

    header = [
        "// MCP tools exposed by the code-mode runtime.",
        "// Example: const result = await codemode.server.tool({ argument: 1 });",
        "// Intermediate values stay in the sandbox; call console.log() with the final answer.",
        "",
    ]
    declaration = "\n".join([*header, *definitions, *api_lines]).strip()
    if len(declaration) > _TYPE_LIMIT:
        declaration = (
            declaration[:_TYPE_LIMIT]
            + "\n// ... (API truncated; ask for a narrower task)"
        )
    return declaration


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class CodeModeExecutor:
    """Run generated TypeScript with ``codemode`` calls bridged to MCP clients."""

    def __init__(
        self,
        clients: dict[str, StdioMCPClient],
        tools: list[ToolInfo],
        *,
        timeout: float = 20.0,
        max_calls: int = 25,
        max_logs: int = 50,
        max_output: int = 8_000,
        sandbox: str = "auto",
        bun_path: str | None = None,
    ) -> None:
        self.clients = clients
        self.tools = {(tool.server, tool.name): tool for tool in tools}
        for client in clients.values():
            client.timeout = min(client.timeout, max(1.0, timeout))
        self.timeout = timeout
        self.max_calls = max_calls
        self.max_logs = max_logs
        self.max_output = max_output
        self.sandbox = sandbox
        self.bun_path = bun_path or shutil.which("bun") or str(
            Path.home() / ".bun" / "bin" / "bun"
        )
        self.stats = CodeModeStats()
        self._runner_path = Path(__file__).with_name(_RUNNER_NAME)

    # -- public API --------------------------------------------------------

    def execute(self, code: str) -> str:
        started = time.monotonic()
        try:
            return self._execute(code, started)
        finally:
            self.stats.duration = time.monotonic() - started

    def as_tool(self) -> StructuredTool:
        executor = self

        def execute_typescript(code: str) -> str:
            """Execute a complete TypeScript module in the sandbox.

            The module may call the generated ``codemode`` API. Use
            ``console.log()`` for any value the agent should see. Return only
            the script, without markdown fences.
            """
            return executor.execute(code)

        return StructuredTool.from_function(
            func=execute_typescript,
            name="execute_typescript",
            description=(
                "Execute a TypeScript module in an isolated Bun sandbox. "
                "Call the generated codemode API and console.log the result."
            ),
        )

    # -- execution internals ----------------------------------------------

    def _execute(self, code: str, started: float) -> str:
        code = _strip_code_fences(code).strip()
        if not code:
            return "Error: no TypeScript code was provided."
        if len(code) > _CODE_LIMIT:
            return f"Error: code exceeds {_CODE_LIMIT} characters."

        if not self._runner_path.exists():
            return f"Error: missing runner file {self._runner_path.name}."

        bun = Path(self.bun_path).expanduser()
        if not bun.exists():
            return (
                f"Error: Bun was not found at {bun}. Install Bun or pass --bun-path."
            )

        with tempfile.TemporaryDirectory(prefix="codemode-") as tmp:
            workdir = Path(tmp)
            runner = workdir / _RUNNER_NAME
            main = workdir / _MAIN_NAME
            runner.write_text(self._runner_path.read_text(encoding="utf-8"), encoding="utf-8")
            main.write_text(code, encoding="utf-8")

            try:
                command = self._command(bun, workdir)
            except CodeModeError as exc:
                return f"Error: {exc}"
            env = self._environment()

            try:
                proc = subprocess.Popen(
                    command,
                    cwd=workdir,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                    preexec_fn=_resource_limits(self.timeout)
                    if os.name == "posix"
                    else None,
                )
            except OSError as exc:
                return f"Error: could not start TypeScript runtime: {exc}"

            try:
                return self._bridge(proc, started)
            finally:
                _stop_process(proc)

    def _command(self, bun: Path, workdir: Path) -> list[str]:
        mode = self.sandbox
        if mode == "auto":
            mode = "bwrap" if sys.platform.startswith("linux") else "none"
        if mode == "bwrap":
            bwrap = shutil.which("bwrap")
            if not bwrap:
                raise CodeModeError(
                    "bwrap is required for code mode on Linux. "
                    "Install bubblewrap or pass --sandbox none (unsafe)."
                )
            return [
                bwrap,
                "--tmpfs",
                "/",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/etc",
                "/etc",
                "--symlink",
                "usr/lib",
                "/lib",
                "--symlink",
                "usr/lib64",
                "/lib64",
                "--ro-bind",
                str(bun.resolve()),
                "/bun",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/work",
                "--ro-bind",
                str(workdir),
                "/work",
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
                "--chdir",
                "/work",
                "/bun",
                f"/work/{_RUNNER_NAME}",
                f"/work/{_MAIN_NAME}",
            ]
        if mode == "none":
            return [
                str(bun),
                str(workdir / _RUNNER_NAME),
                str(workdir / _MAIN_NAME),
            ]
        raise CodeModeError(f"unsupported sandbox mode {mode!r}.")

    def _environment(self) -> dict[str, str]:
        allowlist = {
            server: sorted(
                tool.name
                for tool in self.tools.values()
                if tool.server == server
            )
            for server in self.clients
        }
        return {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "NO_COLOR": "1",
            "CODEMODE_SERVERS": json.dumps(allowlist),
            "CODEMODE_MAX_LOGS": str(self.max_logs),
        }

    def _bridge(self, proc: subprocess.Popen[str], started: float) -> str:
        assert proc.stdout and proc.stderr and proc.stdin
        messages: queue.Queue[str | None] = queue.Queue()
        stderr: deque[str] = deque(maxlen=80)
        event: dict[str, Any] = {}

        def read_stdout() -> None:
            try:
                for line in proc.stdout:
                    line = line.strip()
                    if line:
                        messages.put(line)
            finally:
                messages.put(None)

        def read_stderr() -> None:
            for line in proc.stderr:
                stderr.append(line.rstrip())

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()

        deadline = started + self.timeout
        timed_out = False
        protocol_noise: list[str] = []

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                raw = messages.get(timeout=remaining)
            except queue.Empty:
                timed_out = True
                break
            if raw is None:
                break
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                protocol_noise.append(raw)
                continue
            if not isinstance(message, dict):
                protocol_noise.append(raw)
                continue

            kind = message.get("type")
            if kind == "call":
                self._handle_call(proc, message)
            elif kind in {"done", "error"}:
                event = message
                break
            else:
                protocol_noise.append(raw)

        if timed_out:
            return self._with_stderr(
                f"Error: execution timed out after {self.timeout:g}s.", stderr
            )

        if event.get("type") == "done":
            logs = [str(item) for item in event.get("logs", [])]
            output = "\n".join(logs).strip()
            if protocol_noise:
                output = "\n".join([output, *protocol_noise]).strip()
            if not output:
                output = "(no output)"
            return _truncate(output, self.max_output)

        if event.get("type") == "error":
            error = str(event.get("error") or "unknown TypeScript error")
            logs = [str(item) for item in event.get("logs", [])]
            output = f"Error: {error}"
            if logs:
                output += "\n\nLogs before failure:\n" + "\n".join(logs)
            return self._with_stderr(output, stderr)

        return self._with_stderr(
            "Error: TypeScript runner exited without a result.",
            stderr,
            extra=protocol_noise,
        )

    def _handle_call(self, proc: subprocess.Popen[str], message: dict[str, Any]) -> None:
        request_id = message.get("id")
        server = str(message.get("server") or "")
        tool_name = str(message.get("tool") or "")
        args = message.get("args")
        self.stats.calls += 1
        self.stats.servers[server] += 1

        if self.stats.calls > self.max_calls:
            self._send_result(
                proc,
                request_id,
                ok=False,
                error=f"tool call limit exceeded ({self.max_calls})",
            )
            return
        if not isinstance(args, dict):
            self._send_result(
                proc, request_id, ok=False, error="tool arguments must be an object"
            )
            return

        tool = self.tools.get((server, tool_name))
        if tool is None:
            self._send_result(
                proc,
                request_id,
                ok=False,
                error=f"unknown tool {server}.{tool_name}",
            )
            return

        try:
            result = self.clients[server].call_tool_raw(tool.original_name, args)
            if result.get("isError"):
                text = _format_tool_result(result)
                if text.startswith("Error: "):
                    text = text[7:]
                raise CodeModeError(text)
            value = _result_value(result)
        except Exception as exc:  # noqa: BLE001 - report to the script
            self._send_result(proc, request_id, ok=False, error=str(exc))
            return
        self._send_result(proc, request_id, ok=True, value=value)

    @staticmethod
    def _send_result(
        proc: subprocess.Popen[str],
        request_id: Any,
        *,
        ok: bool,
        value: Any = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"type": "result", "id": request_id, "ok": ok}
        if ok:
            payload["value"] = value
        else:
            payload["error"] = error or "tool call failed"
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _with_stderr(
        self,
        message: str,
        stderr: deque[str],
        *,
        extra: list[str] | None = None,
    ) -> str:
        parts = [message]
        if extra:
            parts.append("\n".join(extra))
        if stderr:
            parts.append("stderr:\n" + "\n".join(stderr))
        return _truncate("\n\n".join(part for part in parts if part), self.max_output)


def _strip_code_fences(code: str) -> str:
    match = re.search(
        r"```(?:typescript|ts|javascript|js)?\s*\n(.*?)```",
        code,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1) if match else code


def _result_value(result: dict[str, Any]) -> Any:
    structured = result.get("structuredContent")
    if structured is not None:
        return structured

    content = result.get("content") or []
    values: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            values.append(block)
            continue
        kind = block.get("type")
        if kind == "text":
            values.append(_parse_json(block.get("text", "")))
        elif kind == "resource":
            resource = block.get("resource") or {}
            values.append(resource.get("text", resource))
        else:
            values.append(block)

    if not values:
        return "(no output)"
    return values[0] if len(values) == 1 else values


def _parse_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return text


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def _resource_limits(timeout: float):
    def apply() -> None:
        try:
            import resource
        except ImportError:  # pragma: no cover - non-POSIX
            return
        limits = (
            (resource.RLIMIT_CPU, max(1, int(timeout) + 2)),
            (resource.RLIMIT_AS, 1024 * 1024 * 1024),
            (resource.RLIMIT_FSIZE, 10 * 1024 * 1024),
        )
        for name, value in limits:
            try:
                resource.setrlimit(name, (value, value))
            except (ValueError, OSError):
                pass

    return apply


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover - Windows
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if stream:
                stream.close()
        except OSError:
            pass


def build_execute_tool(executor: CodeModeExecutor) -> StructuredTool:
    return executor.as_tool()


__all__ = [
    "CodeModeError",
    "CodeModeExecutor",
    "CodeModeStats",
    "ToolInfo",
    "build_execute_tool",
    "collect_tools",
    "generate_typescript_api",
]
