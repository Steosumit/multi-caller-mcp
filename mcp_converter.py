"""Minimal MCP (Model Context Protocol) stdio client and LangChain converter.

This module has no dependency on the `mcp` package: it speaks the JSON-RPC 2.0
stdio transport directly, converts each MCP tool's JSON Schema into a pydantic
model, and wraps it as a LangChain `StructuredTool` the agent can call.

Example:
    with mcp_tools("npx -y @modelcontextprotocol/server-everything") as tools:
        agent = create_agent(model=get_llm(), tools=tools)
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "multi-caller-mcp", "version": "0.1.0"}
_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_]+")
_ALIAS_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")


class MCPError(RuntimeError):
    """Raised when the MCP server cannot be reached or returns an error."""


@dataclass(frozen=True)
class MCPServerSpec:
    """A named MCP stdio server command."""

    name: str
    command: str


def sanitize_identifier(name: str, *, fallback: str = "server", limit: int = 40) -> str:
    """Turn an arbitrary server/tool name into a safe TS/LangChain identifier."""
    clean = _IDENTIFIER_RE.sub("_", name).strip("_") or fallback
    if clean[0].isdigit():
        clean = f"_{clean}"
    return clean[:limit]


def parse_mcp_specs(raw_specs: list[str]) -> list[MCPServerSpec]:
    """Parse repeated ``--mcp [NAME=]COMMAND`` values into server specs."""
    entries = [raw.strip() for raw in raw_specs if raw and raw.strip()]
    specs: list[MCPServerSpec] = []
    for raw in entries:
        name: str | None = None
        command = raw
        if "=" in raw:
            candidate, remainder = raw.split("=", 1)
            if _ALIAS_RE.fullmatch(candidate.strip()):
                name = sanitize_identifier(candidate.strip(), fallback="mcp")
                command = remainder.strip()
        if not command:
            raise MCPError(f"empty MCP server command in {raw!r}.")
        if name is None:
            if len(entries) > 1:
                raise MCPError(
                    "when using multiple --mcp servers, name each one: "
                    "--mcp NAME=COMMAND"
                )
            name = "mcp"
        specs.append(MCPServerSpec(name=name, command=command))

    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise MCPError(f"duplicate MCP server name {spec.name!r}.")
        seen.add(spec.name)
    return specs


class StdioMCPClient:
    """Small synchronous MCP client for servers that use stdio transport."""

    def __init__(
        self,
        command: str | list[str],
        *,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.argv = shlex.split(command) if isinstance(command, str) else list(command)
        self.env = env
        self.timeout = timeout
        self._proc: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "StdioMCPClient":
        if not self.argv:
            raise MCPError("MCP server command is empty.")
        child_env = os.environ.copy()
        if self.env:
            child_env.update(self.env)
        try:
            self._proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1,
                env=child_env,
            )
        except OSError as exc:
            raise MCPError(f"could not start MCP server {self.argv[0]!r}: {exc}") from exc

        threading.Thread(target=self._read_loop, daemon=True).start()
        self._initialize()
        return self

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        finally:
            for stream in (proc.stdout, proc.stderr):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "StdioMCPClient":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- transport ---------------------------------------------------------

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # ignore non-protocol logging on stdout
                if isinstance(message, dict) and "id" in message:
                    self._messages.put(message)
        finally:
            self._messages.put(None)

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise MCPError("MCP server is not running.")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params is not None:
                payload["params"] = params
            self._send(payload)

            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError(f"timed out waiting for MCP response to {method!r}.")
                try:
                    message = self._messages.get(timeout=remaining)
                except queue.Empty as exc:
                    raise MCPError(
                        f"timed out waiting for MCP response to {method!r}."
                    ) from exc
                if message is None:
                    raise MCPError("MCP server closed the connection.")
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    error = message["error"]
                    raise MCPError(error.get("message", str(error)))
                return message.get("result", {})

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
        )
        self._notify("notifications/initialized")

    # -- MCP methods -------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            result = self._request("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool_raw(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Send tools/call and return the raw result object."""
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return _format_tool_result(self.call_tool_raw(name, arguments))


def _format_tool_result(result: dict[str, Any]) -> str:
    chunks: list[str] = []
    for block in result.get("content", []):
        kind = block.get("type")
        if kind == "text":
            chunks.append(block.get("text", ""))
        elif kind == "resource":
            resource = block.get("resource", {})
            chunks.append(resource.get("text") or f"[resource: {resource.get('uri', '?')}]")
        else:
            chunks.append(json.dumps(block, ensure_ascii=False))

    if not chunks and result.get("structuredContent") is not None:
        chunks.append(json.dumps(result["structuredContent"], ensure_ascii=False))

    text = "\n".join(chunk for chunk in chunks if chunk).strip() or "(no output)"
    return f"Error: {text}" if result.get("isError") else text


# ---------------------------------------------------------------------------
# JSON Schema -> pydantic -> LangChain tool
# ---------------------------------------------------------------------------

_SIMPLE_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _annotation(schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        if len(non_null) == 1:
            schema = {**schema, "type": non_null[0]}
            return _annotation(schema)
        return Any
    if schema_type in _SIMPLE_TYPES:
        return _SIMPLE_TYPES[schema_type]
    if schema_type == "array":
        return list[_annotation(schema.get("items") or {})]  # type: ignore[misc]
    if schema_type == "object":
        return dict[str, Any]
    return Any


def _field_description(schema: dict[str, Any]) -> str | None:
    description = schema.get("description")
    if schema.get("enum") is not None:
        choices = ", ".join(json.dumps(value) for value in schema["enum"])
        description = f"{description or ''} One of: {choices}.".strip()
    return description


def _sanitize_name(name: str, *, fallback: str) -> str:
    clean = _TOOL_NAME_RE.sub("_", name).strip("_") or fallback
    return clean[:64]


def _unique_name(base: str, used: set[str] | None) -> str:
    if used is None:
        return base[:64]
    candidate = base[:64]
    index = 2
    while candidate in used:
        suffix = f"_{index}"
        candidate = base[: 64 - len(suffix)] + suffix
        index += 1
    used.add(candidate)
    return candidate


def mcp_tool_to_langchain(
    client: StdioMCPClient,
    spec: dict[str, Any],
    *,
    name_prefix: str = "",
    used_names: set[str] | None = None,
) -> StructuredTool:
    """Convert one MCP tool descriptor into a LangChain StructuredTool."""
    original_name = spec.get("name") or "mcp_tool"
    base_name = _sanitize_name(f"{name_prefix}{original_name}", fallback="mcp_tool")
    tool_name = _unique_name(base_name, used_names)

    schema = spec.get("inputSchema") or {}
    properties: dict[str, Any] = schema.get("properties") or {}
    required: set[str] = set(schema.get("required") or [])

    fields: dict[str, tuple[Any, Any]] = {}
    to_mcp: dict[str, str] = {}
    for raw_name, prop in properties.items():
        py_name = _sanitize_name(raw_name, fallback="argument")
        if py_name in fields:
            continue
        to_mcp[py_name] = raw_name
        annotation = _annotation(prop)
        description = _field_description(prop)
        if raw_name in required:
            fields[py_name] = (annotation, Field(..., description=description))
        else:
            fields[py_name] = (
                annotation | None if annotation is not Any else Any,
                Field(default=None, description=description),
            )

    model_name = "ToolArgs_" + re.sub(r"[^a-zA-Z0-9_]", "_", tool_name)
    args_schema = create_model(model_name, **fields)  # type: ignore[call-overload]

    def _run(**kwargs: Any) -> str:
        arguments = {
            to_mcp[key]: value for key, value in kwargs.items() if value is not None
        }
        return client.call_tool(original_name, arguments)

    _run.__name__ = tool_name
    return StructuredTool.from_function(
        func=_run,
        name=tool_name,
        description=spec.get("description") or f"MCP tool {original_name}",
        args_schema=args_schema,
    )


@contextmanager
def mcp_tools(
    command: str | list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Iterator[list[StructuredTool]]:
    """Start an MCP stdio server and yield its tools as LangChain tools."""
    client = StdioMCPClient(command, env=env, timeout=timeout)
    client.start()
    try:
        yield [mcp_tool_to_langchain(client, spec) for spec in client.list_tools()]
    finally:
        client.close()


@contextmanager
def mcp_servers(
    specs: list[MCPServerSpec],
    *,
    timeout: float = 30.0,
) -> Iterator[dict[str, StdioMCPClient]]:
    """Start every MCP server and yield clients keyed by server name."""
    clients: dict[str, StdioMCPClient] = {}
    with ExitStack() as stack:
        for spec in specs:
            try:
                client = stack.enter_context(
                    StdioMCPClient(spec.command, timeout=timeout)
                )
            except Exception as exc:  # noqa: BLE001 - add server context
                raise MCPError(f"MCP server {spec.name!r} failed: {exc}") from exc
            clients[spec.name] = client
        yield clients


def load_langchain_tools(
    clients: dict[str, StdioMCPClient],
) -> list[StructuredTool]:
    """Convert all clients' tools, prefixing names when several servers exist."""
    tools: list[StructuredTool] = []
    used_names: set[str] = set()
    multiple = len(clients) > 1
    for server_name, client in clients.items():
        try:
            specs = client.list_tools()
        except Exception as exc:  # noqa: BLE001 - add server context
            raise MCPError(f"MCP server {server_name!r} tools/list failed: {exc}") from exc
        prefix = f"{server_name}__" if multiple else ""
        for spec in specs:
            tools.append(
                mcp_tool_to_langchain(
                    client,
                    spec,
                    name_prefix=prefix,
                    used_names=used_names,
                )
            )
    return tools
