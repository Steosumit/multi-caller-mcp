"""Configurable fake MCP stdio server used by the test suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "name": "add",
        "description": "Add two numbers and return a JSON result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "first number"},
                "b": {"type": "number", "description": "second number"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "word_count",
        "description": "Count the words in a piece of text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "text to count"},
                "punctuation": {
                    "type": "boolean",
                    "description": "keep punctuation",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "echo",
        "description": "Return the supplied text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "fail",
        "description": "Always returns an execution error.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
    },
    {
        "name": "slow",
        "description": "Sleep for the requested number of seconds.",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
    },
    {
        "name": "weather",
        "description": "Return structured weather data.",
        "inputSchema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "description": "celsius"},
                "conditions": {"type": "string"},
            },
            "required": ["temperature", "conditions"],
        },
    },
]


def _content(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }


def _call_tool(name: str, args: dict[str, Any], label: str) -> dict[str, Any]:
    if name == "add":
        total = args["a"] + args["b"]
        return _content(
            json.dumps({"server": label, "sum": total, "expression": f"{args['a']}+{args['b']}"})
        )
    if name == "word_count":
        return _content(str(len(str(args["text"]).split())))
    if name == "echo":
        return _content(f"{label}:{args.get('text', '')}")
    if name == "fail":
        return _content(args.get("message", f"{label} failed"), is_error=True)
    if name == "slow":
        time.sleep(float(args.get("seconds", 1)))
        return _content(f"{label} slept")
    if name == "weather":
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"temperature": 22.5, "conditions": "sunny"}),
                }
            ],
            "structuredContent": {"temperature": 22.5, "conditions": "sunny"},
            "isError": False,
        }
    return _content(f"unknown tool {name}", is_error=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="fake")
    parser.add_argument("--name", default="fake-mcp")
    args = parser.parse_args()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict) or "id" not in message:
            continue
        request_id = message["id"]
        method = message.get("method")

        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": args.name, "version": "1.0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = message.get("params") or {}
            result = _call_tool(
                str(params.get("name")),
                params.get("arguments") or {},
                args.label,
            )
        else:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"unknown method {method}"},
                    }
                ),
                flush=True,
            )
            continue

        print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
