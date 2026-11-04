"""Offline tests for the code-mode bridge and multi-server namespacing."""

from __future__ import annotations

import shlex
import shutil
import sys
import unittest
from pathlib import Path

from codemode import (
    CodeModeExecutor,
    ToolInfo,
    collect_tools,
    generate_typescript_api,
)
from mcp_converter import (
    MCPServerSpec,
    load_langchain_tools,
    mcp_servers,
    parse_mcp_specs,
)

PYTHON = sys.executable
FAKE = Path(__file__).with_name("fake_mcp_server.py")
ROOT = Path(__file__).resolve().parents[1]


def fake_spec(name: str, label: str | None = None) -> MCPServerSpec:
    command = f"{shlex.quote(PYTHON)} {shlex.quote(str(FAKE))} --label {label or name}"
    return MCPServerSpec(name, command)


class SpecTests(unittest.TestCase):
    def test_single_unnamed_server(self) -> None:
        specs = parse_mcp_specs(["npx -y server"])
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "mcp")
        self.assertEqual(specs[0].command, "npx -y server")

    def test_named_servers(self) -> None:
        specs = parse_mcp_specs(["alpha=cmd one", "beta=cmd two"])
        self.assertEqual([spec.name for spec in specs], ["alpha", "beta"])
        self.assertEqual(specs[1].command, "cmd two")

    def test_multiple_servers_require_names(self) -> None:
        with self.assertRaisesRegex(Exception, "name each one"):
            parse_mcp_specs(["cmd one", "cmd two"])

    def test_duplicate_names_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "duplicate"):
            parse_mcp_specs(["x=one", "x=two"])


class ApiGenerationTests(unittest.TestCase):
    def test_nested_namespaces_and_types(self) -> None:
        tools = [
            ToolInfo(
                server="alpha",
                original_name="add",
                name="add",
                description="Add two numbers.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "first"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            ),
            ToolInfo(
                server="beta",
                original_name="word-count",
                name="word_count",
                description="Count words.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "punctuation": {"type": "boolean"},
                    },
                    "required": ["text"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                },
            ),
        ]
        api = generate_typescript_api(tools)

        self.assertIn("interface AlphaAddInput", api)
        self.assertIn("a: number;", api)
        self.assertIn("interface BetaWordCountInput", api)
        self.assertIn("punctuation?: boolean;", api)
        self.assertIn("interface CodeModeRootApi", api)
        self.assertIn("alpha: AlphaApi;", api)
        self.assertIn("beta: BetaApi;", api)
        self.assertIn("word_count: (input: BetaWordCountInput)", api)
        self.assertIn("declare const codemode: CodeModeRootApi;", api)

    def test_schema_features(self) -> None:
        tools = [
            ToolInfo(
                server="srv",
                original_name="demo",
                name="demo",
                description="Demo.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["fast", "safe"]},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": ["integer", "null"]},
                    },
                    "required": ["mode"],
                },
            )
        ]
        api = generate_typescript_api(tools)
        self.assertIn('"fast" | "safe"', api)
        self.assertIn("tags?: string[];", api)
        self.assertIn("limit?: number | null;", api)


class CodeModeTests(unittest.TestCase):
    def test_multi_server_routing_and_namespacing(self) -> None:
        with mcp_servers([fake_spec("alpha"), fake_spec("beta")]) as clients:
            tools = collect_tools(clients)
            executor = CodeModeExecutor(clients, tools, sandbox="none", timeout=15)
            output = executor.execute(
                """
                const a = await codemode.alpha.add({ a: 19, b: 23 });
                const b = await codemode.beta.add({ a: 2, b: 3 });
                console.log(a.server, a.sum);
                console.log(b.server, b.sum);
                """
            )
        self.assertIn("alpha 42", output)
        self.assertIn("beta 5", output)
        self.assertEqual(executor.stats.calls, 2)
        self.assertEqual(executor.stats.servers["alpha"], 1)
        self.assertEqual(executor.stats.servers["beta"], 1)

    def test_direct_mode_namespaced_tools(self) -> None:
        with mcp_servers([fake_spec("alpha"), fake_spec("beta")]) as clients:
            tools = load_langchain_tools(clients)
            names = {tool.name for tool in tools}
            self.assertIn("alpha__add", names)
            self.assertIn("beta__add", names)
            result = next(tool for tool in tools if tool.name == "alpha__add").invoke(
                {"a": 1, "b": 2}
            )
        self.assertIn('"alpha"', result)

    def test_markdown_fences_are_stripped(self) -> None:
        with mcp_servers([fake_spec("alpha")]) as clients:
            executor = CodeModeExecutor(
                clients, collect_tools(clients), sandbox="none", timeout=15
            )
            output = executor.execute(
                "```ts\nconsole.log(await codemode.alpha.echo({ text: 'hi' }));\n```"
            )
        self.assertIn("alpha:hi", output)

    def test_unawaited_async_main_is_drained(self) -> None:
        with mcp_servers([fake_spec("alpha")]) as clients:
            executor = CodeModeExecutor(
                clients, collect_tools(clients), sandbox="none", timeout=15
            )
            output = executor.execute(
                """
                async function main() {
                  const result = await codemode.alpha.add({ a: 19, b: 23 });
                  console.log(result.sum);
                }
                main();
                """
            )
        self.assertIn("42", output)

    def test_structured_content(self) -> None:
        with mcp_servers([fake_spec("alpha")]) as clients:
            executor = CodeModeExecutor(
                clients, collect_tools(clients), sandbox="none", timeout=15
            )
            output = executor.execute(
                "console.log((await codemode.alpha.weather({ location: 'x' })).temperature);"
            )
        self.assertIn("22.5", output)

    def test_tool_error_is_catchable(self) -> None:
        with mcp_servers([fake_spec("alpha")]) as clients:
            executor = CodeModeExecutor(
                clients, collect_tools(clients), sandbox="none", timeout=15
            )
            output = executor.execute(
                """
                try {
                  await codemode.alpha.fail({ message: "boom" });
                } catch (error) {
                  console.log("caught", (error as Error).message);
                }
                """
            )
        self.assertIn("caught boom", output)

    def test_unknown_tool_rejected(self) -> None:
        with mcp_servers([fake_spec("alpha")]) as clients:
            executor = CodeModeExecutor(
                clients, collect_tools(clients), sandbox="none", timeout=15
            )
            output = executor.execute("await codemode.alpha.nope({});")
        self.assertIn("Unknown tool", output)

    def test_call_limit(self) -> None:
        with mcp_servers([fake_spec("alpha")]) as clients:
            executor = CodeModeExecutor(
                clients,
                collect_tools(clients),
                sandbox="none",
                timeout=15,
                max_calls=1,
            )
            output = executor.execute(
                """
                await codemode.alpha.add({ a: 1, b: 2 });
                await codemode.alpha.add({ a: 3, b: 4 });
                """
            )
        self.assertIn("call limit", output)

    def test_execution_timeout(self) -> None:
        with mcp_servers([fake_spec("alpha")]) as clients:
            executor = CodeModeExecutor(
                clients, collect_tools(clients), sandbox="none", timeout=1.5
            )
            output = executor.execute("await codemode.alpha.slow({ seconds: 5 });")
        self.assertIn("timed out", output.lower())

    def test_output_is_truncated(self) -> None:
        with mcp_servers([fake_spec("alpha")]) as clients:
            executor = CodeModeExecutor(
                clients,
                collect_tools(clients),
                sandbox="none",
                timeout=15,
                max_output=40,
            )
            output = executor.execute("console.log('x'.repeat(100));")
        self.assertIn("truncated", output)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and shutil.which("bwrap"),
        "bwrap sandbox is only tested on Linux",
    )
    def test_bwrap_blocks_network_files_and_secrets(self) -> None:
        with mcp_servers([fake_spec("alpha")]) as clients:
            executor = CodeModeExecutor(
                clients, collect_tools(clients), sandbox="bwrap", timeout=10
            )
            output = executor.execute(
                f"""
                console.log("secret:", process.env.LLM_API_KEY ?? "none");
                try {{
                  const fs = await import("node:fs");
                  fs.readFileSync({str(ROOT / ".env")!r});
                  console.log("fs: open");
                }} catch {{
                  console.log("fs: blocked");
                }}
                try {{
                  await fetch("http://127.0.0.1:1234/v1/models", {{
                    signal: AbortSignal.timeout(1000),
                  }});
                  console.log("net: open");
                }} catch {{
                  console.log("net: blocked");
                }}
                """
            )
        self.assertIn("secret: none", output)
        self.assertIn("fs: blocked", output)
        self.assertIn("net: blocked", output)


if __name__ == "__main__":
    unittest.main()
