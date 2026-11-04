"""Local tools for the agent, including a small sandboxed Python runner."""

from __future__ import annotations

import ast
import operator
import os
import subprocess
import sys
import tempfile

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Calculator: arithmetic only, evaluated via a whitelisted AST.
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPRESSION = 500
_MAX_POWER = 10_000


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER:
            raise ValueError("exponent is too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. "(12 + 4) * 3 / 2"."""
    if len(expression) > _MAX_EXPRESSION:
        return "Error: expression is too long."
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}"
    return str(result)


# ---------------------------------------------------------------------------
# Python runner: separate process, fresh temp dir, CPU/memory/file limits.
# ---------------------------------------------------------------------------

_SNIPPET_LIMIT = 8_000
_OUTPUT_LIMIT = 4_000
_TIMEOUT_SECONDS = 5


def _apply_limits() -> None:
    """Best-effort resource limits for the child process (POSIX only)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return

    limits = (
        (resource.RLIMIT_CPU, _TIMEOUT_SECONDS),
        (resource.RLIMIT_AS, 512 * 1024 * 1024),
        (resource.RLIMIT_FSIZE, 1024 * 1024),
        (resource.RLIMIT_NOFILE, 64),
    )
    for name, value in limits:
        try:
            resource.setrlimit(name, (value, value))
        except (ValueError, OSError):
            pass


@tool
def run_python(code: str) -> str:
    """Run a short Python snippet in an isolated subprocess and return its output.

    The snippet runs in a temporary directory with a 5 second CPU/wall-clock
    limit, no access to project files, and a 512 MB memory cap. It is still a
    local process, not a hardened security boundary.
    """
    if len(code) > _SNIPPET_LIMIT:
        return f"Error: code exceeds {_SNIPPET_LIMIT} characters."
    if os.name != "posix":
        return "Error: run_python is only supported on POSIX systems."

    with tempfile.TemporaryDirectory(prefix="agent-sandbox-") as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=workdir,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONIOENCODING": "utf-8",
                    "HOME": workdir,
                },
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SECONDS,
                preexec_fn=_apply_limits,
            )
        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {_TIMEOUT_SECONDS}s."

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    parts = []
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if not parts:
        parts.append(f"(no output, exit code {completed.returncode})")
    output = "\n".join(parts)
    if len(output) > _OUTPUT_LIMIT:
        output = output[:_OUTPUT_LIMIT] + "\n... (truncated)"
    return output


LOCAL_TOOLS = [calculator, run_python]
