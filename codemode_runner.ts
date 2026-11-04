// Bun harness for code mode.
//
// The Python side starts this file with the generated user script as argv[2].
// Tool calls travel to Python as JSON lines on stdout; Python writes results
// back on stdin. User output is buffered and returned in the final "done"
// message so that it cannot corrupt the protocol stream.

import { writeSync } from "node:fs";
import { createInterface } from "node:readline";
import { pathToFileURL } from "node:url";

type ServerMap = Record<string, string[]>;
type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
};

const servers: ServerMap = JSON.parse(process.env.CODEMODE_SERVERS ?? "{}");
const maxLogs = Math.max(1, Number(process.env.CODEMODE_MAX_LOGS ?? "50") || 50);
const logs: string[] = [];
let logCount = 0;
let finished = false;

function emit(payload: Record<string, unknown>): void {
  // Bypass the console/stdout overrides below.
  writeSync(1, JSON.stringify(payload) + "\n");
}

function format(value: unknown): string {
  if (typeof value === "string") return value;
  if (value instanceof Error) return value.message;
  try {
    const bun = (globalThis as { Bun?: { inspect?: Function } }).Bun;
    if (bun?.inspect) {
      return String(bun.inspect(value, { depth: 6, colors: false }));
    }
  } catch {
    // Fall through to JSON.
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function addLog(prefix: string, args: unknown[]): void {
  if (logCount >= maxLogs) return;
  logCount += 1;
  const text = args.map(format).join(" ");
  if (logCount === maxLogs) {
    logs.push(`${prefix}${text}\n... (log limit reached)`);
  } else {
    logs.push(`${prefix}${text}`);
  }
}

// Capture everything the generated script might print. The real protocol uses
// writeSync(1), so these overrides are safe.
console.log = (...args: unknown[]) => addLog("", args);
console.info = (...args: unknown[]) => addLog("", args);
console.debug = (...args: unknown[]) => addLog("", args);
console.warn = (...args: unknown[]) => addLog("warn: ", args);
console.error = (...args: unknown[]) => addLog("error: ", args);
(process.stdout as unknown as { write: (chunk: unknown) => boolean }).write = (
  chunk: unknown
) => {
  addLog("", [String(chunk).replace(/\n$/, "")]);
  return true;
};

const pending = new Map<number, Pending>();
let nextId = 0;

function call(server: string, tool: string, input: unknown): Promise<unknown> {
  const tools = servers[server];
  if (!tools) {
    return Promise.reject(new Error(`Unknown MCP server: ${server}`));
  }
  if (!tools.includes(tool)) {
    return Promise.reject(new Error(`Unknown tool: ${server}.${tool}`));
  }
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    emit({ type: "call", id, server, tool, args: input ?? {} });
  });
}

const reader = createInterface({ input: process.stdin, crlfDelay: Infinity });
reader.on("line", (line) => {
  if (!line.trim()) return;
  let message: Record<string, unknown>;
  try {
    message = JSON.parse(line);
  } catch {
    return;
  }
  if (message.type !== "result") return;
  const id = Number(message.id);
  const waiter = pending.get(id);
  if (!waiter) return;
  pending.delete(id);
  if (message.ok) {
    waiter.resolve(message.value);
  } else {
    waiter.reject(new Error(String(message.error ?? "MCP tool call failed")));
  }
});

reader.on("close", () => {
  for (const waiter of pending.values()) {
    waiter.reject(new Error("code-mode runtime closed before the tool replied"));
  }
  pending.clear();
});

const codemode: Record<string, unknown> = {};
for (const [server, tools] of Object.entries(servers)) {
  codemode[server] = new Proxy(
    {},
    {
      get(_target, property) {
        if (typeof property !== "string") return undefined;
        if (!tools.includes(property)) {
          return () =>
            Promise.reject(new Error(`Unknown tool: ${server}.${property}`));
        }
        return (input: unknown) => call(server, property, input);
      },
      has(_target, property) {
        return typeof property === "string" && tools.includes(property);
      },
      ownKeys() {
        return tools;
      },
      getOwnPropertyDescriptor() {
        return { enumerable: true, configurable: true };
      },
    }
  );
}
(globalThis as Record<string, unknown>).codemode = codemode;

function finish(payload: Record<string, unknown>): void {
  if (finished) return;
  finished = true;
  emit({ ...payload, logs });
  reader.close();
  process.exit(0);
}

process.on("uncaughtException", (error) => {
  finish({ type: "error", error: format(error) });
});
process.on("unhandledRejection", (reason) => {
  finish({ type: "error", error: `Unhandled rejection: ${format(reason)}` });
});

const codePath = process.argv[2];
if (!codePath) {
  finish({ type: "error", error: "missing generated script path" });
} else {
  try {
    await import(pathToFileURL(codePath).href);
    // Models often wrap code in `async function main() {...} main();` without
    // awaiting it. Give pending calls and logs a short window to settle before
    // declaring the script finished.
    await drainPending();
    finish({ type: "done" });
  } catch (error) {
    finish({ type: "error", error: format(error) });
  }
}

async function drainPending(): Promise<void> {
  const started = Date.now();
  let quietSince = Date.now();
  while (Date.now() - started < 2_000) {
    if (pending.size === 0) {
      if (Date.now() - quietSince >= 100) return;
    } else {
      quietSince = Date.now();
    }
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}
