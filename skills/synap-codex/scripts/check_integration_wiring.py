#!/usr/bin/env python3
"""Report how a project actually uses Synap: does it only write, or does recall reach a prompt?

    python scripts/check_integration_wiring.py [PROJECT_DIR]

Static and stdlib only. It scans .py, .ts, .tsx, .js, .mjs files (skipping node_modules,
.venv, .git, dist, build, .agents, .claude) for Synap calls and prints:

  WRITE sites : record_message, record_messages_batch, ingest_transcript, memories.create,
                send_message, record_thinking, record_tool_call, record_tool_result,
                addMemory (the places the app sends memory to Synap)
  READ sites  : context.fetch, fetch(...) on the SDK, get_context_for_prompt,
                get_profile, fetchUserContext, getContextForPrompt (the places it reads back)
  DEAD READS  : Python functions that perform a read but are never called anywhere else
                (a recall() that exists and is never used)
  STREAM      : whether listen() is called

It cannot prove the recalled text reaches the model prompt. It does prove the cheap failures:
no read at all, or a read helper nobody calls. After it passes, still drive the app's own
endpoint and look at the prompt it builds. Exit 1 when the integration writes but never reads
or has a dead read helper; exit 0 otherwise (a read-only or no-op project exits 0 with a note).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

SKIP = {"node_modules", ".venv", "venv", ".git", "dist", "build", ".agents", ".claude",
        "__pycache__", ".mypy_cache", ".next", "site-packages"}
EXT = {".py", ".ts", ".tsx", ".js", ".mjs"}
WRITE_RE = re.compile(r"\b(record_messages_batch|record_message|ingest_transcript|record_thinking|"
                      r"record_tool_call|record_tool_result|send_message|addMemory|"
                      r"memories\.create|memories\.create_from_file)\s*\(")
READ_RE = re.compile(r"\b(get_context_for_prompt|getContextForPrompt|get_profile|getProfile|"
                     r"fetchUserContext|fetchCustomerContext|context\.fetch)\s*\(|"
                     r"\.(?:user|customer|client|conversation)\.context\.fetch\s*\(|\bsdk\.fetch\s*\(")
LISTEN_RE = re.compile(r"\blisten\s*\(")


def files(root: Path):
    for p in root.rglob("*"):
        if p.suffix in EXT and p.is_file() and not (set(p.relative_to(root).parts) & SKIP):
            yield p


def dead_reads(root: Path) -> list[str]:
    """Python functions that contain a Synap read but whose name is never referenced elsewhere."""
    defs: dict[str, str] = {}
    all_text = ""
    for p in files(root):
        if p.suffix != ".py":
            continue
        text = p.read_text(errors="ignore")
        all_text += "\n" + text
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = ast.get_source_segment(text, node) or ""
                if READ_RE.search(body):
                    defs[node.name] = f"{p.relative_to(root)}:{node.lineno}"
    out = []
    for name, where in defs.items():
        uses = len(re.findall(r"\b%s\b" % re.escape(name), all_text))
        if uses <= 1:  # only the definition itself
            out.append(f"{name} ({where})")
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[0] if argv else ".").resolve()
    writes, reads, listen = [], [], []
    for p in files(root):
        text = p.read_text(errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith(("#", "//", "*")):
                continue
            rel = f"{p.relative_to(root)}:{i}"
            if WRITE_RE.search(line):
                writes.append(rel)
            if READ_RE.search(line):
                reads.append(rel)
            if LISTEN_RE.search(line) and "instance" in line:
                listen.append(rel)
    dead = dead_reads(root)
    print(f"WRITE sites: {len(writes)}" + (f"  e.g. {writes[0]}" if writes else ""))
    print(f"READ sites : {len(reads)}" + (f"  e.g. {reads[0]}" if reads else ""))
    print(f"DEAD READS : {', '.join(dead) if dead else 'none'}")
    print(f"STREAM     : {'listen() called at ' + listen[0] if listen else 'not wired'}")
    bad = False
    if writes and not reads:
        print("PROBLEM: the project writes to Synap but never reads. Wire recall into the model "
              "prompt, or tell the human it is write-only and why.")
        bad = True
    if dead:
        print("PROBLEM: a read helper exists but nothing calls it. Call it where the prompt is built.")
        bad = True
    if not writes and not reads:
        print("NOTE: no Synap calls found under this directory.")
    print("Still check by hand: the recalled text reaches the model prompt, and memory informs "
          "while your own rules decide.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
