"""Bundle lint and canonical digest (spec §4.2–4.3).

A bundle is **prose, not code**: a `SOUL.md` and optional skills. Everything the lint refuses is a way of
turning prose back into execution or into a channel out of the sandbox — inline shell, an environment
interpolation, a URL to fetch. The rules are cheap and total, so a miner gets the same answer locally that the
validator will give, before anything is submitted.

The digest is the thing a commitment is made over, so its definition is frozen: changing how it is computed
invalidates every commitment ever made.

    python -m sh.cli.lint DIR            # human output, exit 1 on any problem
    python -m sh.cli.lint DIR --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MAX_SKILL = 15 * 1024
MAX_FILE = 64 * 1024
MAX_BUNDLE = 512 * 1024

SOUL = re.compile(r"^SOUL\.md$")
SKILL = re.compile(r"^skills/(?P<n>[a-z0-9][a-z0-9-]{0,63})/SKILL\.md$")
REFERENCE = re.compile(r"^skills/(?P<n>[a-z0-9][a-z0-9-]{0,63})/references/[A-Za-z0-9._-]+\.md$")

DENIED_NAMES = ("mcp.json", ".env", "config.yaml", "AGENTS.md")
DENIED_DIRS = ("scripts/", "memories/")
URLISH = re.compile(r"https?://|www\.|\b\d{1,3}(?:\.\d{1,3}){3}\b")
INLINE_SHELL = "!`"
ENV_INTERP = "${HERMES_"
# Patterns Hermes' own `guard_agent_created` refuses at the pin: a bundle must not try to look like something
# the agent wrote for itself, which is how a skill would escape the write-approval staging.
GUARD_PATTERNS = ("agent-created", "agent_created", "<!-- hermes:generated")
# Skill names Hermes ships at the pin. A bundle taking one of these shadows the built-in silently.  [D]
BUNDLED_SKILLS = ("skill-creator", "memory", "todo", "web", "browser", "discord", "session-search")


def _frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    out = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def collect(root: Path) -> tuple[dict, list[str]]:
    """Every regular file under `root` as `{relative path: bytes}`, plus refusals for what cannot be read."""
    files, problems = {}, []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            problems.append(f"L2 {rel}: symlink")
            continue
        if not path.is_file():
            continue
        files[rel] = path.read_bytes()
    return files, problems


def bundle_digest(files: dict) -> str:
    """Frozen by golden test: changing this invalidates every commitment ever made. The NULL surface is `{}`."""
    body = "\n".join(f"{path}\0{hashlib.sha256(files[path]).hexdigest()}" for path in sorted(files))
    return hashlib.sha256(body.encode()).hexdigest()


def lint(files: dict) -> list[str]:
    problems: list[str] = []
    total = sum(len(b) for b in files.values())
    if total > MAX_BUNDLE:
        problems.append(f"L3 bundle is {total} bytes, over the {MAX_BUNDLE} limit")
    if not files:
        problems.append("L1 bundle is empty; submit a SOUL.md or at least one skill")

    for path in sorted(files):
        data = files[path]
        is_soul, is_skill, is_reference = SOUL.match(path), SKILL.match(path), REFERENCE.match(path)
        if not (is_soul or is_skill or is_reference):
            problems.append(f"L1 {path}: not SOUL.md, skills/<name>/SKILL.md or skills/<name>/references/*.md")
            continue
        if path in DENIED_NAMES or any(path.startswith(d) for d in DENIED_DIRS):
            problems.append(f"L2 {path}: denied path")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"L2 {path}: not valid UTF-8")
            continue
        if "\r\n" in text:
            problems.append(f"L2 {path}: CRLF line endings")
        limit = MAX_SKILL if is_skill else MAX_FILE
        if len(data) > limit:
            problems.append(f"L3 {path}: {len(data)} bytes, over the {limit} limit")
        if INLINE_SHELL in text:
            problems.append(f"L4 {path}: inline shell marker `!``")
        if ENV_INTERP in text:
            problems.append(f"L5 {path}: environment interpolation ${{HERMES_")
        if m := URLISH.search(text):
            problems.append(f"L6 {path}: network reference {m.group(0)!r}")
        for pattern in GUARD_PATTERNS:
            if pattern in text.lower():
                problems.append(f"L8 {path}: looks agent-created ({pattern!r})")
        if is_skill:
            name = is_skill.group("n")
            front = _frontmatter(text)
            if front.get("name") != name:
                problems.append(f"L7 {path}: frontmatter name {front.get('name')!r} != directory {name!r}")
            if len(front.get("description", "")) > 500:
                problems.append(f"L7 {path}: description over 500 characters")
            if name in BUNDLED_SKILLS:
                problems.append(f"L9 {path}: {name!r} shadows a skill Hermes ships at the pin")
    return problems


def check(root: Path) -> dict:
    files, problems = collect(root)
    problems += lint(files)
    return {
        "schema": "sh-lint-v2",
        "bundle": str(root),
        "files": sorted(files),
        "bytes": sum(len(b) for b in files.values()),
        "bundle_sha256": bundle_digest(files),
        "ok": not problems,
        "problems": problems,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="lint a strategy bundle and print its canonical digest")
    ap.add_argument("bundle")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.bundle)
    if not root.is_dir():
        print(f"{root} is not a directory", file=sys.stderr)
        return 2
    result = check(root)
    if a.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"bundle   {root}")
        print(f"files    {len(result['files'])} ({result['bytes']} bytes)")
        print(f"digest   {result['bundle_sha256']}")
        if result["ok"]:
            print("lint     ok")
        else:
            print(f"lint     {len(result['problems'])} problem(s):")
            for p in result["problems"]:
                print(f"  - {p}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
