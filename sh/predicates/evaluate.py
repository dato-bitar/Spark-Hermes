"""Evaluate a check against a finished workspace, in Python, inside the grading container.

Two entry points. `evaluate` answers every predicate and returns the full bitmap -- what
derivation needs, because "this predicate is false" is the answer there, not an error. `judge`
is the grader's conjunction: it stops at the first false predicate and names it.

Nothing here builds a command line from predicate arguments. The only shell that ever runs is a
family-declared `commands[i]`, chosen by index; `custom` dispatches to a family-declared Python
callable. A predicate argument is compared, never interpolated.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .schema import OPS, Predicate, parse_all

__all__ = ["Verdict", "evaluate", "judge"]

CheckFn = Callable[..., bool]


@dataclass(frozen=True)
class Verdict:
    ok: bool
    bits: tuple[bool, ...]
    failed_index: int | None
    message: str


def _inside(workspace: Path, rel: str) -> Path | None:
    """The real path for `rel`, or None if it resolves outside the workspace.

    An agent can leave `report.sh -> /etc/passwd` behind; a digest over the link target would then
    be a digest over the host. Resolving and re-checking containment makes such a file "missing".
    """
    root = workspace.resolve()
    target = (workspace / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def _bytes(workspace: Path, rel: str) -> bytes | None:
    target = _inside(workspace, rel)
    if target is None or not target.is_file():
        return None
    return target.read_bytes()


def _lines(data: bytes) -> list[bytes]:
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def _scalar(node) -> str:
    if node is True:
        return "true"
    if node is False:
        return "false"
    if node is None:
        return "null"
    if isinstance(node, (str, int, float)):
        return str(node)
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def _run(workspace: Path, command: str, timeout_s: int) -> tuple[int, bytes]:
    env = {
        "PATH": f"{workspace / 'bin'}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(workspace),
        "LANG": "C.UTF-8",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        done = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return 124, b""
    return done.returncode, done.stdout


def _one(
    p: Predicate,
    workspace: Path,
    commands: Sequence[str],
    checks: Mapping[str, CheckFn],
    timeout_s: int,
) -> bool:
    op, a = p.op, p.args
    if op == "file_exists":
        t = _inside(workspace, a[0])
        return t is not None and t.is_file()
    if op == "file_absent":
        raw = workspace / a[0]
        return not raw.exists() and not raw.is_symlink()
    data = None
    if OPS[op][0][0] == "path":
        data = _bytes(workspace, a[0])
        if data is None:
            return False
    if op == "digest_is":
        return hashlib.sha256(data).hexdigest() == a[1]
    if op == "line_count_is":
        count = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
        return count == int(a[1])
    if op == "text_equals":
        text = data.decode("utf-8", "surrogateescape")
        return text == a[1] or text == a[1] + "\n"
    if op == "number_is":
        try:
            value = float(data.decode("utf-8", "surrogateescape").strip())
        except ValueError:
            return False
        return abs(value - float(a[1])) <= float(a[2])
    if op == "grep_count":
        needle = a[1].encode("utf-8", "surrogateescape")
        return sum(1 for line in _lines(data) if needle in line) == int(a[2])
    if op == "ordering_is":
        first, second = (a[1].encode(), a[2].encode())
        lines = _lines(data)
        ia = next((i for i, line in enumerate(lines) if first in line), None)
        ib = next((i for i, line in enumerate(lines) if second in line), None)
        return ia is not None and ib is not None and ia < ib
    if op == "json_field_equals":
        try:
            node = json.loads(data)
            for key in a[1].split("."):
                node = node[int(key)] if isinstance(node, list) else node[key]
        except (ValueError, KeyError, IndexError, TypeError):
            return False
        return _scalar(node) == a[2]
    if op == "perms_are":
        target = _inside(workspace, a[0])
        return target is not None and (target.stat().st_mode & 0o7777) == int(a[1], 8)
    if op == "exit_code_is":
        code, _ = _run(workspace, commands[int(a[0])], timeout_s)
        return code == int(a[1])
    if op == "stdout_equals":
        _, out = _run(workspace, commands[int(a[0])], timeout_s)
        return out.decode("utf-8", "surrogateescape").rstrip("\n") == a[1]
    if op == "custom":
        try:
            return bool(checks[a[0]](workspace, *a[1:]))
        except Exception:
            # A family check that raises has failed to establish the fact; it has not established
            # the opposite. False is the only honest answer, and `derive` will notice a check that
            # is false on the reference.
            return False
    raise AssertionError(op)


def evaluate(
    check: Sequence,
    workspace: str | Path,
    *,
    commands: Sequence[str] = (),
    checks: Mapping[str, CheckFn] | None = None,
    timeout_s: int = 60,
) -> list[bool]:
    """One boolean per predicate, in order. Never stops early; never raises for a false predicate."""
    checks = checks or {}
    predicates = parse_all(check, commands=len(commands), checks=frozenset(checks))
    workspace = Path(workspace)
    return [_one(p, workspace, commands, checks, timeout_s) for p in predicates]


def judge(
    check: Sequence,
    workspace: str | Path,
    *,
    commands: Sequence[str] = (),
    checks: Mapping[str, CheckFn] | None = None,
    timeout_s: int = 60,
) -> Verdict:
    """The conjunction, stopping at the first false predicate and naming it."""
    checks = checks or {}
    predicates = parse_all(check, commands=len(commands), checks=frozenset(checks))
    workspace = Path(workspace)
    bits: list[bool] = []
    for index, p in enumerate(predicates):
        ok = _one(p, workspace, commands, checks, timeout_s)
        bits.append(ok)
        if not ok:
            return Verdict(False, tuple(bits), index, p.message)
    return Verdict(True, tuple(bits), None, "")
