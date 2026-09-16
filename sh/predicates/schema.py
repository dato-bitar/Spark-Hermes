"""The closed predicate vocabulary: what a check may assert, and nothing else.

A check is a list of predicates, each `[op, *args]` with single-line scalar arguments. Checks are
DATA. They are never rendered to shell and never authored by the served model; `evaluate.py` runs
them in Python inside the grading container. That is the whole point of the vocabulary: there is
no escaping rule to get wrong because nothing is ever interpolated into a command line -- the only
things that reach a shell are family-declared `commands`, referenced by index.

Adding an operator means adding a row to `OPS` and an evaluator in `evaluate.py`; nothing else in
the pipeline changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "OPS",
    "PATH",
    "TEXT",
    "INT",
    "NUMBER",
    "SHA256",
    "MODE",
    "CMD",
    "CHECK",
    "Predicate",
    "PredicateError",
    "parse",
]

PATH, TEXT, INT, NUMBER, SHA256, MODE, CMD, CHECK = "path", "text", "int", "number", "sha256", "mode", "cmd", "check"

# op -> (argument kinds, failure message template). `custom` is variadic: kinds beyond the first are TEXT.
OPS: dict[str, tuple[tuple[str, ...], str]] = {
    "file_exists": ((PATH,), "{0} does not exist"),
    "file_absent": ((PATH,), "{0} exists and should not"),
    "digest_is": ((PATH, SHA256), "{0} does not have the expected contents"),
    "line_count_is": ((PATH, INT), "{0} does not have {1} lines"),
    "text_equals": ((PATH, TEXT), "{0} does not read exactly {1}"),
    "number_is": ((PATH, NUMBER, NUMBER), "{0} is not {1} within {2}"),
    "grep_count": ((PATH, TEXT, INT), "{0} does not contain {1} on exactly {2} lines"),
    "ordering_is": ((PATH, TEXT, TEXT), "in {0}, {1} does not come before {2}"),
    "json_field_equals": ((PATH, TEXT, TEXT), "in {0}, field {1} is not {2}"),
    "perms_are": ((PATH, MODE), "{0} does not have mode {1}"),
    "exit_code_is": ((CMD, INT), "command {0} did not exit {1}"),
    "stdout_equals": ((CMD, TEXT), "command {0} did not print {1}"),
    "custom": ((CHECK,), "check {0} failed"),
}

# Predicates whose truth says nothing about CONTENT. A published half made only of these lets an
# empty file score as `overfit`; `derive` refuses that shape.
STRUCTURAL = frozenset({"file_exists", "file_absent"})

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_INT = re.compile(r"\A-?[0-9]+\Z")
_NUMBER = re.compile(r"\A-?[0-9]+(?:\.[0-9]+)?\Z")
_MODE = re.compile(r"\A[0-7]{3,4}\Z")
_CHECK_ID = re.compile(r"\A[a-z][a-z0-9_]{0,63}\Z")


class PredicateError(ValueError):
    """A predicate that cannot be built -- distinct from one that is false."""


@dataclass(frozen=True)
class Predicate:
    op: str
    args: tuple[str, ...]

    @property
    def message(self) -> str:
        template = OPS[self.op][1]
        return template.format(*self.args) if self.op != "custom" else template.format(self.args[0])

    def to_json(self) -> list:
        return [self.op, *self.args]


def _check_path(value: str, what: str) -> None:
    if not value:
        raise PredicateError(f"{what} is empty")
    if value.startswith("/"):
        raise PredicateError(f"{what} is absolute: {value[:80]!r}; paths are workspace-relative")
    if any(part in ("..", "") for part in value.split("/")):
        raise PredicateError(f"{what} climbs or has an empty segment: {value[:80]!r}")


def parse(item: list | tuple | Predicate) -> Predicate:
    """Build a `Predicate` from `[op, *args]`, refusing anything outside the vocabulary.

    Rejection here is free: it happens before any workspace exists, so a family author learns about
    a bad predicate when they write it, not when a round runs.
    """
    if isinstance(item, Predicate):
        return item
    if not isinstance(item, (list, tuple)) or not item:
        raise PredicateError("a predicate is a non-empty list [op, *args]")
    op, raw = item[0], list(item[1:])
    if op not in OPS:
        raise PredicateError(f"unknown operator {op!r}; the vocabulary is {', '.join(sorted(OPS))}")
    kinds = list(OPS[op][0])
    if op == "custom":
        if not raw:
            raise PredicateError("custom takes a check id and optional arguments")
        kinds += [TEXT] * (len(raw) - 1)
    if len(raw) != len(kinds):
        raise PredicateError(f"{op} takes {len(kinds)} argument(s), got {len(raw)}")

    args: list[str] = []
    for index, (value, kind) in enumerate(zip(raw, kinds), start=1):
        what = f"argument {index} of {op}"
        if kind in (INT, CMD) and isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str):
            raise PredicateError(f"{what} must be a string, got {type(value).__name__}")
        if any(ch in value for ch in "\n\r\0"):
            raise PredicateError(f"{what} must be a single line; pin multi-line content with digest_is")
        if kind == PATH:
            _check_path(value, what)
        elif kind == SHA256 and not _SHA256.match(value):
            raise PredicateError(f"{what} is not a lowercase hex sha256")
        elif kind == INT and not _INT.match(value):
            raise PredicateError(f"{what} is not an integer: {value[:40]!r}")
        elif kind == NUMBER and not _NUMBER.match(value):
            raise PredicateError(f"{what} is not a number: {value[:40]!r}")
        elif kind == MODE and not _MODE.match(value):
            raise PredicateError(f"{what} is not an octal mode: {value[:40]!r}")
        elif kind == CMD and (not _INT.match(value) or int(value) < 0):
            raise PredicateError(f"{what} is not a command index: {value[:40]!r}")
        elif kind == CHECK and not _CHECK_ID.match(value):
            raise PredicateError(f"{what} is not a check id: {value[:40]!r}")
        args.append(value)
    return Predicate(op, tuple(args))


def parse_all(items, *, commands: int = 0, checks: frozenset[str] | set[str] = frozenset()) -> tuple[Predicate, ...]:
    """Parse a whole check and resolve every command index and check id against what the family declares."""
    out = tuple(parse(item) for item in items)
    for p in out:
        kind = OPS[p.op][0][0]
        if kind == CMD and int(p.args[0]) >= commands:
            raise PredicateError(f"{p.op} references command {p.args[0]} but only {commands} are declared")
        if kind == CHECK and p.args[0] not in checks:
            raise PredicateError(f"custom references check {p.args[0]!r} which the family does not declare")
    return out
