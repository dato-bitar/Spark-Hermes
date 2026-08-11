"""Task DNA: what a real agent trace teaches, with none of what it said.

A public trace dataset is a record of agents doing real work. That record is worth mining, and it is
worth mining for exactly one thing: the *shape* of the problem. Domain, skills, how many tool calls
it took, which tools, what went wrong in the middle. Not the prompt, not the code, not the answer.

## Why the abstraction is the product

The obvious move -- hand a trace to a model and ask for "something similar" -- produces near
duplicates of the seed, inherits the teacher's habits, and carries the seed's licence into whatever
comes out. It also does not expand the skill space at all: a paraphrase of a Python debugging task is
another Python debugging task.

Mining the abstraction instead turns one behavioural seed ("find failing test, inspect, edit, rerun")
into as many concrete worlds as there are environments to put it in -- a CMake failure, a CUDA
indexing bug, a dependency regression, a serialization mismatch. The seed supplies the grammar; the
generator supplies the worlds.

## The invariant this module exists to enforce

**No verbatim content from a seed may travel in its DNA.** `assert_abstract` checks it rather than
trusting it, because the failure is silent: a DNA that quietly carries a sentence of the original
prompt produces tasks that are derivative in a way nobody notices until somebody diffs them against
the source. Seeds are read for structure and never enter the corpus as rows -- the licence of a seed
constrains what may be copied from it, and this is the line where that is decided.

Provenance travels too. `TaskDNA.source` names the dataset, its licence and the row, so a generated
task can always be traced back to the seed whose shape it borrowed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

# Substring length at which a match with the seed stops being a coincidence. Short technical phrases
# ("read the file", "the test fails") recur across any corpus of engineering text and flagging them
# would make the check unusable; a run this long shared with the source is quotation.
VERBATIM_WINDOW = 40


class DNAError(ValueError):
    """A seed could not be reduced to a usable abstraction."""


@dataclass(frozen=True)
class TaskDNA:
    """The shape of a task, with nothing of its content.

    Every field is either a category from a closed vocabulary, a count, or a short label the
    extractor produced -- never text lifted from the seed.
    """

    domain: str
    skills: tuple[str, ...]
    environment: str
    # 1..5. Derived from the horizon and whether the trace recovered from a failure, not claimed by
    # the seed: a difficulty a generator asserts about its own output is not a measurement. The real
    # difficulty of a generated task is measured later, by running the pinned model against it.
    difficulty: int
    # Tool calls the seed took, as a (low, high) band. The generator aims a new task at this horizon.
    horizon: tuple[int, int]
    failure_mode: str
    required_tools: tuple[str, ...]
    verification: str
    source: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= self.difficulty <= 5:
            raise DNAError(f"difficulty {self.difficulty} is outside 1..5")
        if self.horizon[0] > self.horizon[1]:
            raise DNAError(f"horizon {self.horizon} is inverted")
        if not self.required_tools:
            raise DNAError("a task with no tools cannot require execution, which is the whole point")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


# Closed vocabularies. A generator prompt built from free text drifts; these are the axes a task can
# vary along, and anything the extractor cannot place lands in the catch-all rather than inventing a
# category nobody downstream knows how to render.
DOMAINS = (
    "repository_engineering",
    "data_processing",
    "build_and_packaging",
    "system_administration",
    "numerical_computing",
    "text_and_encoding",
)
SKILLS = (
    "debugging",
    "multi_file_edit",
    "test_generation",
    "log_analysis",
    "dependency_resolution",
    "data_extraction",
    "ordering_and_sequencing",
    "unit_conversion",
    "error_recovery",
    "self_verification",
)
FAILURE_MODES = (
    "incorrect_assumption_about_api",
    "tool_unavailable",
    "misleading_documentation",
    "silent_truncation",
    "wrong_ordering",
    "off_by_one",
    "encoding_mismatch",
    "stale_cache",
)

_TOOL_ALIASES = {
    "bash": "terminal",
    "shell": "terminal",
    "run_command": "terminal",
    "execute": "terminal",
    "read_file": "file_read",
    "cat": "file_read",
    "view": "file_read",
    "write_file": "file_write",
    "create_file": "file_write",
    "edit": "file_write",
    "str_replace_editor": "file_write",
    "python": "python",
    "run_python": "python",
    "ipython": "python",
}

# What this harness can actually offer. A DNA asking for a browser produces a task the runtime cannot
# execute, so the extractor maps into this set or drops the seed.
HARNESS_TOOLS = ("terminal", "file_read", "file_write", "python")


def _normalise_tools(names: list[str]) -> tuple[str, ...]:
    mapped = []
    for raw in names:
        key = str(raw or "").strip().lower()
        tool = _TOOL_ALIASES.get(key, key)
        if tool in HARNESS_TOOLS and tool not in mapped:
            mapped.append(tool)
    return tuple(mapped)


_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("debugging", ("traceback", "assertionerror", "failing test", "pytest", "stack trace", "exception")),
    ("log_analysis", ("log", "logfile", ".log", "journalctl", "syslog")),
    ("dependency_resolution", ("requirements", "pip install", "package", "version conflict", "lockfile")),
    ("test_generation", ("write a test", "add a test", "test case", "unittest")),
    ("data_extraction", ("csv", "json", "parse", "extract", "records")),
    ("ordering_and_sequencing", ("sort", "order", "chronolog", "sequence", "timestamp")),
    ("unit_conversion", ("cents", "bytes", "milliseconds", "convert", "unit")),
    ("multi_file_edit", ("refactor", "rename across", "multiple files", "module")),
    ("encoding_mismatch", ("utf-8", "encoding", "unicode", "latin-1")),
)

_ENVIRONMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("python_repository", ("pytest", ".py", "import ", "pip")),
    ("c_or_cpp_build", ("cmake", "makefile", "gcc", "g++", ".cpp")),
    ("shell_workspace", ("bash", "shell", "directory", "ls ")),
    ("data_files", ("csv", "json", "gzip", "archive", "parquet")),
)


def _classify(text: str, table: tuple[tuple[str, tuple[str, ...]], ...], default: str) -> str:
    lowered = text.lower()
    scored = Counter()
    for label, needles in table:
        hits = sum(1 for n in needles if n in lowered)
        if hits:
            scored[label] = hits
    return scored.most_common(1)[0][0] if scored else default


def _skills_in(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    found = [label for label, needles in _SIGNALS if label in SKILLS and any(n in lowered for n in needles)]
    return tuple(found)


def _difficulty(calls: int, recovered: bool) -> int:
    """From the horizon and whether the trace had to recover, never from a claim.

    A seed that says "difficulty: 4" is asserting something about its own output. What the trace
    actually shows is how many actions it took and whether something went wrong in the middle, and
    both of those are observations.
    """
    base = 1 if calls <= 3 else 2 if calls <= 7 else 3 if calls <= 14 else 4
    return min(5, base + (1 if recovered else 0))


def extract(record: dict[str, Any], *, dataset: str, licence: str, index: int) -> TaskDNA:
    """One seed trace to one abstraction.

    Reads the task text and the tool calls, and keeps neither. Raises `DNAError` when the trace has
    no tool calls this harness can offer -- a seed that browsed the web teaches nothing a terminal
    can reproduce, and inventing a mapping would produce tasks that cannot be executed.
    """
    text = _seed_text(record)
    if not text.strip():
        raise DNAError("the seed carries no task text to classify")

    tools = _normalise_tools(_seed_tools(record))
    if not tools:
        raise DNAError("no tool in this seed maps onto the harness; it cannot teach an executable task")

    calls = _seed_call_count(record)
    recovered = _seed_recovered(record)
    skills = _skills_in(text) or ("debugging",)
    if recovered and "error_recovery" not in skills:
        skills = skills + ("error_recovery",)

    low = max(2, int(calls * 0.7))
    high = max(low + 2, int(calls * 1.4))

    dna = TaskDNA(
        domain=_classify(text, tuple((d, n) for d, n in _SIGNALS if d in DOMAINS), DOMAINS[0]),
        skills=skills,
        environment=_classify(text, _ENVIRONMENTS, "shell_workspace"),
        difficulty=_difficulty(calls, recovered),
        horizon=(low, high),
        failure_mode=_failure_mode(text, recovered),
        required_tools=tools,
        verification="deterministic script over the workspace",
        source={"dataset": dataset, "licence": licence, "row": index, "observed_tool_calls": calls},
    )
    assert_abstract(dna, text)
    return dna


def _failure_mode(text: str, recovered: bool) -> str:
    lowered = text.lower()
    for mode, needles in (
        ("tool_unavailable", ("not installed", "command not found", "unavailable")),
        ("misleading_documentation", ("readme", "docs say", "notes", "comment claims")),
        ("wrong_ordering", ("order", "sort", "chronolog")),
        ("encoding_mismatch", ("encoding", "utf-8", "unicode")),
        ("off_by_one", ("off by one", "index", "boundary")),
    ):
        if any(n in lowered for n in needles):
            return mode
    return "incorrect_assumption_about_api" if recovered else "misleading_documentation"


def assert_abstract(dna: TaskDNA, seed_text: str) -> None:
    """No run of `VERBATIM_WINDOW` characters from the seed may appear anywhere in the DNA.

    Checked rather than trusted. The failure is silent -- a DNA that carries a sentence of the seed
    produces tasks that are derivative in a way nobody notices until someone diffs them against the
    source -- and the licence of a seed constrains what may be copied from it, so this is the line
    where that is decided rather than a matter of good intentions.
    """
    rendered = json.dumps(dna.to_record(), sort_keys=True)
    haystack = re.sub(r"\s+", " ", seed_text)
    needle = re.sub(r"\s+", " ", rendered)
    for start in range(0, max(0, len(haystack) - VERBATIM_WINDOW) + 1):
        window = haystack[start : start + VERBATIM_WINDOW]
        if window in needle:
            raise DNAError(f"the DNA quotes {VERBATIM_WINDOW} characters of its seed: {window!r}")


# --- reading whatever shape a public dataset happens to use ------------------------------------


def _seed_text(record: dict[str, Any]) -> str:
    for key in ("task", "prompt", "instruction", "question", "problem"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("messages", "conversations", "conversation"):
        turns = record.get(key)
        if isinstance(turns, list):
            for turn in turns:
                if isinstance(turn, dict) and turn.get("role") in ("user", "human"):
                    content = turn.get("content") or turn.get("value")
                    if isinstance(content, str) and content.strip():
                        return content
    return ""


def _seed_tools(record: dict[str, Any]) -> list[str]:
    declared = record.get("tools") or record.get("tools_available")
    names: list[str] = []
    if isinstance(declared, list):
        for entry in declared:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict):
                fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
                if isinstance(fn.get("name"), str):
                    names.append(fn["name"])
    for turn in record.get("messages") or []:
        if isinstance(turn, dict):
            for call in turn.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call, dict) else None
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    names.append(fn["name"])
    return names


def _seed_call_count(record: dict[str, Any]) -> int:
    count = 0
    for turn in record.get("messages") or []:
        if isinstance(turn, dict):
            count += len(turn.get("tool_calls") or [])
    if count:
        return count
    steps = record.get("steps")
    return len(steps) // 3 if isinstance(steps, list) and steps else 4


def _seed_recovered(record: dict[str, Any]) -> bool:
    """Whether something in the trace actually failed and the agent kept going.

    This is the single most valuable bit in a seed: a trace that hit a wall and worked around it
    teaches a behaviour that a clean run does not, and it is what makes `error_recovery` a real
    skill rather than a label.
    """
    for turn in record.get("messages") or []:
        if not isinstance(turn, dict) or turn.get("role") != "tool":
            continue
        content = str(turn.get("content") or "").lower()
        if any(n in content for n in ("error", "traceback", "not found", "no such file", "failed", "exit 1")):
            return True
    return False


__all__ = [
    "DOMAINS",
    "FAILURE_MODES",
    "HARNESS_TOOLS",
    "SKILLS",
    "VERBATIM_WINDOW",
    "DNAError",
    "TaskDNA",
    "assert_abstract",
    "extract",
]
