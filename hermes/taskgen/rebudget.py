"""Raise the action budget on tasks already written, and say which ones moved.

    python -m hermes.taskgen.rebudget --tasks var/tasks/gen-1 --apply

`cli._action_budget` fixed the formula for tasks generated from now on. It does nothing for the ones
already on disk, and those were written with `max(6, horizon[1])` -- the seed's own upper estimate as
a ceiling, leaving an agent no room to look around, be wrong once, or check its work.

Measured on a 320-episode probe of 160 such tasks: **30% of all episodes truncated**, and 11 of the
14 tasks that looked "hard" had failed with every attempt hitting the cap. Their traps were never
tested; the budget was. That is a resource limit being read as a capability measurement, which is the
same defect this harness had internally when it charged reasoning steps against the action budget --
found twice in one night, from two directions.

## The new value is derivable from the old one

The old budget was `max(6, horizon[1])` and the new one is `max(12, horizon[1] * 2)`, so the new one
is exactly `max(12, old * 2)`:

  * `horizon[1] >= 6` -> old is `horizon[1]`, and `max(12, 2 * horizon[1])` is the new rule
  * `horizon[1] < 6`  -> old is 6, `max(12, 12)` is 12, and the new rule also floors at 12

So the horizon does not need re-mining from the seed, which matters because the task YAML never
recorded it.

## Why a separate tool rather than a regeneration

A generated task is a workspace, a prompt, two checks and two solutions that together passed nine
executed checks. The budget is one integer beside all of that. Throwing the rest away to change it
would cost a night of GPU time and produce different tasks, which is not the same experiment.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

# `max_steps: 42` at the start of a line. Rewritten with a regex rather than a YAML round-trip on
# purpose: these files carry comments explaining what each task traps, and every YAML dumper in
# Python discards them.
_MAX_STEPS_RE = re.compile(r"^(max_steps:\s*)(\d+)\s*$", re.M)

FLOOR = 12


def new_budget(old: int) -> int:
    """The committed `_action_budget` rule, expressed in terms of the value it replaces."""
    return max(FLOOR, old * 2)


@dataclass
class Change:
    task_id: str
    before: int
    after: int


def truncated_tasks(episodes: Path) -> set[str]:
    """Task ids where at least one probe attempt hit the cap.

    This is the whole argument for a targeted raise. Applying the new formula to every task on disk
    doubles budgets that were never binding -- one task went from 58 to 116 -- and a budget far above
    what a task needs lets a wasteful trajectory run to the end, which is the signal `tool_efficiency`
    is trying to read. Raise it where the measurement showed it binding, and nowhere else.
    """
    import json

    from hermes.challenge import episode_metrics_of

    hit: set[str] = set()
    for line in episodes.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        metrics = episode_metrics_of(json.loads(line))
        if metrics.get("max_steps_hit"):
            hit.add(str(metrics.get("task_id") or ""))
    return hit


def rebudget(tasks: Path, *, apply: bool = False, only: set[str] | None = None) -> list[Change]:
    """Every task whose budget would move, and move it when asked.

    Dry by default. A tool that rewrites a hundred task definitions on being run is a tool that
    rewrites them once by accident, and the tasks are the input to every measurement downstream.

    `only` restricts it to named tasks -- in practice the ones a probe showed truncating.
    """
    changes: list[Change] = []
    for path in sorted(tasks.glob("*.yaml")):
        if only is not None and path.stem not in only:
            continue
        text = path.read_text(encoding="utf-8")
        match = _MAX_STEPS_RE.search(text)
        if match is None:
            continue
        before = int(match.group(2))
        after = new_budget(before)
        if after == before:
            continue
        changes.append(Change(task_id=path.stem, before=before, after=after))
        if apply:
            path.write_text(_MAX_STEPS_RE.sub(rf"\g<1>{after}", text, count=1), encoding="utf-8")
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="write the change; otherwise report only")
    parser.add_argument(
        "--only-truncated",
        type=Path,
        default=None,
        help="an episodes.jsonl; raise the budget only on tasks that actually hit the cap. Without it "
        "every task is raised, including ones whose budget was never binding -- and a budget far above "
        "what a task needs lets a wasteful trajectory run to the end, which is what tool_efficiency "
        "is trying to measure.",
    )
    args = parser.parse_args(argv)

    only = truncated_tasks(args.only_truncated) if args.only_truncated else None
    if only is not None:
        print(f"{len(only)} task(s) hit the cap in {args.only_truncated}")
    changes = rebudget(args.tasks, apply=args.apply, only=only)
    if not changes:
        print(f"no task in {args.tasks} needs a larger budget")
        return 0

    widest = max(len(c.task_id) for c in changes)
    for change in sorted(changes, key=lambda c: c.before):
        print(f"  {change.task_id:<{widest}}  {change.before:>3} -> {change.after:>3}")
    verb = "raised" if args.apply else "would raise"
    print(f"\n{verb} the action budget on {len(changes)} task(s)")
    if not args.apply:
        print("dry run; pass --apply to write")
    return 0


__all__ = ["FLOOR", "Change", "main", "new_budget", "rebudget"]


if __name__ == "__main__":
    raise SystemExit(main())
