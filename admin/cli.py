"""The operator's entry point.

    python -m admin.cli status --root var/admin/run-1
    python -m admin.cli rollout --root var/admin/run-1 --repeats 8 --base-url http://127.0.0.1:8001/v1
    python -m admin.cli corpus  --root var/admin/run-1
    python -m admin.cli evaluate --root var/admin/run-1 --model qwen3.8-27b --print-only

`evaluate` prints its command rather than running it. The eval suite is what every claim about this
model rests on, so it gets launched deliberately -- with the withheld environment set, on a quiet
machine -- and not as a side effect of a stage that was really about something else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from admin.pipeline import StageError, Workspace, build_corpus, evaluate_command, run_rollouts, status
from admin.split import SplitError


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "rollout", "corpus", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path("var/admin/run-1"))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--dialect", default="atem")
    parser.add_argument("--max-pairs-per-task", type=int, default=8)
    parser.add_argument("--print-only", action="store_true", help="evaluate: print the command, do not run it")
    args = parser.parse_args(argv)

    workspace = Workspace(args.root)

    if args.command == "status":
        print(f"pipeline at {workspace.root}")
        for row in status(workspace):
            mark = "done" if row["done"] else "  --"
            print(f"  [{mark}] {row['stage']:<9} {row['summary']}")
        missing = [r["stage"] for r in status(workspace) if not r["done"]]
        if missing:
            print(f"\nnext: {missing[0]}")
            if missing[0] == "generate":
                print(
                    "  `generate` needs hermes.taskgen.synth, which is not built yet. Until it is,\n"
                    "  write accepted task ids into tasks/stage.json under an `accepted` key by hand."
                )
        return 0

    try:
        if args.command == "rollout":
            payload = run_rollouts(
                workspace,
                repeats=args.repeats,
                base_url=args.base_url,
                model=args.model,
                concurrency=args.concurrency,
                dialect=args.dialect,
            )
            payload["summary"] = f"{payload['tasks']} task(s) x {payload['repeats']}"
        elif args.command == "corpus":
            payload = build_corpus(workspace, max_pairs_per_task=args.max_pairs_per_task)
            payload["summary"] = f"{payload['sft_rows']} SFT row(s), {payload['preference_pairs']} pair(s)"
        else:
            command = evaluate_command(workspace, base_url=args.base_url, model=args.model, dialect=args.dialect)
            print(" ".join(command))
            if args.print_only:
                return 0
            print(
                "\nRefusing to run it from here. Set SPARKDISTILL_WITHHELD_ROOT and\n"
                "HERMESBENCH_WITHHELD_SALT and run the line above: without them the eval scores only\n"
                "the published half, and a model tuned on generated tasks is exactly the case where\n"
                "the withheld half is the number that matters.",
                file=sys.stderr,
            )
            return 1
    except (StageError, SplitError) as exc:
        print(f"admin: {exc}", file=sys.stderr)
        return 2

    payload["completed_at"] = _stamp()
    path = workspace.record(args.command, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
