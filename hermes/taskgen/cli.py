"""Generate tasks at scale: seeds in, gated tasks out, every rejection counted.

    python -m hermes.taskgen.cli --count 350 --out var/tasks/gen-1 \\
        --base-url http://127.0.0.1:8001/v1 --model qwen3.8-27b

Parallel because generation is I/O bound on a served model and the gate is process bound, and the
two overlap. Resumable because a run of several hundred will be interrupted, and re-generating a
task that already passed costs a GPU minute for nothing.

## The acceptance rate is the headline, not a footnote

A generator that reports "300 tasks written" and not "300 of 900 attempts, 412 killed by the
disagreement check" tells an operator nothing about whether their prompt is working. The failure
histogram is where the information is: heavy `checks_disagree_on_a_cheat` means the instruction is
not conveying what a withheld check is for; heavy `setup_is_deterministic` means it is not conveying
determinism; heavy `parse` means the model is not following the output format at all and no amount
of GPU time will fix it.

## Nothing is written before it is accepted

An accepted task's YAML, its withheld check and its salted commitment are written together, after
the gate. A directory of maybe-tasks is worse than no directory: the point of the gate is that
everything downstream can trust what is in here without re-checking it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from hermes.taskgen.dna import TaskDNA, similarity
from hermes.taskgen.gate import Verdict, gate
from hermes.taskgen.seeds import SOURCES, SeedStats, read
from hermes.taskgen.synth import SynthError, synthesise, to_task_yaml

# Above this, two task prompts are the same task told twice. Chosen against the real corpus: two
# hand-written tasks from the same family (tc-log-rotation-order and tc-nested-archive-manifest, both
# ordering traps over a directory of files) score 0.06, so a threshold this high cannot mistake
# "same skill" for "same task" -- which is the distinction that matters, since a corpus wants many
# tasks per skill and no task twice.
DUPLICATE_AT = 0.5


def _is_duplicate(prompt: str, accepted_prompts: list[str]) -> tuple[bool, float]:
    """Whether this task has already been generated, and how close the nearest one is."""
    if not accepted_prompts:
        return False, 0.0
    nearest = max(similarity(prompt, other) for other in accepted_prompts)
    return nearest >= DUPLICATE_AT, nearest


def _action_budget(dna: TaskDNA) -> int:
    """Room to work, not the expected number of calls.

    `max(6, horizon[1])` set the budget to the seed's own upper estimate, which leaves an agent no
    slack for looking around, being wrong once, or checking its work -- and this harness counts
    verification against a separate allowance precisely because it wants that behaviour. Measured on a
    real probe: a task budgeted at 6 hit `step budget exhausted (6)` after three productive calls, and
    a truncated episode is scored as a failure of the agent.

    Twice the upper estimate, floored at 12. Generous on purpose: the step budget is not where
    difficulty should come from -- a trap in the workspace is -- and a task made hard by an
    ungenerous budget measures the budget.
    """
    return max(12, dna.horizon[1] * 2)


def task_id_for(index: int, dna: TaskDNA) -> str:
    """Stable and readable: the domain says what it is, the number says which attempt made it."""
    return f"gen-{dna.domain.split('_')[0][:4]}-{index:04d}"


def _write_accepted(
    out: Path,
    withheld_out: Path,
    synthesised: Any,
    *,
    salt: str,
    max_steps: int,
) -> dict[str, Any]:
    from hermes.harness import derive_task_salt, salted_digest

    task_id = synthesised.candidate.task_id
    body = synthesised.candidate.withheld_verify
    # Under the PER-TASK salt derived from the master, never the master itself: revealing one spent
    # task's salt must not make every commitment still sealed brute-forceable.
    commitment = salted_digest(body, derive_task_salt(salt, task_id))

    out.mkdir(parents=True, exist_ok=True)
    withheld_out.mkdir(parents=True, exist_ok=True)
    (out / f"{task_id}.yaml").write_text(
        to_task_yaml(synthesised, commitment=commitment, max_steps=max_steps), encoding="utf-8"
    )
    (withheld_out / f"{task_id}.sh").write_text(body, encoding="utf-8")
    # The reference solution is kept beside the withheld check, not with the task. It is the proof
    # the task is solvable and it is also a complete answer, so it lives on the private side.
    (withheld_out / f"{task_id}.solution.sh").write_text(synthesised.candidate.reference_solution, encoding="utf-8")
    return {"task_id": task_id, "commitment": commitment}


def _save_reject(rejects: Path, task_id: str, verdict: Verdict | None, synthesised: Any, error: str) -> None:
    """Everything about a rejected attempt, so the histogram is actionable rather than decorative.

    A count of `setup_exits_zero` says the instruction is not producing runnable scripts. It does not
    say WHY, and without the script and the shell's own complaint the only way to find out is to
    generate more and read them by hand -- which is what this exists to stop.

    Written for rejects only. An accepted task's artefacts are already on the public side.
    """
    rejects.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "error": error,
        "failed_check": getattr(verdict, "failed_check", ""),
        "detail": getattr(verdict, "detail", ""),
        "checks_run": list(getattr(verdict, "checks_run", [])),
    }
    (rejects / f"{task_id}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if isinstance(synthesised, SynthError):
        # A parse failure has no candidate, only a reply. Writing it is the whole point: `parse: 9`
        # with nothing beside it cannot be diagnosed.
        (rejects / f"{task_id}.reply.txt").write_text(synthesised.raw or "<empty content>", encoding="utf-8")
        if synthesised.reasoning:
            (rejects / f"{task_id}.reasoning.txt").write_text(synthesised.reasoning, encoding="utf-8")
        return
    if synthesised is not None:
        candidate = synthesised.candidate
        for name, script in (
            ("setup", candidate.setup),
            ("verify", candidate.verify),
            ("withheld", candidate.withheld_verify),
            ("reference", candidate.reference_solution),
            ("cheat", candidate.cheat_solution),
        ):
            (rejects / f"{task_id}.{name}.sh").write_text(script, encoding="utf-8")


def _attempt(dna: TaskDNA, index: int, complete: Any, *, gate_timeout: int) -> tuple[str, Verdict | None, Any]:
    task_id = task_id_for(index, dna)
    try:
        synthesised = synthesise(dna, task_id=task_id, complete=complete)
    except SynthError as exc:
        # A reply that could not be read is a distinct outcome from a task that was read and failed.
        # Folding them together hides the case where the model has stopped following the format,
        # which is the one case more GPU time cannot fix. The reply travels with the error so the
        # histogram can be acted on rather than only counted.
        return f"parse: {exc}", None, exc
    except Exception as exc:  # noqa: BLE001 - a served model can fail in many ways; none should stop the run
        return f"generate: {type(exc).__name__}: {exc}", None, None
    verdict = gate(synthesised.candidate, timeout_s=gate_timeout)
    return "", verdict, synthesised


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=50, help="tasks to ACCEPT, not attempts to make")
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means count * 4")
    parser.add_argument("--source", default="lambda", choices=sorted(SOURCES))
    parser.add_argument("--out", type=Path, default=Path("var/tasks/gen-1"))
    parser.add_argument("--withheld-out", type=Path, default=None, help="defaults to <out>/withheld")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gate-timeout", type=int, default=120)
    parser.add_argument("--salt-file", type=Path, default=None, help="master withheld salt; required to write")
    parser.add_argument("--temperature", type=float, default=1.0, help="task variety wants sampling, not greedy")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="ceiling on one generation. Without one, a reply that starts repeating runs to the 32k "
        "context limit, which at this throughput is the ~900s the timeouts were measuring -- a "
        "runaway now ends as an unparseable reply in a fraction of the time. Set at 10k rather than "
        "6k because 6k truncated real replies mid-section: `parse` jumped to 5 of 12 rejections, "
        "which trades a slow waste for a fast one. A task with a heredoc-heavy setup genuinely "
        "needs the room.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=900,
        help="seconds per generation. The default of 300 in `openai_completion` is tuned for an "
        "agent turn; one generation here is six shell scripts, and at concurrency 8 the server "
        "queues them. Measured on a real run: 8 of 29 attempts died on APITimeoutError -- 28% of "
        "the GPU time spent, discarded, for a client setting rather than anything about the task.",
    )
    args = parser.parse_args(argv)

    import os as _os

    if (
        not _os.environ.get(args.api_key_env, "").strip()
        and "://" in args.base_url
        and "127.0.0.1" not in args.base_url
    ):
        # A remote endpoint with no credential produces one 401 per attempt, and a gateway that sees
        # a run of them rate-limits the caller. Measured: 20 attempts, 20 failures, then a 120-second
        # block -- caused by a shell prefix assignment that was expanded before it took effect, so the
        # key arrived empty. The histogram said `generate: 20`, which is true and says nothing.
        print(
            f"hermes.taskgen: ${args.api_key_env} is empty and --base-url is remote ({args.base_url}). "
            "Every attempt would fail authentication and a gateway will rate-limit the run for it. "
            "Export the key first.",
            file=sys.stderr,
        )
        return 2

    if args.salt_file is None or not args.salt_file.is_file():
        print(
            "hermes.taskgen: --salt-file is required. Every accepted task publishes a commitment to "
            "its withheld check, and a commitment needs the master salt. Without one the tasks would "
            "have to ship their withheld check in the clear, which is not a withheld check.",
            file=sys.stderr,
        )
        return 2
    salt = args.salt_file.read_text(encoding="utf-8").strip()

    withheld_out = args.withheld_out or (args.out / "withheld")
    rejects_dir = args.out / "rejected"
    already = {path.stem for path in args.out.glob("*.yaml")} if args.out.is_dir() else set()
    if already:
        # Flushed, like the per-acceptance lines. Under nohup stdout is a pipe and therefore block
        # buffered, so this sat unwritten while stderr's warnings appeared above it -- which reads as
        # a resume that did not happen, on the one line whose whole job is to say that it did.
        print(f"resuming: {len(already)} task(s) already accepted in {args.out}", flush=True)

    import os

    from hermesbench.policy import openai_completion

    complete = openai_completion(
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get(args.api_key_env, ""),
        timeout_s=args.request_timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    max_attempts = args.max_attempts or args.count * 4
    stats = SeedStats()
    pool = read(args.source, limit=max_attempts, stats=stats)

    # Seeded with what a previous run already wrote, so `--count` means "this many tasks in the
    # directory" rather than "this many MORE". Counting only the current session's acceptances made a
    # resumed run target 150 on top of the 116 it had just found -- and the progress line said 13/150
    # while 129 files sat on disk, which is the kind of number nobody re-derives.
    accepted: list[dict[str, Any]] = [{"task_id": task_id} for task_id in sorted(already)]
    prompts: list[str] = []
    failures: Counter[str] = Counter()
    attempted = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending = {}
        for index, dna in enumerate(pool):
            if len(accepted) >= args.count:
                break
            if task_id_for(index, dna) in already:
                continue
            pending[executor.submit(_attempt, dna, index, complete, gate_timeout=args.gate_timeout)] = index
            attempted += 1
            if len(pending) < args.concurrency * 2:
                continue
            for future in as_completed(list(pending)):
                del pending[future]
                error, verdict, synthesised = future.result()
                if error:
                    failures[error.split(":")[0]] += 1
                    _save_reject(rejects_dir, f"attempt-{index:04d}", verdict, synthesised, error)
                elif verdict is not None and verdict.accepted and _is_duplicate(synthesised.prompt, prompts)[0]:
                    failures["duplicate"] += 1
                    _save_reject(rejects_dir, synthesised.candidate.task_id, verdict, synthesised, "duplicate")
                elif verdict is not None and verdict.accepted:
                    prompts.append(synthesised.prompt)
                    accepted.append(
                        _write_accepted(
                            args.out, withheld_out, synthesised, salt=salt, max_steps=_action_budget(synthesised.dna)
                        )
                    )
                    print(f"  accepted {accepted[-1]['task_id']} ({len(accepted)}/{args.count})", flush=True)
                elif verdict is not None:
                    # `or "unnamed"` because an empty bucket is unreadable and, worse, was hiding a
                    # real bug: accepted-past-target tasks were landing here with no failed_check.
                    failures[verdict.failed_check or "unnamed"] += 1
                    _save_reject(rejects_dir, synthesised.candidate.task_id, verdict, synthesised, "")
                break

        for future in as_completed(list(pending)):
            error, verdict, synthesised = future.result()
            if error:
                failures[error.split(":")[0]] += 1
            elif verdict is not None and verdict.accepted and _is_duplicate(synthesised.prompt, prompts)[0]:
                failures["duplicate"] += 1
            elif verdict is not None and verdict.accepted:
                # Written even past the target. Submission already stopped at `--count`, so what is
                # still in flight is bounded by the concurrency -- and discarding a task that cleared
                # nine executed checks to keep a round number is the wrong trade. An earlier version
                # dropped these AND counted them as rejections under an empty name, which made the
                # generator look worse than it was and hid that the work was being thrown away.
                prompts.append(synthesised.prompt)
                accepted.append(
                    _write_accepted(
                        args.out, withheld_out, synthesised, salt=salt, max_steps=_action_budget(synthesised.dna)
                    )
                )
            elif verdict is not None:
                failures[verdict.failed_check or "unnamed"] += 1
                _save_reject(rejects_dir, synthesised.candidate.task_id, verdict, synthesised, "")

    # The rate is over THIS session's work. Seeding `accepted` with the resumed ids made `--count`
    # mean the right thing and immediately made this ratio mean the wrong one: 160 accepted over 49
    # attempted reported an acceptance rate of 3.265. A resumed task was not attempted here, so it
    # cannot be in the numerator of a rate whose denominator is attempts.
    fresh = len(accepted) - len(already)
    report = {
        "accepted": [entry["task_id"] for entry in accepted],
        "attempted": attempted,
        "resumed": len(already),
        "accepted_this_session": fresh,
        "acceptance_rate": round(fresh / attempted, 3) if attempted else 0.0,
        "rejected_by": dict(failures.most_common()),
        "seeds": stats.to_record(),
        "out": str(args.out),
        "rejected_dir": str(rejects_dir),
        "withheld_out": str(withheld_out),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    if already:
        print(f"{len(accepted)} task(s) in {args.out}: {len(already)} resumed, {fresh} added here")
    print(f"accepted {fresh} of {attempted} attempt(s) this session  ({report['acceptance_rate']:.0%})")
    for check, count in failures.most_common():
        print(f"  {count:>4}  {check}")
    print(f"\nwrote {args.out}/stage.json")
    if failures:
        print(f"every rejected attempt's scripts and the shell's own complaint are in {rejects_dir}")
    if failures.get("checks_disagree_on_a_cheat", 0) > len(accepted):
        print(
            "\nMost rejections are the disagreement check. That is the instruction failing to convey "
            "what a withheld check is FOR, not the model failing to write shell: it keeps producing a "
            "second copy of the published check. Sharpen requirement 3 before spending more GPU time."
        )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
