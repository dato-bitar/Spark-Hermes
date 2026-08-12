"""The stages, what each one needs, and what it refuses to do without.

The pipeline is deliberately a state machine over a directory rather than one long script. Every
stage reads what the previous one wrote and writes a manifest of its own, so a run that dies at
hour six of a twelve-hour rollout resumes instead of restarting -- which is the difference between
an experiment you can iterate on and one you run once and defend forever.

    generate  ->  rollout  ->  corpus  ->  train  ->  evaluate

## Stages refuse rather than degrade

Each stage states what it needs and stops if it is absent. The alternative -- proceeding with less --
is how a corpus ends up built from three tasks, or an evaluation ends up run against a baseline
nobody measured. Both look like results.

## What is measured where

`rollout` runs the TRAIN tasks. `evaluate` runs the EVAL tasks, which are never trained on, and
compares before against after through `hermes.promotion`. Keeping those two commands separate is not
tidiness: a single `run everything` that happened to include the eval tasks in the rollout set would
produce a number nobody could distinguish from a real one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from admin.split import SplitError, refuse_eval_tasks

STAGES = ("generate", "rollout", "corpus", "train", "evaluate")

MANIFEST = "stage.json"


class StageError(RuntimeError):
    """A stage cannot run, and says what is missing rather than doing less."""


@dataclass
class Workspace:
    """A pipeline run on disk. Every stage writes into its own subdirectory.

    Gitignored by living under `var/`: a corpus is regenerable and a rollout log is large, and the
    repository is not a place to keep either.
    """

    root: Path
    tasks: Path = field(init=False)
    rollouts: Path = field(init=False)
    corpus: Path = field(init=False)
    models: Path = field(init=False)
    reports: Path = field(init=False)

    def __post_init__(self) -> None:
        self.tasks = self.root / "tasks"
        self.rollouts = self.root / "rollouts"
        self.corpus = self.root / "corpus"
        self.models = self.root / "models"
        self.reports = self.root / "reports"

    def stage_dir(self, stage: str) -> Path:
        return {
            "generate": self.tasks,
            "rollout": self.rollouts,
            "corpus": self.corpus,
            "train": self.models,
            "evaluate": self.reports,
        }[stage]

    def manifest_of(self, stage: str) -> dict[str, Any] | None:
        path = self.stage_dir(stage) / MANIFEST
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A half-written manifest is worse than none: it makes a stage look complete. Reported as
            # absent so the stage re-runs rather than being skipped on corrupt evidence.
            return None

    def record(self, stage: str, payload: dict[str, Any]) -> Path:
        directory = self.stage_dir(stage)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path


def status(workspace: Workspace) -> list[dict[str, Any]]:
    """What each stage has produced, in order, with the first unmet dependency named.

    The point of showing all five rather than only the next one is that a pipeline whose third stage
    was re-run has a fourth stage holding stale output, and a status that reported only "next: train"
    would hide it.
    """
    rows = []
    for stage in STAGES:
        manifest = workspace.manifest_of(stage)
        rows.append(
            {
                "stage": stage,
                "done": manifest is not None,
                "summary": (manifest or {}).get("summary", ""),
                "at": (manifest or {}).get("completed_at", ""),
            }
        )
    return rows


def _require(workspace: Workspace, stage: str) -> dict[str, Any]:
    manifest = workspace.manifest_of(stage)
    if manifest is None:
        raise StageError(f"stage {stage!r} has not run; {workspace.stage_dir(stage) / MANIFEST} does not exist")
    return manifest


def generated_task_ids(workspace: Workspace) -> list[str]:
    """Task ids the generator produced and the gate accepted."""
    manifest = _require(workspace, "generate")
    return list(manifest.get("accepted") or [])


def run_rollouts(
    workspace: Workspace,
    *,
    repeats: int,
    base_url: str,
    model: str,
    concurrency: int = 12,
    dialect: str = "atem",
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """N attempts on every accepted TRAIN task, through the real harness.

    `refuse_eval_tasks` runs first and raises rather than filtering. A rollout set that quietly
    included an eval task would produce a corpus that is correct and a count that is wrong, and the
    operator would go on believing they trained on what they asked for.
    """
    accepted = generated_task_ids(workspace)
    if not accepted:
        raise StageError("the generate stage accepted no tasks; there is nothing to roll out")
    refuse_eval_tasks(accepted)

    workspace.rollouts.mkdir(parents=True, exist_ok=True)
    episodes = workspace.rollouts / "episodes.jsonl"
    command = [
        sys.executable,
        "-m",
        "hermesbench.runner",
        "--suite",
        "generated",
        "--task-ids",
        ",".join(accepted),
        "--workspace-root",
        str(workspace.rollouts / "ws"),
        "--model",
        model,
        "--base-url",
        base_url,
        "--dialect",
        dialect,
        "--repeats",
        str(repeats),
        "--concurrency",
        str(concurrency),
        "--keep-trajectories",
        "--allow-unsandboxed",
        "--episodes-out",
        str(episodes),
        "--out",
        str(workspace.rollouts / "manifest.json"),
    ]
    done = runner(command, capture_output=True, text=True)
    if getattr(done, "returncode", 1) != 0:
        raise StageError(f"the runner exited {done.returncode}: {(done.stderr or '')[-2000:]}")
    return {"tasks": len(accepted), "repeats": repeats, "episodes": str(episodes)}


def build_corpus(workspace: Workspace, *, max_pairs_per_task: int = 8) -> dict[str, Any]:
    """Rollouts to SFT rows and preference pairs, with the split enforced at the boundary.

    The cap defaults lower than `validator.aggregate`'s 32. Measured on a real 152-episode run, 33 of
    36 pairs came from two tasks; at that concentration a preference set teaches those two tasks and
    calls it a policy.
    """
    from hermes.challenge import episode_metrics_of
    from validator.aggregate import Episode, aggregate

    _require(workspace, "rollout")
    episodes_path = workspace.rollouts / "episodes.jsonl"
    if not episodes_path.is_file():
        raise StageError(f"{episodes_path} does not exist; the rollout stage recorded no episodes")

    episodes: list[Episode] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = episode_metrics_of(row)
        public = bool(metrics.get("public_passed"))
        hidden = metrics.get("hidden_passed")
        episodes.append(
            Episode(
                task_id=str(metrics.get("task_id") or ""),
                round_id="admin",
                miner_id="operator",
                verified=public and hidden is not False,
                overfit=public and hidden is False,
                tokens=int(metrics.get("tokens_used") or 0),
                trajectory=row.get("trajectory") or {},
                dialect=str(metrics.get("dialect") or ""),
                truncated=bool(metrics.get("max_steps_hit")),
            )
        )
    if not episodes:
        raise StageError(f"{episodes_path} holds no episodes; refusing to write an empty corpus")

    # The last line of defence. Everything upstream should have kept eval tasks out; this is where
    # a mistake anywhere upstream becomes visible instead of becoming training data.
    refuse_eval_tasks(sorted({e.task_id for e in episodes}))

    summary = aggregate(episodes, workspace.corpus, max_per_task=max_pairs_per_task)
    return summary.to_record()


def evaluate_command(workspace: Workspace, *, base_url: str, model: str, dialect: str = "atem") -> list[str]:
    """The command that measures a model on the EVAL suite.

    Returned rather than run, because this is the number everything else is judged by and it should
    be launched deliberately -- with the withheld environment set, on a quiet machine, by someone who
    means to. A convenience wrapper that ran it as a side effect of another stage is how a baseline
    gets measured under load and then compared against one that was not.
    """
    return [
        sys.executable,
        "-m",
        "hermesbench.runner",
        "--suite",
        "v0,v1",
        "--workspace-root",
        str(workspace.reports / "ws"),
        "--model",
        model,
        "--base-url",
        base_url,
        "--dialect",
        dialect,
        "--keep-trajectories",
        "--allow-unsandboxed",
        "--repeats",
        "10",
        "--episodes-out",
        str(workspace.reports / "episodes.jsonl"),
        "--out",
        str(workspace.reports / "manifest.json"),
    ]


__all__ = [
    "MANIFEST",
    "STAGES",
    "SplitError",
    "StageError",
    "Workspace",
    "build_corpus",
    "evaluate_command",
    "generated_task_ids",
    "run_rollouts",
    "status",
]
