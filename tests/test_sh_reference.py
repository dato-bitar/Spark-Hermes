"""The reference arms: same runner as a miner's episode, appended to an archive, and a stats view over it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sh.validator.batch as batch
from sh.reference.run import main, run_arms

TASK = {"task_id": "posix-report-r1-00", "family": "posix_report", "timeout_s": 60, "published": {"predicates": []}}


def _round(tmp_path: Path) -> Path:
    rd = tmp_path / "round"
    (rd / "tasks").mkdir(parents=True, exist_ok=True)
    (rd / "withheld").mkdir(exist_ok=True)
    (rd / "tasks" / f"{TASK['task_id']}.json").write_text(json.dumps(TASK))
    return rd


@pytest.fixture
def stub(monkeypatch):
    seen = []

    def fake_run(task, bundle, image, inference, ep, **k):
        seen.append((task["task_id"], bundle))
        Path(ep).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(batch, "run_episode", fake_run)
    monkeypatch.setattr(
        batch,
        "grade",
        lambda ep, task, w, image, *, bundle_dir, surface, round_id, **kw: {
            "task_id": task["task_id"],
            "family": task["family"],
            "surface": surface,
            "round_id": "r1",
            "verified_success": surface == "canon",
            "api_calls": 5,
            "tool_calls": 4,
            "wall_s": 1.0,
            "overfit": False,
            "disqualified": False,
        },
    )
    return seen


def test_both_arms_run_the_same_task_and_only_canon_carries_a_bundle(tmp_path, stub):
    recs = run_arms(_round(tmp_path), ["null", "canon"], tmp_path / "canon", "img", "url", tmp_path / "out")
    assert {r["surface"] for r in recs} == {"null", "canon"}
    assert sorted(b is None for _, b in stub) == [False, True]  # canon gets the prose, null gets nothing


def test_canon_without_a_surface_is_refused_rather_than_run_as_null(tmp_path, stub):
    with pytest.raises(SystemExit, match="canon-dir"):
        run_arms(_round(tmp_path), ["canon"], None, "img", "url", tmp_path / "out")


def test_the_archive_is_append_only_across_rounds(tmp_path, stub):
    out = tmp_path / "out"
    run_arms(_round(tmp_path), ["null"], None, "img", "url", out)
    (out / "null" / TASK["task_id"] / "episode.json").unlink()  # a later round, same file
    run_arms(_round(tmp_path), ["null"], None, "img", "url", out)
    assert len((out / "reference.jsonl").read_text().strip().splitlines()) == 2


def test_stats_mode_reads_the_archive_without_running_anything(tmp_path, stub, capsys):
    arch = tmp_path / "episodes.jsonl"
    arch.write_text(
        "\n".join(
            json.dumps(
                {
                    "family": "posix_report",
                    "round_id": "r1",
                    "surface": s,
                    "verified_success": s == "canon",
                    "api_calls": 5,
                    "tool_calls": 4,
                }
            )
            for s in ["null"] * 8 + ["canon"] * 8
        )
    )
    assert main(["--stats", "--archive", str(arch), "--family", "posix_report", "--window", "r1"]) == 0
    assert json.loads(capsys.readouterr().out)["canon"]["delta_c"] == 1.0
    assert stub == []
