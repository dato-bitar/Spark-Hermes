"""B6: the round queue survives a kill. Resuming re-grades what crashed and re-runs nothing that finished."""

from __future__ import annotations

import json
from pathlib import Path

import sh.validator.batch as batch

TASK = {"task_id": "t-1", "timeout_s": 60, "published": {"predicates": []}}


def _round(tmp_path: Path) -> Path:
    rd = tmp_path / "round"
    (rd / "tasks").mkdir(parents=True)
    (rd / "withheld").mkdir()
    (rd / "tasks" / "t-1.json").write_text(json.dumps(TASK))
    return rd


def test_a_finished_episode_is_never_run_twice(tmp_path, monkeypatch):
    runs = []

    def fake_run(task, bundle, image, inference, ep, **k):
        runs.append(ep)
        Path(ep).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(batch, "run_episode", fake_run)
    monkeypatch.setattr(batch, "grade", lambda ep, *a, **k: {"verified_success": True, "episode_dir": str(ep)})
    out = tmp_path / "out"
    first = batch.one(TASK, None, "null", None, "img", "url", out)
    again = batch.one(TASK, None, "null", None, "img", "url", out)
    assert len(runs) == 1 and again == first  # second call is a no-op replay


def test_an_episode_that_ran_but_failed_to_grade_is_only_re_graded(tmp_path, monkeypatch):
    """The V5 failure mode: grading crashed on 24 episodes that had each cost a GPU minute."""
    out = tmp_path / "out"
    ep = out / "null" / "t-1"
    ep.mkdir(parents=True)
    ep.joinpath("finish.json").write_text(json.dumps({"stage": "done", "api_calls": 5}))
    runs = []
    monkeypatch.setattr(batch, "run_episode", lambda *a, **k: runs.append(1))
    monkeypatch.setattr(batch, "grade", lambda *a, **k: {"verified_success": True})
    rec = batch.one(TASK, None, "null", None, "img", "url", out)
    assert runs == [] and rec["verified_success"]
    assert json.loads((ep / "episode.json").read_text())["verified_success"]


def test_a_half_written_episode_is_run_again(tmp_path, monkeypatch):
    out = tmp_path / "out"
    ep = out / "null" / "t-1"
    ep.mkdir(parents=True)
    ep.joinpath("finish.json").write_text(json.dumps({"stage": "no_finish"}))  # killed mid-run
    runs = []
    monkeypatch.setattr(batch, "run_episode", lambda *a, **k: runs.append(1))
    monkeypatch.setattr(batch, "grade", lambda *a, **k: {"verified_success": False})
    batch.one(TASK, None, "null", None, "img", "url", out)
    assert runs == [1]


def test_a_void_episode_is_not_cached_so_a_resume_re_runs_it(tmp_path, monkeypatch):
    """A provider outage must not be frozen into the archive as a failed attempt."""
    out = tmp_path / "out"
    runs = []

    def fake_run(task, bundle, image, inference, ep, **k):
        runs.append(ep)
        Path(ep).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(batch, "run_episode", fake_run)
    monkeypatch.setattr(
        batch, "grade", lambda *a, **k: {"verified_success": False, "void": True, "void_reason": "overloaded"}
    )
    rec = batch.one(TASK, None, "null", None, "img", "url", out)
    assert rec["void"]
    assert not (out / "null" / "t-1" / "episode.json").exists()
    batch.one(TASK, None, "null", None, "img", "url", out)
    assert len(runs) == 2  # re-run, not replayed
