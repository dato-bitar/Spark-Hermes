"""A family's `custom` predicates only mean anything if their semantics reach the grading container.

`checks.py` lives outside the workspace, is read only by the grader, and never enters the boundary the agent
runs in — so a `custom` check can assert things the agent must not be able to read (spec §6).
"""

from __future__ import annotations

import json
from pathlib import Path

import sh.validator.grade as g


def test_the_family_checks_file_is_placed_in_the_grading_volume(tmp_path, monkeypatch):
    sent = {}

    def fake_run(args, *, input=None, timeout=None):
        class R:
            returncode = 0
            stdout = json.dumps({"published_pass": True, "protected_modified": []}).encode()
            stderr = b""

            def check_returncode(self):
                pass

        if input:
            sent["tar"] = input
        return R()

    monkeypatch.setattr(g, "_run", fake_run)
    checks = tmp_path / "checks.py"
    checks.write_text("CHECKS = {'pid_agrees': lambda ws: True}\n")
    (tmp_path / "snapshot.tar").write_bytes(b"")
    task = {"task_id": "t-1", "commands": [], "timeout_s": 60, "protected_paths": [], "published": {"predicates": []}}

    g.grade_in_container(tmp_path, task, None, "img", checks)

    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(sent["tar"])) as tf:
        names = tf.getnames()
        assert "checks.py" in names
        assert b"pid_agrees" in tf.extractfile("checks.py").read()


def test_a_family_without_custom_predicates_sends_no_checks_file(tmp_path, monkeypatch):
    sent = {}

    def fake_run(args, *, input=None, timeout=None):
        class R:
            returncode = 0
            stdout = json.dumps({"published_pass": True, "protected_modified": []}).encode()
            stderr = b""

            def check_returncode(self):
                pass

        if input:
            sent["tar"] = input
        return R()

    monkeypatch.setattr(g, "_run", fake_run)
    (tmp_path / "snapshot.tar").write_bytes(b"")
    task = {"task_id": "t-1", "commands": [], "timeout_s": 60, "protected_paths": [], "published": {"predicates": []}}

    g.grade_in_container(tmp_path, task, None, "img")

    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(sent["tar"])) as tf:
        assert "checks.py" not in tf.getnames()


def test_the_in_container_grader_loads_checks_and_refuses_a_malformed_file(tmp_path, monkeypatch):
    """Exercises the loader itself, which runs inside the grading container."""
    import importlib.util

    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent / "sh/validator/runner"))
    spec = importlib.util.spec_from_file_location(
        "runner_grade", Path(__file__).parent.parent / "sh/validator/runner/grade.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("SH_EP", str(tmp_path))
    spec.loader.exec_module(module)

    assert module._checks() == {}  # no file: a family with no custom predicates
    (tmp_path / "checks.py").write_text("CHECKS = {'always': lambda ws: True}\n")
    assert set(module._checks()) == {"always"}
    (tmp_path / "checks.py").write_text("SOMETHING_ELSE = 1\n")
    try:
        module._checks()
    except SystemExit as e:
        assert "no CHECKS dict" in str(e)
    else:
        raise AssertionError("a checks.py without CHECKS must be refused, not silently ignored")


def test_a_missing_baseline_is_sent_as_missing_not_as_an_empty_one(tmp_path, monkeypatch):
    """The grader treats an absent `before.json` as "unknown" rather than "everything changed" — but only if
    it actually sees it absent. Substituting `{}` here defeated that and disqualified timed-out episodes."""
    import io
    import tarfile

    sent = {}

    def fake_run(args, *, input=None, timeout=None):
        class R:
            returncode = 0
            stdout = json.dumps({"published_pass": True, "protected_modified": None}).encode()
            stderr = b""

            def check_returncode(self):
                pass

        if input:
            sent["tar"] = input
        return R()

    monkeypatch.setattr(g, "_run", fake_run)
    (tmp_path / "snapshot.tar").write_bytes(b"")
    task = {
        "task_id": "t-1",
        "commands": [],
        "timeout_s": 60,
        "protected_paths": ["bin/server.py"],
        "published": {"predicates": []},
    }

    g.grade_in_container(tmp_path, task, None, "img")
    with tarfile.open(fileobj=io.BytesIO(sent["tar"])) as tf:
        assert "before.json" not in tf.getnames()

    (tmp_path / "before.json").write_text(json.dumps({"bin/server.py": "abc"}))
    g.grade_in_container(tmp_path, task, None, "img")
    with tarfile.open(fileobj=io.BytesIO(sent["tar"])) as tf:
        assert b"abc" in tf.extractfile("before.json").read()
