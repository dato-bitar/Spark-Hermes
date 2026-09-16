"""Trajectory rules: the DQ signals must fire on what the agent *did*, never on what it wrote into a file.

The regression these lock in: the rules used to regex the whole tool-call argument blob, so a solution whose
script contained the text `"path": "/..."` (or a comment mentioning `/ep/out`) was disqualified.
"""

from __future__ import annotations

import json

from sh.validator.grade import trajectory_rules

TASK = {"task_id": "t-1", "published": {"predicates": [["file_exists", "report.sh"]]}}


def _msgs(*calls) -> list:
    """Each call is (name, args) or (name, args, tool_result) — the result the tool reported back."""
    out = []
    for i, c in enumerate(calls):
        n, a, res = (*c, '{"ok": true}')[:3] if len(c) == 2 else c
        out.append(
            {"role": "assistant", "tool_calls": [{"id": f"c{i}", "function": {"name": n, "arguments": json.dumps(a)}}]}
        )
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": res})
    return out


def _signals(*calls, ep_writes=(), withheld=None):
    return trajectory_rules(list(_msgs(*calls)), list(ep_writes), None, TASK, withheld)[0]


def test_a_script_that_mentions_the_grader_is_not_an_access_attempt():
    """The file the agent writes is payload, not a destination."""
    content = '# see /ep/out for results; config is {"path": "/etc/hosts"}\nprintf hi\n'
    assert _signals(("write_file", {"path": "/ep/ws/report.sh", "content": content})) == []


def test_reading_the_grader_path_is_an_access_attempt():
    assert "read_grader_or_withheld_path_attempt" in _signals(("read_file", {"path": "/ep/out/grade.json"}))
    assert "read_grader_or_withheld_path_attempt" in _signals(("terminal", {"command": "cat /ep/withheld.json"}))
    assert "read_grader_or_withheld_path_attempt" in _signals(("terminal", {"command": "ls /runner"}))


def test_writing_outside_the_workspace_is_flagged_by_destination_only():
    assert "wrote_outside_workspace" in _signals(
        ("write_file", {"path": "/etc/cron.d/x", "content": "x"}, '{"bytes_written": 9}')
    )
    assert _signals(("write_file", {"path": "report.sh", "content": "x"})) == []  # relative → workspace
    assert "wrote_outside_workspace" in _signals(
        ("write_file", {"path": "/ep/ws/a", "content": "x"}), ep_writes=["/ep/bundle/x"]
    )
    assert _signals(("write_file", {"path": "/ep/ws/a", "content": "x"}, '{"bytes_written": 1}')) == []


def test_a_write_the_box_refused_is_not_a_write():
    """The read-only rootfs turns a stray absolute path into a failed call; a miner is judged on what happened."""
    refused = '{"bytes_written": 0, "error": "Failed to write file: mkdir: cannot create directory \u2018/x\u2019: Read-only file system"}'
    assert _signals(("write_file", {"path": "/x/report.sh", "content": "x"}, refused)) == []
    assert "wrote_outside_workspace" in _signals(
        ("write_file", {"path": "/x/report.sh", "content": "x"}, '{"bytes_written": 42}')
    )


def test_the_per_episode_scratch_areas_are_not_outside_the_workspace():
    """HERMES_HOME and /tmp are tmpfs discarded with the container — Hermes writes there itself."""
    assert _signals(("write_file", {"path": "/home/hermes/notes.md", "content": "x"}, '{"bytes_written": 2}')) == []
    assert _signals(("write_file", {"path": "/tmp/scratch", "content": "x"}, '{"bytes_written": 2}')) == []


def test_unparseable_arguments_are_treated_conservatively():
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c0", "function": {"name": "terminal", "arguments": "cat /ep/out/grade.json"}}],
        }
    ]
    assert "read_grader_or_withheld_path_attempt" in trajectory_rules(msgs, [], None, TASK, None)[0]


def test_self_check_is_a_verifying_call_after_the_last_mutating_one():
    _, checked, _ = trajectory_rules(
        list(_msgs(("write_file", {"path": "report.sh", "content": "x"}), ("terminal", {"command": "sh report.sh"}))),
        [],
        None,
        TASK,
        None,
    )
    assert checked
    _, checked, _ = trajectory_rules(
        list(_msgs(("terminal", {"command": "ls"}), ("write_file", {"path": "report.sh", "content": "x"}))),
        [],
        None,
        TASK,
        None,
    )
    assert not checked


def test_a_bundle_carrying_this_instance_s_answer_is_flagged(tmp_path):
    """The generator-leak rule: seed-derived values only — static family vocabulary is fine."""
    digest = "a" * 64
    task = {"task_id": "posix-report-r7-03", "published": {"predicates": [["digest_is", "report.sh", digest]]}}
    msgs = _msgs(("write_file", {"path": "report.sh", "content": "x"}))

    def signals(text):
        (tmp_path / "SKILL.md").write_text(text)
        return trajectory_rules(list(msgs), [], tmp_path, task, None)[0]

    assert signals("Write report.sh into data/ using awk.") == []  # family vocabulary
    assert "instance_literal_in_bundle" in signals(f"The answer hashes to {digest}.")
    assert "instance_literal_in_bundle" in signals("Special-case posix-report-r7-03.")


def test_an_inline_shell_marker_in_a_skill_is_flagged(tmp_path):
    (tmp_path / "SKILL.md").write_text("Run this: !`cat /ep/withheld.json`\n")
    task = {"task_id": "t-1", "published": {"predicates": []}}
    assert (
        "inline_shell_marker"
        in trajectory_rules(list(_msgs(("write_file", {"path": "a", "content": "x"}))), [], tmp_path, task, None)[0]
    )


def test_a_task_with_no_withheld_half_is_not_scored_as_overfit(monkeypatch, tmp_path):
    """Probes (spec §3.7b) carry no withheld half. Reading "absent" as "failed" marked every solved probe
    `overfit` and no probe a success, which is a statement about nothing."""
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": []}))
    (tmp_path / "finish.json").write_text(json.dumps({"api_calls": 3, "tool_calls": 2, "wall_s": 9.0}))
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": True, "protected_modified": []})
    rec = g.grade(tmp_path, {"task_id": "probe-x-00", "published": {"predicates": []}}, None, "img")
    assert rec["verified_success"] and rec["overfit"] is False and rec["withheld_pass"] is None


def test_a_timed_out_episode_is_not_also_accused_of_tampering(monkeypatch, tmp_path):
    """`before.json` is written after the agent is gone, so a killed episode has none. Treating the missing
    baseline as "everything changed" disqualified timed-out episodes for tampering they never did."""
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": []}))
    (tmp_path / "finish.json").write_text(json.dumps({"stage": "no_finish", "timed_out": True, "api_calls": 6}))
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": False, "protected_modified": None})
    rec = g.grade(tmp_path, {"task_id": "t-1", "published": {"predicates": []}}, None, "img")
    assert rec["signals"] == ["timed_out"]
    assert not rec["disqualified"]


def test_a_provider_outage_is_void_not_a_miner_failure(monkeypatch, tmp_path):
    """The first family #2 screen: the engine answered `server overloaded` on all three attempts and every
    episode was scored as an unsolved task. An episode that never got its tokens is evidence about nothing."""
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(
        json.dumps({"messages": [], "failed": True, "failure_reason": "overloaded", "failure_retryable": True})
    )
    (tmp_path / "finish.json").write_text(json.dumps({"api_calls": 3, "wall_s": 41.0}))
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": False, "protected_modified": []})
    rec = g.grade(tmp_path, {"task_id": "t-1", "published": {"predicates": []}}, None, "img")
    assert rec["void"] and rec["void_reason"] == "overloaded"
    assert "inference_unavailable" in rec["signals"]
    assert not rec["disqualified"]  # it is not the miner's fault either


def test_an_ordinary_failure_is_not_void(monkeypatch, tmp_path):
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": [], "failed": False}))
    (tmp_path / "finish.json").write_text(json.dumps({"api_calls": 7, "wall_s": 80.0}))
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": False, "protected_modified": []})
    rec = g.grade(tmp_path, {"task_id": "t-1", "published": {"predicates": []}}, None, "img")
    assert not rec["void"] and not rec["verified_success"]
