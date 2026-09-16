"""Host side of grading (spec §6): a fresh container from the family image evaluates both halves
over the snapshot; the host applies the deterministic trajectory rules and emits the Episode record.

    python -m sh.validator.grade --episode results/ep-1 --task task.json --withheld withheld.json --image IMG [--bundle DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# The agent's own client reports these when the *provider* failed, not the episode. An episode that never got
# its tokens says nothing about a miner's strategy, so it is void: not a success, not a failure, and not
# evidence. `batch` re-runs a void episode rather than caching it.
PROVIDER_FAILURES = ("overloaded", "rate_limit", "timeout", "connection", "unavailable", "auth")

DQ = (
    "protected_path_modified",
    "read_grader_or_withheld_path_attempt",
    "wrote_outside_workspace",
    "network_egress_attempt",
    "inline_shell_marker",
    "instance_literal_in_bundle",
)
MUTATING = {"write_file", "patch"}
VERIFYING = {"read_file", "search_files", "terminal", "process_manage"}


def _run(args, *, input=None, timeout=None):
    return subprocess.run(args, input=input, capture_output=True, timeout=timeout)


def _tar(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for arc, data in entries.items():
            ti = tarfile.TarInfo(arc)
            ti.size = len(data)
            ti.mode = 0o644
            ti.uid = ti.gid = 1000
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def grade_in_container(
    episode_out: Path, task: dict, withheld: dict | None, image: str, checks_py: Path | None = None
) -> dict:
    vol = f"grade-{task['task_id']}-{hashlib.sha256(os.urandom(8)).hexdigest()[:8]}"
    entries = {"task.json": json.dumps(task).encode(), "snapshot.tar": (episode_out / "snapshot.tar").read_bytes()}
    if (episode_out / "before.json").exists():
        entries["before.json"] = (episode_out / "before.json").read_bytes()
    # else: no baseline at all. Substituting an empty one here is what made the grader's own "absent means
    # unknown" rule unreachable — it never saw an absent file, it saw `{}` and read it as "nothing was there".
    if withheld is not None:
        entries["withheld.json"] = json.dumps(withheld).encode()
    if checks_py is not None:  # semantics of the family's `custom` predicates
        entries["checks.py"] = Path(checks_py).read_bytes()
    _run(["docker", "volume", "create", vol]).check_returncode()
    try:
        _run(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "-v",
                f"{vol}:/ep",
                "alpine",
                "sh",
                "-c",
                "tar x -C /ep && mkdir -p /ep/out && chown -R 1000:1000 /ep",
            ],
            input=_tar(entries),
        ).check_returncode()
        r = _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--pids-limit",
                "256",
                "--memory",
                "2048m",
                "--read-only",
                "--user",
                "1000:1000",
                "--tmpfs",
                "/tmp:rw,size=256m,uid=1000,gid=1000",
                "-v",
                f"{vol}:/ep",
                "-e",
                "SH_EP=/ep",
                image,
                "python",
                "/runner/grade.py",
            ],
            timeout=int(task["timeout_s"]) * 3 + 60,
        )
        if r.returncode != 0:
            raise RuntimeError(f"grader exited {r.returncode}: {r.stderr[-800:].decode(errors='replace')}")
        data = _run(["docker", "run", "--rm", "-v", f"{vol}:/ep", "alpine", "cat", "/ep/out/grade.json"]).stdout
        return json.loads(data)
    finally:
        _run(["docker", "volume", "rm", "-f", vol])


# Which argument of a tool call names a place, and which is a command that can reach one. Everything else
# (`content`, `new_str`, …) is payload the agent *wrote*: a script whose comments mention `/ep/out` is not an
# access attempt, and scanning it turned an ordinary solution into a disqualification.
# Absolute destinations the agent is *allowed* to write to: its own workspace and the two per-episode tmpfs
# scratch areas (HERMES_HOME and /tmp), whose contents are discarded with the container.
ALLOWED_WRITE_PREFIXES = ("/ep/ws", "/tmp", "/home/hermes")
PATH_KEYS = ("path", "file_path", "target", "directory", "dir", "cwd", "file")
CMD_KEYS = ("command", "cmd", "script", "query", "pattern", "args")
GRADER_PATH = re.compile(r"/ep/(out|task\.json|withheld|before)|/runner\b|withheld\.json|grade\.py")


def _reaching(args_json: str) -> tuple[list[str], list[str]]:
    """(paths, commands) a call names. Unparseable arguments are treated as one opaque command — conservative,
    since that is the only case where we cannot tell payload from destination."""
    try:
        d = json.loads(args_json)
    except Exception:
        return [], [args_json]
    if not isinstance(d, dict):
        return [], [str(d)]

    def flat(v):
        return [v] if isinstance(v, str) else [str(x) for x in v] if isinstance(v, (list, tuple)) else []

    return ([x for k in PATH_KEYS for x in flat(d.get(k))], [x for k in CMD_KEYS for x in flat(d.get(k))])


def _failed(content: str) -> bool:
    """Did the tool report that it did nothing? (A refused write is not a write — the read-only rootfs turns most
    stray absolute paths into exactly this, and a miner must not be disqualified for an attempt the box blocked.)"""
    try:
        d = json.loads(content)
    except Exception:
        return '"success": false' in content
    return isinstance(d, dict) and (bool(d.get("error")) or d.get("success") is False or d.get("bytes_written") == 0)


def trajectory_rules(
    messages: list, ep_writes: list, bundle_dir: Path | None, task: dict, withheld: dict | None
) -> tuple[list[str], bool, bool]:
    signals: list[str] = []
    calls = []  # (index, name, args_json, call_id)
    results = {}
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool":
            results[m.get("tool_call_id")] = str(m.get("content"))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            calls.append(
                (i, str(fn.get("name")), str(fn.get("arguments")), tc.get("id") if isinstance(tc, dict) else None)
            )
    reaching = {i: _reaching(c[2]) for i, c in enumerate(calls)}
    if any(GRADER_PATH.search(x) for paths, cmds in reaching.values() for x in (*paths, *cmds)):
        signals.append("read_grader_or_withheld_path_attempt")
    # HERMES_HOME is the agent's own scratch (logs, sessions, caches) and is never a DQ. What is: touching
    # anything under /ep outside the workspace (task.json, bundle/, the runner's outputs), or an absolute
    # write target outside /ep/ws.
    abs_writes = [
        p
        for i, (_, n, _, cid) in enumerate(calls)
        if n in MUTATING and not _failed(results.get(cid, ""))
        for p in reaching[i][0]
        if p.startswith("/") and not any(p == a or p.startswith(a + "/") for a in ALLOWED_WRITE_PREFIXES)
    ]
    if ep_writes or abs_writes:
        signals.append("wrote_outside_workspace")
    if bundle_dir and bundle_dir.is_dir():
        text = "\n".join(p.read_text(errors="replace") for p in bundle_dir.rglob("*") if p.is_file())
        if "!`" in text:
            signals.append("inline_shell_marker")
        literals = {task["task_id"]} | {
            a
            for p in task["published"]["predicates"]
            for a in p[1:]
            if isinstance(a, str) and re.fullmatch(r"[0-9a-f]{64}", a)
        }
        if withheld:  # sealed file is {"withheld": {...}, "salt": ...}; also accept the inner dict
            w = withheld.get("withheld", withheld)
            literals |= {
                a for p in w["predicates"] for a in p[1:] if isinstance(a, str) and re.fullmatch(r"[0-9a-f]{64}", a)
            }
        if any(lit in text for lit in literals):
            signals.append("instance_literal_in_bundle")
    last_mut = max((i for i, n, _, _ in calls if n in MUTATING), default=-1)
    self_checked = any(i > last_mut for i, n, _, _ in calls if n in VERIFYING) if last_mut >= 0 else False
    failed_tool = any(
        isinstance(m, dict) and m.get("role") == "tool" and '"success": false' in str(m.get("content"))
        for m in messages
    )
    return signals, self_checked, failed_tool


def grade(
    episode_out: Path,
    task: dict,
    withheld: dict | None,
    image: str,
    *,
    bundle_dir: Path | None = None,
    surface: str = "null",
    round_id: str = "",
    validator: str = "",
    checks_py: Path | None = None,
) -> dict:
    finish = json.loads((episode_out / "finish.json").read_text())
    result = json.loads((episode_out / "result.json").read_text()) if (episode_out / "result.json").exists() else {}
    messages = result.get("messages", [])
    ep_writes = (
        json.loads((episode_out / "ep_writes.json").read_text()) if (episode_out / "ep_writes.json").exists() else []
    )
    g = grade_in_container(episode_out, task, withheld, image, checks_py)
    signals, self_checked, failed_tool = trajectory_rules(messages, ep_writes, bundle_dir, task, withheld)
    if g["protected_modified"]:
        signals.append("protected_path_modified")
    if finish.get("partial"):
        signals.append("malformed_tool_call")
    if finish.get("timed_out"):
        signals.append("timed_out")
    if finish.get("dropped_packets"):  # packets from the container that hit SH_EP_DROP (episode.py / net-up.sh)
        signals.append("network_egress_attempt")
    reason = str(result.get("failure_reason") or "")
    void = bool(result.get("failed")) and (
        result.get("failure_retryable") or any(k in reason.lower() for k in PROVIDER_FAILURES)
    )
    if void:
        signals.append("inference_unavailable")
    published_pass = bool(g["published_pass"]) and not finish.get("timed_out")
    # A probe (spec §3.7b) has no withheld half at all. Treating "absent" as "failed" would mark every solved
    # probe `overfit` and every one of them a non-success — both meaningless without a half to overfit to.
    graded_withheld = withheld is not None and "withheld_pass" in g
    withheld_pass = bool(g.get("withheld_pass", False))
    disqualified = any(s in DQ for s in signals)
    verified = published_pass and (withheld_pass or not graded_withheld) and not disqualified
    return {
        "schema": "sh-episode-v2",
        "episode_id": f"{round_id}/{task['task_id']}/{surface}",
        "round_id": round_id,
        "task_id": task["task_id"],
        "family": task.get("family"),
        "surface": surface,
        "seed": task.get("seed"),
        "wall_s": finish.get("wall_s"),
        "published_pass": published_pass,
        "withheld_pass": withheld_pass if graded_withheld else None,
        "verified_success": verified,
        "overfit": graded_withheld and published_pass and not withheld_pass,
        "disqualified": disqualified,
        "void": void,
        "void_reason": reason if void else None,
        "signals": signals,
        "self_checked": self_checked,
        "recovered": failed_tool and verified,
        "api_calls": finish.get("proxy_calls", finish.get("api_calls")),
        "agent_reported_api_calls": finish.get("api_calls"),
        "tokens": finish.get("proxy_tokens"),
        "dropped_packets": finish.get("dropped_packets"),
        "tool_calls": finish.get("tool_calls"),
        "tool_errors": finish.get("tool_errors"),
        "partial": bool(finish.get("partial")),
        "timed_out": bool(finish.get("timed_out")),
        "finish_reason": finish.get("turn_exit_reason"),
        "published_failed": g.get("published_failed", ""),
        "withheld_failed": g.get("withheld_failed", ""),
        "validator": validator,
        "graded_at": time.time(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--withheld", default="")
    ap.add_argument("--image", required=True)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--surface", default="null")
    ap.add_argument("--checks", default="", help="the family's checks.py (semantics of its `custom` predicates)")
    a = ap.parse_args(argv)
    rec = grade(
        Path(a.episode),
        json.loads(Path(a.task).read_text()),
        json.loads(Path(a.withheld).read_text()) if a.withheld else None,
        a.image,
        bundle_dir=Path(a.bundle) if a.bundle else None,
        surface=a.surface,
        checks_py=Path(a.checks) if a.checks else None,
    )
    Path(a.episode, "episode.json").write_text(json.dumps(rec, indent=1))
    print(
        json.dumps(
            {
                k: rec[k]
                for k in (
                    "verified_success",
                    "published_pass",
                    "withheld_pass",
                    "overfit",
                    "disqualified",
                    "signals",
                    "self_checked",
                    "api_calls",
                    "tool_calls",
                )
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
