"""Runs INSIDE a fresh grading container (same family image, no network, no bundle, uid 1000).
Reads /ep/{task.json, withheld.json, before.json, snapshot.tar}; writes /ep/out/grade.json.
The agent is gone; this container never shares a boundary with it (spec §6)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, "/runner")
import predicates  # noqa: E402

EP = Path(os.environ.get("SH_EP", "/ep"))
WS = EP / "ws"
OUT = EP / "out"


def _checks() -> dict:
    """The semantics of any `custom` predicate the family uses (spec §6): `checks.py` is placed in /ep by the
    host, outside the workspace, and is read-only to this container. Absent for families that use none.

    A check is a callable taking the workspace root and returning a bool. It runs here rather than in the
    boundary, so the agent never sees it; it is published at close(r) along with the withheld half."""
    path = EP / "checks.py"
    if not path.exists():
        return {}
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)  # noqa: S102 — validator-authored, never the agent's
    checks = ns.get("CHECKS")
    if not isinstance(checks, dict):
        raise SystemExit("checks.py defines no CHECKS dict")
    return checks


def main() -> int:
    task = json.loads((EP / "task.json").read_text())
    withheld = json.loads((EP / "withheld.json").read_text())["withheld"] if (EP / "withheld.json").exists() else None
    WS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(EP / "snapshot.tar") as tf:
        tf.extractall(WS, filter="data")  # agent-authored content: refuse links/abs paths
    kw = dict(commands=task["commands"], checks=_checks(), timeout_s=int(task["timeout_s"]))
    pub = predicates.judge(task["published"]["predicates"], WS, **kw)
    g = {
        "published_pass": pub.ok,
        "published_failed": pub.message,
        "protected_after": {
            rel: (hashlib.sha256((WS / rel).read_bytes()).hexdigest() if (WS / rel).is_file() else None)
            for rel in task["protected_paths"]
        },
    }
    if withheld is not None:
        whd = predicates.judge(withheld["predicates"], WS, **kw)
        g.update(withheld_pass=whd.ok, withheld_failed=whd.message)
    # `before.json` is written by the runner after the agent is gone, so it is *absent* when the episode was
    # killed on its timeout. An absent baseline is unknown, not "everything changed": comparing against {} made
    # every timed-out episode also a `protected_path_modified` disqualification for tampering it never did.
    # Nothing is lost by declining to judge it — a timed-out episode cannot be a verified success anyway.
    before_path = EP / "before.json"
    before = json.loads(before_path.read_text()) if before_path.exists() else None
    g["protected_modified"] = (
        None if before is None else [rel for rel, h in g["protected_after"].items() if before.get(rel) != h]
    )
    (OUT / "grade.json").write_text(json.dumps(g, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
