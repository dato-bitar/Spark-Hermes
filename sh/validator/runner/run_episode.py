"""Runs INSIDE the episode boundary as uid 1000. Everything it needs is under /ep (the per-episode
volume): task.json (public projection + bundle_sha256), bundle/ (the surface; empty for NULL),
seed. Everything it produces goes to /ep/out. It never sees the withheld check.

Steps (spec §5.3): copy the surface into HERMES_HOME and verify its digest; write config.yaml and
.env; materialise the fixture and run the family's setup_commands; digest protected paths; run
unmodified Hermes; write the result, the ShareGPT trajectory, home_writes.json and a snapshot tar.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

EP = Path(os.environ.get("SH_EP", "/ep"))
OUT = EP / "out"
WS = EP / "ws"
HOME = Path(os.environ["HERMES_HOME"])
MODEL = os.environ.get("SH_MODEL", "custom/qwen38-nvfp4")
INFERENCE = os.environ["SH_INFERENCE"]
TOKEN = os.environ.get("SH_TOKEN", "none")
SKILLS_TOOLSET = "skills"  # required for the skills index to render (V3); skill_manage neutralised by write_approval


def bundle_digest(root: Path) -> str:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    lines = [f"{p.relative_to(root).as_posix()}\0{hashlib.sha256(p.read_bytes()).hexdigest()}" for p in files]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def digests(root: Path, paths) -> dict:
    return {
        rel: (hashlib.sha256((root / rel).read_bytes()).hexdigest() if (root / rel).is_file() else None)
        for rel in paths
    }


def sh(cmd: str, cwd: Path, timeout: int) -> tuple[int, str]:
    env = {
        "PATH": f"{cwd / 'bin'}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(HOME),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        d = subprocess.run(
            ["/bin/sh", "-c", cmd],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return d.returncode, (d.stdout + d.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    finish = {"started": time.time(), "stage": "init"}
    before, ep_before, home_before = {}, {}, {}
    try:
        task = json.loads((EP / "task.json").read_text())
        # /ep/seed is written by the host and deliberately not applied here: sampling, the seed included, is
        # pinned by the proxy so that no bundle and no runner change can alter it (spec §5.6). Wiring the
        # per-episode seed through the proxy is still open — it pins one sampling set for all episodes today.

        # 1. surface → HERMES_HOME (writable copy), digest-verified
        HOME.mkdir(parents=True, exist_ok=True)
        bundle = EP / "bundle"
        if bundle.is_dir():
            shutil.copytree(bundle, HOME, dirs_exist_ok=True)
        got = bundle_digest(bundle) if bundle.is_dir() else bundle_digest(Path(tempfile_empty()))
        if task.get("bundle_sha256") and got != task["bundle_sha256"]:
            raise RuntimeError(f"bundle digest mismatch: {got} != {task['bundle_sha256']}")
        finish["bundle_sha256"] = got

        # 2. config + env
        (HOME / "config.yaml").write_text(
            "model:\n  default: %s\nproviders:\n  custom:\n    base_url: %s\n    api_key_env: SH_TOKEN\n"
            "agent:\n  max_turns: %d\nterminal:\n  backend: local\n  cwd: %s\n"
            "skills:\n  inline_shell: false\n  template_vars: false\n  auto_load: []\n  write_approval: true\n  guard_agent_created: true\n"
            % (MODEL, INFERENCE, int(task["max_turns"]), WS)
        )
        (HOME / ".env").write_text(f"SH_TOKEN={TOKEN}\n")

        # 3. fixture, setup commands, pre-agent digests
        WS.mkdir(parents=True, exist_ok=True)
        for rel, f in task["fixture"].items():
            p = WS / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(base64.b64decode(f["b64"]))
            os.chmod(p, int(f.get("mode", "0644"), 8))
        for cmd in task.get("setup_commands", []):
            code, out = sh(cmd, WS, int(task["timeout_s"]))
            if code != 0:
                finish.update(stage="setup_failed", setup_error=out)
                raise SystemExit(0)
        # Held in MEMORY until the end: the agent runs as the same uid and could otherwise edit /ep/out/before.json.
        before = digests(WS, task["protected_paths"])
        ep_before = {
            str(p): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in EP.rglob("*")
            if p.is_file() and WS not in p.parents and p != WS
        }
        home_before = {str(p): p.stat().st_mtime for p in HOME.rglob("*") if p.is_file()}
        os.chdir(WS)

        # 4–6. unmodified Hermes
        finish["stage"] = "agent"
        sys.path.insert(0, "/opt/hermes")
        from run_agent import AIAgent

        agent = AIAgent(
            model=MODEL,
            base_url=INFERENCE,
            api_key=TOKEN,
            max_iterations=int(task["max_turns"]),
            enabled_toolsets=[*task["tools"], SKILLS_TOOLSET],
            skip_memory=True,
            skip_context_files=False,
            load_soul_identity=True,
            quiet_mode=True,
            platform="batch",
            max_tokens=8192,
        )
        sp = agent._build_system_prompt()
        (OUT / "system_prompt.txt").write_text(sp)
        t0 = time.time()
        result = agent.run_conversation(user_message=task["prompt"], task_id=task["task_id"])
        finish["agent_wall_s"] = round(time.time() - t0, 1)
        messages = result.get("messages", [])
        json.dump(result, open(OUT / "result.json", "w"), default=str)
        try:
            traj = agent._convert_to_trajectory_format(messages, task["prompt"], bool(result.get("completed", True)))
            json.dump(traj, open(OUT / "trajectory.json", "w"), default=str)
        except Exception as e:
            finish["trajectory_error"] = repr(e)[:300]
        finish.update(
            stage="done",
            api_calls=int(
                result.get("api_calls")
                or sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
            ),
            tool_calls=sum(
                len(m.get("tool_calls") or []) for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
            ),
            tool_errors=sum(
                1
                for m in messages
                if isinstance(m, dict) and m.get("role") == "tool" and '"success": false' in str(m.get("content"))
            ),
            completed=bool(result.get("completed")),
            partial=bool(result.get("partial")),
            failed=bool(result.get("failed")),
            turn_exit_reason=str(result.get("turn_exit_reason")),
            final_head=str(result.get("final_response", ""))[:300],
        )
    except SystemExit:
        pass
    except Exception:
        finish["stage"] = finish.get("stage", "?") + "_error"
        finish["error"] = traceback.format_exc()[-2000:]
    finally:
        # 7. what the agent wrote outside the workspace, and the snapshot
        try:
            (OUT / "before.json").write_text(json.dumps(before))  # from memory, after the agent is gone
            writes = [
                str(p)
                for p in HOME.rglob("*")
                if p.is_file() and (str(p) not in home_before or p.stat().st_mtime != home_before[str(p)])
            ]
            (OUT / "home_writes.json").write_text(
                json.dumps(writes)
            )  # informational: HERMES_HOME is the agent's scratch
            ep_now = {
                str(p): (p.stat().st_mtime_ns, p.stat().st_size)
                for p in EP.rglob("*")
                if p.is_file() and WS not in p.parents and p != WS and OUT not in p.parents
            }
            ep_writes = sorted(
                set(ep_now) - set(ep_before) | {k for k in ep_now if k in ep_before and ep_now[k] != ep_before[k]}
            )
            (OUT / "ep_writes.json").write_text(
                json.dumps(ep_writes)
            )  # anything under /ep outside ws touched by the agent → DQ
        except Exception:
            pass
        if WS.exists():
            with tarfile.open(OUT / "snapshot.tar", "w") as tf:
                tf.add(WS, arcname=".")
        finish["ended"] = time.time()
        finish["wall_s"] = round(finish["ended"] - finish["started"], 1)
        (OUT / "finish.json").write_text(json.dumps(finish, indent=1))
    return 0


def tempfile_empty() -> str:
    import tempfile

    return tempfile.mkdtemp()


if __name__ == "__main__":
    sys.exit(main())
