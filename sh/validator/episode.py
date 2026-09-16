"""Host side of one episode (spec §5.2, adapted): a per-episode NAMED VOLUME carries inputs in and
results out, because host-path bind mounts are unavailable on the validator host. The volume is the
only writable path shared with the host; the rootfs is read-only; HERMES_HOME and /tmp are tmpfs.

    python -m sh.validator.episode --task task.json --bundle DIR|none --image family-posix-report:pin \
        --inference http://127.0.0.1:8080/v1 --out results/ep-1 [--network host]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from sh.validator.proxy import Tokens


def _run(args: list[str], *, input: bytes | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, input=input, capture_output=True, timeout=timeout)


def _tar_bytes(entries: dict[str, bytes | Path]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for arc, src in entries.items():
            if isinstance(src, Path):
                tf.add(src, arcname=arc)
            else:
                ti = tarfile.TarInfo(arc)
                ti.size = len(src)
                ti.mode = 0o644
                ti.uid = ti.gid = 1000
                tf.addfile(ti, io.BytesIO(src))
    return buf.getvalue()


SH_EP_GW = "172.30.0.1"  # gateway of the sh-ep network (net/net-up.sh); the proxy listens here
SH_EP_PROXY_PORT = 8090
DROP_CHAIN = "SH_EP_DROP"  # per-episode accounting rules live here (net-up.sh creates the chain)


def _iptables(*args: str) -> subprocess.CompletedProcess | None:
    if shutil.which("iptables") is None:
        return None
    return _run(["iptables", *args])


def _container_ip(ep: str) -> str:
    r = _run(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", ep])
    return r.stdout.decode().strip()


def _drops_for(ip: str) -> int | None:
    """Packets from `ip` that reached the drop chain — the firewall signal the grader reads (spec §12.1).
    The accounting rule has no target, so the `target` column is absent: match the line from the right."""
    r = _iptables("-L", DROP_CHAIN, "-v", "-n", "-x")
    if r is None or r.returncode:
        return None
    for line in r.stdout.decode().splitlines():
        m = re.match(r"^\s*(\d+)\s+\d+\s+.*?\s(\S+)\s+0\.0\.0\.0/0\s*$", line)
        if m and m.group(2) == ip:
            return int(m.group(1))
    return None


def run_episode(
    task: dict,
    bundle_dir: Path | None,
    image: str,
    inference: str,
    out: Path,
    *,
    token: str = "none",
    network: str = "host",
    extra_add_hosts: list[str] = (),
    timeout_s: int | None = None,
    mem_mb: int = 4096,
    cpus: float = 2.0,
    disk_mb: int = 1024,
    tokens_path: Path | None = None,
    usage_dir: Path | None = None,
) -> dict:
    """`network="sh-ep"`: the boundary of spec §12 — the container sees only the proxy (single-use token issued here
    and revoked in `finally`), names never resolve, every other packet is counted against its IP. `network="host"` is
    the spike/dev mode with no boundary."""
    out.mkdir(parents=True, exist_ok=True)
    ep = f"ep-{task['task_id']}-{hashlib.sha256(os.urandom(8)).hexdigest()[:8]}"
    timeout_s = timeout_s or int(task["timeout_s"])
    tokens = Tokens(tokens_path) if tokens_path else None
    if tokens:
        token = tokens.issue(ep, ttl=timeout_s + 300)
    if network == "sh-ep":
        inference = f"http://inference:{SH_EP_PROXY_PORT}/v1"
        extra_add_hosts = [f"inference:{SH_EP_GW}", *extra_add_hosts]
    ip: str = ""
    entries: dict[str, bytes | Path] = {
        "task.json": json.dumps(task).encode(),
        "seed": str(task.get("seed", "")).encode(),
    }
    if bundle_dir and bundle_dir.is_dir():
        entries["bundle"] = bundle_dir
    else:
        entries["bundle/.keep"] = b""
    t0 = time.time()
    _run(["docker", "volume", "create", ep]).check_returncode()
    try:
        # populate the volume (helper container; chown so uid 1000 can write /ep/out)
        _run(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "-v",
                f"{ep}:/ep",
                "alpine",
                "sh",
                "-c",
                "tar x -C /ep && rm -f /ep/bundle/.keep && mkdir -p /ep/bundle /ep/out && chown -R 1000:1000 /ep",
            ],
            input=_tar_bytes(entries),
        ).check_returncode()
        net = (
            ["--network", network, "--dns", SH_EP_GW] if network != "host" else ["--network", "host"]
        )  # no resolver at the gateway → names never resolve
        for h in extra_add_hosts:
            net += ["--add-host", h]
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            ep,
            *net,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "512",
            "--memory",
            f"{mem_mb}m",
            "--cpus",
            str(cpus),
            "--read-only",
            "--user",
            "1000:1000",
            "--tmpfs",
            "/home/hermes:rw,size=256m,uid=1000,gid=1000",
            "--tmpfs",
            f"/tmp:rw,size={disk_mb}m,uid=1000,gid=1000",
            "-v",
            f"{ep}:/ep",
            "-e",
            "HOME=/home/hermes",
            "-e",
            "HERMES_HOME=/home/hermes",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            f"SH_TOKEN={token}",
            "-e",
            f"SH_INFERENCE={inference}",
            "-e",
            "SH_EP=/ep",
            image,
            "python",
            "/runner/run_episode.py",
        ]
        _run(cmd).check_returncode()
        if network != "host":
            ip = _container_ip(ep)
            if ip:
                _iptables("-I", DROP_CHAIN, "1", "-s", ip)  # count-only rule; the chain's final rule drops
        timed_out = False
        try:
            _run(["docker", "wait", ep], timeout=timeout_s + 30)
        except subprocess.TimeoutExpired:
            timed_out = True
            _run(
                ["docker", "exec", ep, "sh", "-c", "tar cf /ep/out/snapshot.tar -C /ep/ws . 2>/dev/null; true"],
                timeout=60,
            )
            _run(["docker", "kill", ep])
        logs = _run(["docker", "logs", ep]).stdout[-4000:] + _run(["docker", "logs", ep]).stderr[-4000:]
        (out / "container.log").write_bytes(logs)
        # extract results
        data = _run(["docker", "run", "--rm", "-v", f"{ep}:/ep", "alpine", "tar", "c", "-C", "/ep/out", "."]).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(out)
    finally:
        _run(["docker", "rm", "-f", ep])
        _run(["docker", "volume", "rm", "-f", ep])
        if tokens:
            tokens.revoke(token)
        drops = _drops_for(ip) if ip else None
        if ip:
            _iptables("-D", DROP_CHAIN, "-s", ip)
        (out / "net.json").write_text(json.dumps({"network": network, "ip": ip or None, "dropped_packets": drops}))
        if usage_dir and (usage_dir / f"{ep}.jsonl").exists():  # the proxy's per-call usage, independent of the agent
            shutil.move(usage_dir / f"{ep}.jsonl", out / "proxy_usage.jsonl")
    finish = json.loads((out / "finish.json").read_text()) if (out / "finish.json").exists() else {"stage": "no_finish"}
    finish["timed_out"] = timed_out
    finish["host_wall_s"] = round(time.time() - t0, 1)
    finish["episode"] = ep
    finish["dropped_packets"] = drops
    if (out / "proxy_usage.jsonl").exists():
        rows = [json.loads(line) for line in (out / "proxy_usage.jsonl").read_text().splitlines() if line.strip()]
        finish["proxy_calls"] = len(rows)
        finish["proxy_tokens"] = {
            k: sum(int(r["usage"].get(k) or 0) for r in rows) for k in ("prompt_tokens", "completion_tokens")
        }
    (out / "finish.json").write_text(json.dumps(finish, indent=1))
    return finish


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--bundle", default="none")
    ap.add_argument("--image", required=True)
    ap.add_argument("--inference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--network", default="host")
    ap.add_argument("--token", default="none")
    ap.add_argument("--tokens", default=None, help="proxy token file (issue a single-use token for this episode)")
    ap.add_argument("--usage-dir", default=None, help="proxy usage dir (per-call usage is moved into --out)")
    a = ap.parse_args(argv)
    task = json.loads(Path(a.task).read_text())
    f = run_episode(
        task,
        None if a.bundle == "none" else Path(a.bundle),
        a.image,
        a.inference,
        Path(a.out),
        token=a.token,
        network=a.network,
        tokens_path=Path(a.tokens) if a.tokens else None,
        usage_dir=Path(a.usage_dir) if a.usage_dir else None,
    )
    print(
        json.dumps(
            {
                k: f.get(k)
                for k in (
                    "stage",
                    "api_calls",
                    "proxy_calls",
                    "proxy_tokens",
                    "tool_calls",
                    "completed",
                    "partial",
                    "wall_s",
                    "host_wall_s",
                    "timed_out",
                    "dropped_packets",
                    "error",
                )
            },
            indent=1,
        )
    )
    return 0 if f.get("stage") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
