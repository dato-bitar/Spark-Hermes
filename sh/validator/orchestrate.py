"""The round loop (spec §9.1) — every stage, unattended, forever.

    open -> window (2 h) -> seal -> evaluate -> close -> crown -> announce -> export -> publish -> next, at once

Tasks are minted ahead by the private queue daemon (`supply.queue`), so a round opens the instant the previous
one closes. Opening publishes the round's tasks and its **submission window**; miners fetch the tasks with the
CLI and submit one signed pull request per hotkey, replacing it as often as they like until the window closes.
The seal takes every strategy PR at its head SHA at that instant, verifies the hotkey's signature over this
round and this digest, evaluates, and pays on the pooled window while crowning on this round alone: the
crowned PR is merged as the incumbent, every other competition PR is closed with the reason, and a PR that
arrives after the seal is closed as outside the window.

Runs wherever the validator's credentials live; the GPU worker is reached over ssh and holds none. Everything
a miner is judged by is published under `rounds/<id>/` and mirrored into `docs/live/live.json` for the board.

Transparency rules the loop enforces, each because its absence would let a validator cheat quietly:

  * the window's open and close times are published with the tasks, before any submission exists;
  * a bundle is sealed by PR head SHA and digest before any evaluation, and the seal is published;
  * the withheld half is committed to in the published task and revealed at close with its salt;
  * future rounds exist only as digests (`rounds/queue.json`) until they open — verifiable, unreadable;
  * the crown is recomputable from the published episodes; the weights from the published close.json.

    python -m sh.validator.orchestrate --queue ../Spark-Hermes-Withheld/queue
    python -m sh.validator.orchestrate --queue ... --mock-miners DIR --mock-keys DIR   # test challengers
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sh.cli.lint import check_files, collect
from sh.cli.scorecard import render as render_scorecard
from sh.exports.build import build as build_exports
from sh.exports.upload import upload as upload_exports
from sh.scoring.crown import crown as crown_rule
from sh.validator.round import close as close_round
from sh.validator.stats import load_episodes
from sh.web.build import render as render_leaderboard

REPO = "gittensor-model-hub/Spark-Hermes"
BRANCH = "main"
LABEL_STRATEGY, LABEL_SCORED, LABEL_CROWN = "sh:strategy", "sh:round:scored", "sh:round:crown"
HF_REPO = "gittensor-model-hub/spark-hermes-rounds"
LIVE = "docs/live/live.json"  # what the dashboard polls; committed on every stage change
STAGES = ("open", "window", "seal", "evaluate", "close", "crown", "announce", "export", "publish_close", "done")


@dataclass
class Config:
    """The control plane runs wherever the validator's credentials live; the GPU worker is reached over ssh and
    holds no credentials at all — it receives a round directory, runs episodes, and hands the episodes back."""

    state: Path  # durable validator state: rounds/, archive/, salt secret
    repo: Path  # a checkout of BRANCH the loop commits to
    queue: Path  # the private queue daemon's directory of minted rounds
    pkg: Path  # the public package's parent
    worker: str = "root@91.224.44.223"
    worker_port: int = 50199
    worker_root: str = "/root/sh"  # holds pkg/ (the public package), state/tokens, state/usage
    image: str = "hermes-ubuntu:pin"  # the fallback; an image-defined task names its own
    window: int = 8  # rounds pooled for payment
    window_s: int = 2 * 3600  # the submission window
    min_paired: int = 4  # instances a strategy must share with the baseline to be crowned
    concurrency: int = 2
    era: str = "e0"

    @property
    def rounds(self) -> Path:
        return self.state / "rounds"


def sh(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:4])}… exited {r.returncode}: {r.stderr[-800:]}")
    return r.stdout


def gh(*args: str) -> str:
    return sh(["gh", *args])


def _label(number: int, name: str) -> None:
    """Add a label to a PR, creating the label if the repository has never seen it; idempotent."""
    sh(["gh", "label", "create", name, "--repo", REPO, "--color", "5319e7", "--force"], check=False)
    sh(["gh", "api", "-X", "POST", f"repos/{REPO}/issues/{number}/labels", "-f", f"labels[]={name}"], check=False)


def _worker(cfg: Config, cmd: str) -> str:
    return sh(
        ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-p", str(cfg.worker_port), cfg.worker, cmd],
        check=False,
    )


def _worker_launch(cfg: Config, cmd: str) -> None:
    """Start a long-running command on the worker and come back at once. `-f` backgrounds ssh after
    authentication and `-n` detaches stdin; the polling loop, not this call, decides whether the job runs."""
    try:
        subprocess.run(
            ["ssh", "-f", "-n", "-o", "BatchMode=yes", "-p", str(cfg.worker_port), cfg.worker, cmd],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pass


def _rsync(src: str, dst: str, cfg: Config) -> None:
    sh(["rsync", "-az", "--delete", "-e", f"ssh -o BatchMode=yes -p {cfg.worker_port}", src, dst])


# ─── round bookkeeping ─────────────────────────────────────────────────────────────────────────────
def log(rd: Path, stage: str, **fields) -> None:
    rec = {"t": time.time(), "stage": stage, **fields}
    with (rd / "phases.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{rd.name}] {stage} {json.dumps(fields)[:200]}", flush=True)


def _commit(cfg: Config, message: str, paths: tuple[str, ...] = ("rounds", "docs/live", "docs/rounds")) -> None:
    paths = tuple(p for p in paths if (cfg.repo / p).exists())  # a fresh checkout has no docs/rounds yet
    if not paths:
        return
    sh(["git", "add", *paths], cwd=cfg.repo)
    if sh(["git", "status", "--porcelain", *paths], cwd=cfg.repo).strip():
        sh(["git", "commit", "-q", "-m", message, "--", *paths], cwd=cfg.repo)  # only these paths
        sh(["git", "pull", "-q", "--rebase", "origin", BRANCH], cwd=cfg.repo, check=False)
        sh(["git", "push", "-q", "origin", BRANCH], cwd=cfg.repo)


def _read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def _trim_scores(closed: dict) -> dict:
    """Per-hotkey score, weight and the paired deltas from a close record — what the board and history show."""
    weights = closed.get("weights", {})
    return {
        h: {
            "score": s.get("score"),
            "weight": weights.get(h, 0.0),
            "mean_d": s.get("mean_d"),
            "delta_c": s.get("delta_c"),
            "n": s.get("n"),
        }
        for h, s in closed.get("scores", {}).items()
    }


def live(
    cfg: Config,
    rd: Path,
    stage: str,
    *,
    progress: dict | None = None,
    submissions: list[dict] | None = None,
    push: bool = True,
) -> None:
    """The dashboard's single source: the round, its window, its stage and progress, the crown and scores once
    it closes, the queue of rounds waiting unread, and the history of closed rounds."""
    phases = (
        [json.loads(line) for line in (rd / "phases.jsonl").read_text().splitlines()]
        if (rd / "phases.jsonl").exists()
        else []
    )
    sealed = _read(rd / "seal.json", {})
    history = _read(cfg.repo / "rounds" / "index.json", {"rounds": []})["rounds"]
    closed = _read(rd / "close" / "close.json")
    crowned = _read(rd / "close" / "crown.json")
    prev = _read(cfg.repo / LIVE, {})
    same = prev.get("round_id") == rd.name
    if progress is None:  # stages after evaluation carry the counts forward rather than blanking the board
        progress = prev.get("progress") if same else None
    if submissions is None and same:
        submissions = prev.get("submissions")
    last = history[-1] if history else None
    scores = None
    if closed:
        weights = closed.get("weights", {})
        scores = {
            "weights": weights,
            "king": crowned.get("king") if crowned else None,
            "per_hotkey": {
                h: {
                    k: s.get(k)
                    for k in ("n", "score", "mean_d", "se", "delta_c", "gate", "overfit_rate", "dq", "reason")
                }
                for h, s in closed.get("scores", {}).items()
            },
            "family_stats": {
                f: {"null_p": r["null"].get("p"), "canon_p": r["canon"].get("p"), "label": r.get("label")}
                for f, r in closed.get("family_stats", {}).items()
            },
            "commitments_ok": closed.get("commitments_ok"),
        }
    state = {
        "schema": "sh-live-v3",
        "updated": time.time(),
        "round_id": rd.name,
        "stage": stage,
        "stages": list(STAGES),
        "started": phases[0]["t"] if phases else time.time(),
        "phases": [{"stage": p["stage"], "t": p["t"]} for p in phases],
        "window": _read(rd / "window.json"),
        "submissions": submissions or [],
        "standings": (last or {}).get("scores") or {},  # the pooled standing after the last close: what pays now
        "last_round": (last or {}).get("round_id"),
        "tasks": len(list((rd / "tasks").glob("*.json"))) if (rd / "tasks").exists() else 0,
        "active": sealed.get("active", {}),
        "rejected": sealed.get("rejected", {}),
        "progress": progress or {},
        "crown": crowned,
        "scores": scores,
        "queue": _read(cfg.repo / "rounds" / "queue.json", {}).get("ready", []),
        "history": history[-20:],
        "repo": REPO,
        "branch": BRANCH,
        "hf_repo": HF_REPO,
    }
    out = cfg.repo / LIVE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=1))
    if push:
        _commit(cfg, f"{rd.name}: live — {stage}", paths=("docs/live",))


# ─── stages ────────────────────────────────────────────────────────────────────────────────────────
def queue_ready(cfg: Config) -> list[str]:
    if not cfg.queue.exists():
        return []
    return sorted(p.name for p in cfg.queue.iterdir() if p.is_dir() and (p / "READY").exists())


def open_round(cfg: Config) -> Path:
    """Take the lowest ready round from the queue and give it a window. Waits — visibly, on the board — while
    the queue is empty. Publishing is a separate, resumable step."""
    waited = 0
    while not queue_ready(cfg):
        if waited % 300 == 0:  # the board shows the wait rather than going stale on the last round's "done"
            print("[loop] queue empty; waiting for the daemon", flush=True)
            previous = sorted(p for p in cfg.rounds.glob("r*") if p.is_dir()) if cfg.rounds.exists() else []
            if previous:
                live(cfg, previous[-1], "waiting")
        time.sleep(60)
        waited += 60
    round_id = queue_ready(cfg)[0]
    rd = cfg.rounds / round_id
    cfg.rounds.mkdir(parents=True, exist_ok=True)
    shutil.move(str(cfg.queue / round_id), str(rd))  # consumed: the daemon refills
    (rd / "READY").unlink(missing_ok=True)
    for tag in _image_tags(rd):  # derivation is done; the images only cost disk here now
        subprocess.run(["docker", "image", "rm", "-f", tag], capture_output=True)
    if _image_tags(rd):
        subprocess.run(["docker", "builder", "prune", "-f", "--filter", "until=48h"], capture_output=True)
    now = time.time()
    (rd / "window.json").write_text(
        json.dumps({"opens_at": now, "closes_at": now + cfg.window_s, "seconds": cfg.window_s})
    )
    log(rd, "start")
    return rd


def publish_round(cfg: Config, rd: Path) -> None:
    """The tasks, the window, the queue digests, and each task image's Dockerfile and provenance (never the
    fixture tree — that is in the image the agent sees, and upstream), committed where miners can read them."""
    round_id = rd.name
    dest = cfg.repo / "rounds" / round_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(rd / "tasks", dest / "tasks", dirs_exist_ok=True)
    shutil.copy(rd / "window.json", dest / "window.json")
    for ctx in sorted((rd / "images").glob("*/")) if (rd / "images").exists() else []:
        pub = dest / "images" / ctx.name
        pub.mkdir(parents=True, exist_ok=True)
        for name in ("Dockerfile", "TAG", "SOURCE.json"):
            if (ctx / name).exists():
                shutil.copy(ctx / name, pub / name)
    # The daemon rewrites its index only after its next mint, so drop the round just taken and anything not READY.
    waiting = set(queue_ready(cfg)) - {round_id}
    ready = [r for r in _read(cfg.queue / "index.json", {"ready": []}).get("ready", []) if r["round_id"] in waiting]
    (cfg.repo / "rounds" / "queue.json").write_text(
        json.dumps({"schema": "sh-queue-v1", "published_at": time.time(), "ready": ready}, indent=1)
    )
    live(cfg, rd, "window", progress={}, submissions=[], push=False)
    _commit(
        cfg, f"{round_id}: open — {len(list((rd / 'tasks').glob('*.json')))} tasks, window {cfg.window_s // 60} min"
    )
    log(rd, "open", tasks=len(list((rd / "tasks").glob("*.json"))), closes_at=_read(rd / "window.json")["closes_at"])


def _image_tags(rd: Path) -> list[str]:
    return [p.read_text().strip() for p in sorted((rd / "images").glob("*/TAG"))] if (rd / "images").exists() else []


def window_state(rd: Path, now: float | None = None) -> dict:
    """Where the round is in its window: seconds left, and whether submissions are open."""
    w = _read(rd / "window.json")
    now = now if now is not None else time.time()
    if not w:
        return {"open": False, "remaining": 0}
    return {"open": now < w["closes_at"], "remaining": max(0.0, w["closes_at"] - now), "closes_at": w["closes_at"]}


def _submissions(cfg: Config) -> list[dict]:
    """Open strategy PRs as the board lists them during the window."""
    return [
        {
            "pr": p["number"],
            "hotkey": p["changed"][0],
            "created_at": p.get("createdAt"),
            "updated_at": p.get("updatedAt"),
            "url": p.get("url"),
        }
        for p in _strategy_prs(cfg, f"origin/{BRANCH}")
        if pr_role(p["changed"]) == "strategy"
    ]


def wait_window(cfg: Config, rd: Path, mock: tuple[Path, Path] | None = None) -> None:
    """Hold the round open until the window closes, showing the count of submissions on the board. Mock
    challengers submit at the start of the window, signed for this round, like any miner would."""
    if mock and "mock_submit" not in done_stages(rd):
        from sh.cli.mock_miners import open_prs

        opened = open_prs(mock[0], mock[1], REPO, BRANCH, cfg.repo, rd.name)
        log(rd, "mock_submit", opened=opened)
    last_push = 0.0
    while True:
        st = window_state(rd)
        if not st["open"]:
            break
        if time.time() - last_push > 120:
            live(cfg, rd, "window", submissions=_submissions(cfg))
            last_push = time.time()
        time.sleep(min(60.0, max(1.0, st["remaining"])))
    log(rd, "window", closed_at=time.time())


def _bundle_from_tree(cfg: Config, ref: str, hotkey: str, dest: Path, *, round_id: str | None) -> dict | None:
    """Materialise `submissions/<hotkey>/` as of `ref` into `dest` and lint it; None if there is nothing there.
    With `round_id`, the bundle must carry a valid attestation for that round (a challenger); without, it is an
    incumbent whose attestation was checked when it was sealed."""
    listing = sh(["git", "ls-tree", "-r", "--name-only", ref, f"submissions/{hotkey}/"], cwd=cfg.repo, check=False)
    prefix = f"submissions/{hotkey}/"
    rels = [r for r in listing.split() if r.startswith(prefix)]
    if not rels:
        return None
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for rel in rels:
        out = dest / rel[len(prefix) :]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(sh(["git", "show", f"{ref}:{rel}"], cwd=cfg.repo).encode())
    files, problems = collect(dest)
    result = check_files(files, problems, hotkey=hotkey, round_id=round_id, require_attestation=round_id is not None)
    return {"problems": result["problems"], "digest": result["bundle_sha256"], "attestation": result["attestation"]}


def _changed_submissions(cfg: Config, base: str, head: str) -> list[str]:
    """The submission directories a head changes relative to where it forked from the base (three-dot: the
    merge base, not the base tip — a crown merged after the miner branched is not the miner's change)."""
    names = sh(["git", "diff", "--name-only", f"{base}...{head}", "--", "submissions/"], cwd=cfg.repo, check=False)
    return sorted({p.split("/")[1] for p in names.split() if p.count("/") >= 2 and p.split("/")[1] != "README.md"})


def pr_role(changed: list[str]) -> str:
    """What a pull request is to the round, from the submission directories it touches: none — maintenance,
    never sealed, never closed by a round; one — a strategy; more — a strategy PR done wrong, rejected."""
    return "maintenance" if not changed else "strategy" if len(changed) == 1 else "malformed"


def _strategy_prs(cfg: Config, tip: str) -> list[dict]:
    """Every open PR against the branch that touches `submissions/`, with the directories it changes. Recognised
    by content, not by label — a miner cannot label a PR — and labelled `sh:strategy` here so the board can see
    it. Fork heads are fetched by their pull ref; `origin` alone carries only same-repository branches."""
    sh(["git", "fetch", "-q", "origin", "+refs/pull/*/head:refs/remotes/origin/pr/*"], cwd=cfg.repo, check=False)
    prs = json.loads(
        gh(
            "pr",
            "list",
            "--repo",
            REPO,
            "--base",
            BRANCH,
            "--state",
            "open",
            "--json",
            "number,headRefOid,headRefName,title,labels,createdAt,updatedAt,url",
            "--limit",
            "100",
        )
    )
    out = []
    for pr in prs:
        changed = _changed_submissions(cfg, tip, pr["headRefOid"])
        if not changed:
            continue
        if LABEL_STRATEGY not in {lb["name"] for lb in pr.get("labels", [])}:
            _label(pr["number"], LABEL_STRATEGY)
        out.append({**pr, "changed": changed})
    return out


def one_per_hotkey(prs: list[dict]) -> tuple[dict[str, dict], dict[int, str]]:
    """Of several open PRs for one hotkey, the newest counts; the others are rejected as superseded. Pure."""
    keep: dict[str, dict] = {}
    superseded: dict[int, str] = {}
    for pr in sorted(prs, key=lambda p: p["number"]):
        hotkey = pr["changed"][0]
        if hotkey in keep:
            superseded[keep[hotkey]["number"]] = f"superseded by #{pr['number']} (one PR per hotkey per round)"
        keep[hotkey] = pr
    return keep, superseded


def seal(cfg: Config, round_id: str, rd: Path) -> dict:
    """Which bundles are in this round, decided the instant the window closes and published before evaluation.

    **Incumbents**: strategies already merged into `submissions/` — a crowned king defends the crown every round
    without resubmitting. **Challengers**: every open PR touching exactly one `submissions/<hotkey>/`, taken at
    its head SHA, whose bundle lints and carries the hotkey's signature over *this* round and *this* digest. One
    PR per hotkey: the newest wins. A challenger for a hotkey supersedes that hotkey's incumbent."""
    sh(["git", "fetch", "-q", "origin"], cwd=cfg.repo)
    bundles = rd / "bundles"
    bundles.mkdir(exist_ok=True)
    tip = f"origin/{BRANCH}"
    active: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    for entry in sh(["git", "ls-tree", "--name-only", tip, "submissions/"], cwd=cfg.repo, check=False).split():
        hotkey = entry.split("/")[-1]
        if not hotkey or hotkey == "README.md":
            continue
        b = _bundle_from_tree(cfg, tip, hotkey, bundles / hotkey, round_id=None)
        if b and not b["problems"]:
            active[hotkey] = {"pr": None, "head": tip, "bundle_sha256": b["digest"], "incumbent": True}
    prs = _strategy_prs(cfg, tip)
    for pr in prs:
        if pr_role(pr["changed"]) == "malformed":
            rejected[str(pr["number"])] = f"{len(pr['changed'])} changed submission directories (need exactly 1)"
    keep, superseded = one_per_hotkey([p for p in prs if pr_role(p["changed"]) == "strategy"])
    rejected.update({str(n): why for n, why in superseded.items()})
    for hotkey, pr in keep.items():
        head = pr["headRefOid"]
        b = _bundle_from_tree(cfg, head, hotkey, bundles / hotkey, round_id=round_id)
        if b is None or b["problems"]:
            rejected[str(pr["number"])] = (b or {}).get("problems", ["empty submission"])[0]
            shutil.rmtree(bundles / hotkey, ignore_errors=True)
            if hotkey in active and active[hotkey]["incumbent"]:
                _bundle_from_tree(cfg, tip, hotkey, bundles / hotkey, round_id=None)  # the incumbent stands
            continue
        active[hotkey] = {"pr": pr["number"], "head": head, "bundle_sha256": b["digest"], "incumbent": False}
    for number in [i["pr"] for i in active.values() if i.get("pr")] + [int(n) for n in rejected]:
        _label(number, f"sh:round:{round_id}")
    record = {
        "schema": "sh-seal-v3",
        "round_id": round_id,
        "sealed_at": time.time(),
        "active": active,
        "rejected": rejected,
    }
    (rd / "seal.json").write_text(json.dumps(record, indent=1))
    dest = cfg.repo / "rounds" / round_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(rd / "seal.json", dest / "seal.json")
    live(cfg, rd, "evaluate", progress={"done": 0, "total": 0, "by_surface": {}}, push=False)
    _commit(cfg, f"{round_id}: seal — {len(active)} active bundles, {len(rejected)} rejected")
    log(
        rd,
        "seal",
        active=len(active),
        incumbents=sum(1 for a in active.values() if a["incumbent"]),
        rejected=len(rejected),
    )
    return record


def _progress(cfg: Config, remote: str, total: int) -> dict:
    """Per-surface counts from the worker's episode records so far — what the dashboard shows mid-round."""
    raw = _worker(
        cfg,
        f"cd {remote}/episodes 2>/dev/null && for s in *; do "
        f"n=$(ls $s/*/episode.json 2>/dev/null | wc -l); "
        f"v=$(grep -l '\"verified_success\": true' $s/*/episode.json 2>/dev/null | wc -l); "
        f'echo "$s $n $v"; done',
    )
    by = {}
    for line in raw.split("\n"):
        parts = line.split()
        if len(parts) == 3 and parts[1].isdigit():
            by[parts[0]] = {"n": int(parts[1]), "verified": int(parts[2])}
    return {"done": sum(v["n"] for v in by.values()), "total": total, "by_surface": by}


def evaluate(cfg: Config, rd: Path, sealed: dict) -> None:
    """Runs on the GPU worker in the background while this side publishes progress every few minutes. The
    worker gets exactly what an episode needs — tasks, withheld halves for the grader, checks, canon, the sealed
    bundles — and hands the episodes back. No credential is ever on it."""
    remote = f"{cfg.worker_root}/rounds/{rd.name}"
    _worker(cfg, f"mkdir -p {remote}")
    for sub in ("tasks", "withheld", "checks", "canon", "bundles"):
        if (rd / sub).exists():
            _rsync(f"{rd / sub}/", f"{cfg.worker}:{remote}/{sub}/", cfg)
    if (rd / "images").exists():  # image-defined tasks: the worker builds each task's image from its context
        _worker(cfg, f"mkdir -p {remote}/images")
        _rsync(f"{rd / 'images'}/", f"{cfg.worker}:{remote}/images/", cfg)
        built = _worker(
            cfg,
            f"for d in {remote}/images/*/; do t=$(cat $d/TAG); docker image inspect $t >/dev/null 2>&1 "
            f'|| docker build -q -t $t $d >/dev/null 2>&1 || echo "FAILED $t"; done; echo built',
        )
        if "FAILED" in built:
            raise RuntimeError(f"task image build failed on the worker: {built.strip()[-300:]}")
        log(rd, "images", built=len(list((rd / "images").iterdir())))
    surfaces = ["null", f"canon={remote}/canon"] + [f"{h}={remote}/bundles/{h}" for h in sealed["active"]]
    total = len(surfaces) * len(list((rd / "tasks").glob("*.json")))
    # Resume-safe: a restarted control plane finds the batch still running and polls it rather than launching a
    # second one; if it is not running, launching is safe — `batch` resumes on its own episode records.
    already = _worker(cfg, f"pgrep -f '[b]atch --round {remote}' | wc -l").strip() not in ("", "0")
    if already:
        log(rd, "evaluate_resume", note="batch already running on the worker; polling")
    else:
        _worker_launch(
            cfg,
            f"cd {cfg.worker_root}/pkg && PYTHONPATH={cfg.worker_root}/pkg setsid nohup python3 -m sh.validator.batch "
            f"--round {remote} --surfaces {','.join(surfaces)} --image {cfg.image} --inference unused "
            f"--out {remote}/episodes --concurrency {cfg.concurrency} --network sh-ep "
            f"--tokens {cfg.worker_root}/state/tokens --usage-dir {cfg.worker_root}/state/usage "
            f"> {remote}/batch.log 2>&1 < /dev/null & echo started",
        )
    last_push = 0.0
    while True:
        time.sleep(45)
        running = _worker(cfg, f"pgrep -f '[b]atch --round {remote}' | wc -l").strip()
        prog = _progress(cfg, remote, total)
        if running == "0" or time.time() - last_push > 150:
            live(cfg, rd, "evaluate", progress=prog)
            last_push = time.time()
        if running == "0":
            break
    _rsync(f"{cfg.worker}:{remote}/episodes/", f"{rd / 'episodes'}/", cfg)
    if tags := _image_tags(rd):
        _worker(
            cfg,
            "docker image rm -f " + " ".join(tags) + " >/dev/null 2>&1; docker image prune -f >/dev/null 2>&1; "
            "docker builder prune -f --filter until=48h >/dev/null 2>&1; true",
        )
    # Rounds older than the previous one leave the worker: their episodes are archived here.
    _worker(cfg, f"ls -d {cfg.worker_root}/rounds/r* 2>/dev/null | sort | head -n -2 | xargs -r rm -rf")
    n = len(list((rd / "episodes").rglob("episode.json")))
    if not n:
        raise RuntimeError("the worker returned no episodes")
    log(rd, "evaluate", episodes=n, surfaces=len(surfaces))


def window_archive(cfg: Config, round_id: str) -> Path:
    """The last W rounds' episodes, pooled: what the scorer sees. Each round's episodes are archived once,
    under the round they belong to, so a re-run of the loop never double-counts."""
    archive = cfg.state / "archive"
    archive.mkdir(exist_ok=True)
    src = cfg.rounds / round_id / "episodes"
    dst = archive / round_id
    if src.exists() and not dst.exists():
        shutil.copytree(src, dst)
    rounds = sorted(p.name for p in archive.iterdir() if p.is_dir())[-cfg.window :]
    pooled = cfg.state / "window"
    shutil.rmtree(pooled, ignore_errors=True)
    for r in rounds:
        shutil.copytree(archive / r, pooled / r)
    (pooled / "rounds.json").write_text(json.dumps(rounds))
    return pooled


def close(cfg: Config, round_id: str, rd: Path) -> dict:
    pooled = window_archive(cfg, round_id)
    window = json.loads((pooled / "rounds.json").read_text())
    record = close_round(rd, pooled, rd / "close", reveal_dir=rd / "withheld", era=cfg.era, window=window)
    log(
        rd,
        "close",
        window=window,
        episodes=record["episodes"],
        commitments_ok=record["commitments_ok"],
        weights=record["weights"],
    )
    return record


def crown_round(cfg: Config, rd: Path, record: dict, sealed: dict) -> dict:
    """This round's king, from this round's episodes alone (spec: the crown is a merge, not a payment)."""
    pooled = {h: s.get("delta_c", 0.0) for h, s in record.get("scores", {}).items()}
    result = crown_rule(
        load_episodes(rd / "episodes"),
        set(sealed["active"]),
        round_id=rd.name,
        pooled_delta_c=pooled,
        min_paired=cfg.min_paired,
    )
    (rd / "close" / "crown.json").write_text(json.dumps(result, indent=1))
    log(rd, "crown", king=result["king"], ranked=[h for h, s in result["standings"].items() if "rank" in s])
    return result


def outcome(sealed: dict, king: str | None) -> dict:
    """What happens to each PR the seal named: the king's PR is merged, every other challenger's is closed, the
    rejected ones too. Pure. Only PRs the seal named are ever touched — a maintenance PR is never in the seal."""
    if king is not None and king not in sealed["active"]:
        raise ValueError(f"king {king} is not a sealed strategy")
    king_pr = sealed["active"].get(king, {}).get("pr") if king else None
    close_prs = sorted(
        {info["pr"] for h, info in sealed["active"].items() if info.get("pr") and h != king}
        | {int(n) for n in sealed.get("rejected", {})}
    )
    return {"king": king, "merge": king_pr, "close": close_prs}


def dethroned(sealed: dict, king: str | None) -> list[str]:
    """Incumbents that are not this round's king. `submissions/` carries exactly the current king: a strategy
    that lost the crown does not keep competing for free, round after round, at the validator's expense. Pure."""
    return sorted(h for h, info in sealed["active"].items() if info.get("incumbent") and h != king)


def announce(cfg: Config, round_id: str, rd: Path, record: dict, sealed: dict, crowned: dict) -> str | None:
    """Scorecards on every PR; `scored` on every PR; the crown moved to the king; the king's PR merged; every
    other competition PR closed with the reason; PRs that arrived after the seal closed as outside the window."""
    reveal = json.loads((rd / "close" / "reveal.json").read_text())
    (rd / "scorecards").mkdir(exist_ok=True)
    plan = outcome(sealed, crowned["king"])
    king = plan["king"]
    already = {
        pr["number"]
        for pr in json.loads(
            gh(
                "pr",
                "list",
                "--repo",
                REPO,
                "--label",
                LABEL_SCORED,
                "--state",
                "all",
                "--json",
                "number",
                "--limit",
                "200",
            )
        )
    }
    for hotkey, info in sealed["active"].items():
        st = crowned["standings"].get(hotkey, {})
        this_round = (
            f"**This round:** {'rank ' + str(st['rank']) if st.get('rank') else 'not ranked'} · Δ vs baseline "
            f"{st['delta']:+.3f} on {st['n']} paired instances · {st['verified']} verified.\n\n"
            if st.get("delta") is not None
            else ""
        )
        card = this_round + render_scorecard(record, hotkey, reveal)
        (rd / "scorecards" / f"{hotkey}.md").write_text(card)
        if not info.get("pr") or info["pr"] in already:
            continue  # an incumbent has no PR to write to; a PR scored before a restart is not written to twice
        body = card
        if hotkey == king:
            body = "👑 **Crowned: best against the baseline on this round's instances. Merging.**\n\n" + body
        else:
            body += f"\n\n---\nNot crowned in `{round_id}`; this PR is closed with the round. Submit again in the next window."
        gh("pr", "comment", str(info["pr"]), "--repo", REPO, "--body", body)
        _label(info["pr"], LABEL_SCORED)
    # One crown. Remove it wherever it was; place it on the king's PR.
    for pr in json.loads(
        gh("pr", "list", "--repo", REPO, "--label", LABEL_CROWN, "--state", "all", "--json", "number", "--limit", "100")
    ):
        sh(["gh", "api", "-X", "DELETE", f"repos/{REPO}/issues/{pr['number']}/labels/{LABEL_CROWN}"], check=False)
    if plan["merge"]:
        _label(plan["merge"], LABEL_CROWN)
        merged = subprocess.run(
            [
                "gh",
                "pr",
                "merge",
                str(plan["merge"]),
                "--repo",
                REPO,
                "--squash",
                "--subject",
                f"crown {round_id}: {king}",
            ],
            capture_output=True,
            text=True,
        )
        log(rd, "merge", pr=plan["merge"], ok=merged.returncode == 0, err=merged.stderr[-200:])
    elif king:
        log(rd, "merge", pr=None, ok=True, note="incumbent retains the crown")
    if gone := dethroned(sealed, king):  # after the merge, so the tree the removal commits onto is current
        sh(["git", "pull", "-q", "--rebase", "origin", BRANCH], cwd=cfg.repo, check=False)
        for hotkey in gone:
            sh(["git", "rm", "-r", "-q", f"submissions/{hotkey}"], cwd=cfg.repo, check=False)
        if sh(["git", "status", "--porcelain", "submissions"], cwd=cfg.repo).strip():
            sh(
                ["git", "commit", "-q", "-m", f"{round_id}: dethroned {', '.join(gone)}", "--", "submissions"],
                cwd=cfg.repo,
            )
            sh(["git", "push", "-q", "origin", BRANCH], cwd=cfg.repo)
        log(rd, "dethroned", hotkeys=gone)
    for number in plan["close"]:
        reason = sealed.get("rejected", {}).get(str(number))
        why = f"rejected at seal: {reason}" if reason else f"not crowned in `{round_id}`"
        sh(
            [
                "gh",
                "pr",
                "close",
                str(number),
                "--repo",
                REPO,
                "--comment",
                f"Closed with round `{round_id}` — {why}. Submit again in the next window.",
            ],
            check=False,
        )
    sealed_prs = {i["pr"] for i in sealed["active"].values() if i.get("pr")} | {
        int(n) for n in sealed.get("rejected", {})
    }
    sh(["git", "fetch", "-q", "origin"], cwd=cfg.repo)
    late = [pr["number"] for pr in _strategy_prs(cfg, f"origin/{BRANCH}") if pr["number"] not in sealed_prs]
    for number in late:
        sh(
            [
                "gh",
                "pr",
                "close",
                str(number),
                "--repo",
                REPO,
                "--comment",
                f"Arrived after the submission window of `{round_id}` closed, so it was not sealed. "
                "Fetch the next round's tasks when its window opens and submit again.",
            ],
            check=False,
        )
    log(rd, "announce", king=king, merged=plan["merge"], closed=plan["close"], late=late)
    return king


def export_and_upload(cfg: Config, round_id: str, rd: Path, king: str | None) -> dict:
    manifest = build_exports(rd, rd / "episodes", rd / "close" / "close.json", rd / "export", king=king)
    token = os.environ.get("HF_TOKEN", "")
    if not manifest["sft_rows"] and not manifest["dpo_pairs"]:
        why = "no king this round" if king is None else "the king's episodes yielded no rows"
        log(rd, "export", sft=0, dpo=0, uploaded=False, reason=why)
        return {"manifest": manifest, "upload": {"uploaded": False, "reason": why}}
    if not token:
        log(
            rd, "export", sft=manifest["sft_rows"], dpo=manifest["dpo_pairs"], uploaded=False, reason="HF_TOKEN not set"
        )
        return {"manifest": manifest, "upload": {"uploaded": False, "reason": "HF_TOKEN not set"}}
    result = upload_exports(rd / "export", HF_REPO, token, round_id=round_id)
    log(
        rd,
        "export",
        sft=manifest["sft_rows"],
        dpo=manifest["dpo_pairs"],
        uploaded=result.get("uploaded"),
        url=result.get("url"),
    )
    return {"manifest": manifest, "upload": result}


def publish_close(cfg: Config, round_id: str, rd: Path, record: dict, crowned: dict, exported: dict) -> None:
    """Everything a miner needs to check the round, in the repository, under the round."""
    sh(["git", "pull", "-q", "--rebase", "origin", BRANCH], cwd=cfg.repo, check=False)  # the merge just landed
    king = crowned["king"]
    dest = cfg.repo / "rounds" / round_id
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("close.json", "reveal.json", "crown.json"):
        shutil.copy(rd / "close" / name, dest / name)
    if (rd / "checks").exists():
        shutil.copytree(rd / "checks", dest / "checks", dirs_exist_ok=True)  # semantics of `custom` predicates
    shutil.copytree(rd / "scorecards", dest / "scorecards", dirs_exist_ok=True)
    shutil.copy(rd / "export" / "manifest.json", dest / "manifest.json")
    artefacts = sorted(p.name for p in dest.iterdir())
    entry = {
        "round_id": round_id,
        "closed_at": time.time(),
        "window": _read(rd / "window.json"),
        "episodes": record["episodes"],
        "king": king,
        "weights": record["weights"],
        "scores": _trim_scores(record),
        "crown": {h: st for h, st in crowned.get("standings", {}).items() if st.get("rank")},
        "commitments_ok": record["commitments_ok"],
        "sft_rows": exported["manifest"]["sft_rows"],
        "dpo_pairs": exported["manifest"]["dpo_pairs"],
        "hf": exported["upload"].get("url"),
    }
    index_path = cfg.repo / "rounds" / "index.json"
    index = _read(index_path, {"schema": "sh-rounds-index-v2", "rounds": []})
    index["rounds"] = [r for r in index["rounds"] if r["round_id"] != round_id] + [entry]
    index_path.write_text(json.dumps(index, indent=1))
    page = cfg.repo / "docs" / "rounds" / round_id  # the round's page, where Pages serves it
    page.mkdir(parents=True, exist_ok=True)
    (page / "index.html").write_text(
        render_leaderboard(record, {**entry, "crown": crowned, "artefacts": artefacts, "repo": REPO, "branch": BRANCH})
    )
    live(cfg, rd, "done", push=False)
    _commit(
        cfg,
        f"{round_id}: close — king {king or 'none'}, {record['episodes']} episodes, "
        f"{exported['manifest']['sft_rows']} SFT rows, {exported['manifest']['dpo_pairs']} DPO pairs",
    )
    log(rd, "publish_close", king=king)


# ─── the round, and the loop ───────────────────────────────────────────────────────────────────────
def done_stages(rd: Path) -> set[str]:
    """The stages a round has completed, from its own log — what a restart resumes from (spec §5.7)."""
    p = rd / "phases.jsonl"
    if not p.exists():
        return set()
    return {json.loads(line)["stage"] for line in p.read_text().splitlines() if line.strip()}


def unfinished_round(cfg: Config) -> Path | None:
    """The latest round directory without a DONE marker, if any — the one a restarted loop must pick up."""
    if not cfg.rounds.exists():
        return None
    rounds = sorted(p for p in cfg.rounds.iterdir() if p.is_dir() and p.name.startswith("r"))
    if rounds and not (rounds[-1] / "DONE").exists():
        return rounds[-1]
    return None


def _logged(rd: Path, stage: str, field: str):
    for line in (rd / "phases.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("stage") == stage:
            return rec.get(field)
    return None


def run_round(cfg: Config, mock: tuple[Path, Path] | None = None, resume: Path | None = None) -> dict:
    """One round. With `resume`, the stages that round already logged are skipped, and the seal that was
    published is reused rather than recomputed — a restart must never change what a round sealed. A round
    minted by the previous loop (logged `mint`, never `open`) is taken as already open with its window over."""
    if resume:
        rd, round_id = resume, resume.name
        done = done_stages(rd)
        log(rd, "resume", completed=sorted(done))
    else:
        rd = open_round(cfg)
        round_id, done = rd.name, set()
    if "open" not in done and "seal" not in done:  # a restart between taking a round and publishing it
        publish_round(cfg, rd)
    if "window" not in done and "seal" not in done:
        wait_window(cfg, rd, mock)
    if "seal" not in done:
        sealed = seal(cfg, round_id, rd)
    else:
        sealed = json.loads((rd / "seal.json").read_text())
    if "evaluate" not in done:
        evaluate(cfg, rd, sealed)
    live(cfg, rd, "close")
    record = close(cfg, round_id, rd)
    crowned = crown_round(cfg, rd, record, sealed)
    if "announce" not in done:
        live(cfg, rd, "announce", push=False)
        king = announce(cfg, round_id, rd, record, sealed, crowned)
    else:  # announced before the restart: the king is in the round's own log
        king = _logged(rd, "announce", "king")
    live(cfg, rd, "export", push=False)
    exported = export_and_upload(cfg, round_id, rd, king)
    publish_close(cfg, round_id, rd, record, crowned, exported)
    (rd / "DONE").write_text(json.dumps({"king": king, "weights": record["weights"]}))
    log(rd, "done", king=king)
    return {"round_id": round_id, "king": king, "weights": record["weights"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.environ.get("SH_STATE", str(Path.home() / ".spark-hermes-state")))
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--queue", default=str(Path(__file__).resolve().parents[3] / "Spark-Hermes-Withheld" / "queue"))
    ap.add_argument("--pkg", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--window", type=int, default=8, help="rounds pooled for payment")
    ap.add_argument("--window-minutes", type=int, default=120, help="the submission window")
    ap.add_argument("--min-paired", type=int, default=4)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--pause", type=int, default=0, help="seconds between rounds")
    ap.add_argument("--mock-miners", help="directory of mock miner bundles that submit each window (test only)")
    ap.add_argument("--mock-keys", help="directory holding the mock miners' keypairs (created on demand)")
    a = ap.parse_args(argv)
    cfg = Config(
        state=Path(a.state),
        repo=Path(a.repo),
        queue=Path(a.queue),
        pkg=Path(a.pkg),
        window=a.window,
        window_s=a.window_minutes * 60,
        min_paired=a.min_paired,
    )
    mock = (Path(a.mock_miners), Path(a.mock_keys or (cfg.state / "mock-keys"))) if a.mock_miners else None
    resume = unfinished_round(cfg)  # a restart picks up the round it was in the middle of
    while True:
        try:
            result = run_round(cfg, mock, resume=resume)
            resume = None
            print(json.dumps(result), flush=True)
        except Exception as e:  # a failed round is logged, and resumed — never abandoned for a fresh one
            print(f"round failed: {e!r}", flush=True)
            if a.once:
                return 1
            time.sleep(60)
            resume = unfinished_round(cfg)
            continue
        if a.once:
            return 0
        time.sleep(a.pause)


if __name__ == "__main__":
    sys.exit(main())
