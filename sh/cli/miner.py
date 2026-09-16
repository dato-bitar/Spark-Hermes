"""The miner's CLI: fetch the current round's tasks, submit a signed strategy, see where you stand.

    python -m sh.cli.miner tasks  [--out tasks/]                       # this round's tasks; nothing else exists to fetch
    python -m sh.cli.miner submit --bundle DIR --key HOTKEY_FILE       # one PR per hotkey per round; resubmit to replace
    python -m sh.cli.miner status [--hotkey SS58]

Only the open round's tasks are in the repository — future rounds exist as digests until they open — so `tasks`
cannot fetch ahead. `submit` lints the bundle, signs `round_id:bundle_sha256` with the hotkey, writes
`attestation.json` beside the prose, and pushes `miner/<hotkey>` from a temporary worktree: the first push
opens the pull request, every later push replaces it. It refuses outside the submission window, because the
validator would.

A miner working from a fork passes `--head-owner <github user>`; the branch is pushed to `origin` (the fork)
and the pull request is opened against the competition repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from sh.cli import attest
from sh.cli.lint import bundle_digest, check, collect

REPO = "gittensor-model-hub/Spark-Hermes"
BRANCH = "main"
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"
CONTENTS = "https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "spark-hermes-miner"}), timeout=30
    ) as r:
        return r.read()


def current(repo: str = REPO, branch: str = BRANCH) -> dict:
    """The board's live state: round, stage, window."""
    return json.loads(_get(RAW.format(repo=repo, branch=branch, path="docs/live/live.json") + f"?t={int(time.time())}"))


def _remaining(live: dict) -> str:
    w = live.get("window") or {}
    if not w or live.get("stage") != "window":
        return "closed"
    s = max(0, int(w["closes_at"] - time.time()))
    return f"{s // 3600} h {s % 3600 // 60} min left"


def tasks(out: Path, repo: str = REPO, branch: str = BRANCH) -> dict:
    """Download the open round's tasks to `out/<round_id>/`. Refuses to guess at any other round."""
    live = current(repo, branch)
    rid, stage = live["round_id"], live["stage"]
    listing = json.loads(_get(CONTENTS.format(repo=repo, branch=branch, path=f"rounds/{rid}/tasks")))
    dest = out / rid
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for entry in listing:
        if entry.get("type") == "file" and entry["name"].endswith(".json"):
            (dest / entry["name"]).write_bytes(_get(entry["download_url"]))
            names.append(entry["name"])
    if live.get("window"):
        (dest / "window.json").write_text(json.dumps(live["window"]))
    return {"round_id": rid, "stage": stage, "window": _remaining(live), "tasks": sorted(names), "dir": str(dest)}


def _run(cmd: list[str], cwd: Path | None = None, check_rc: bool = True) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check_rc and r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])}… exited {r.returncode}: {r.stderr[-400:]}")
    return r.stdout


def _open_pr_for(repo: str, base: str, head: str) -> int | None:
    prs = json.loads(
        _run(
            ["gh", "pr", "list", "--repo", repo, "--base", base, "--head", head, "--state", "open", "--json", "number"]
        )
    )
    return prs[0]["number"] if prs else None


def submit_bundle(
    bundle: Path,
    keypair,
    *,
    round_id: str,
    repo: str = REPO,
    base: str = BRANCH,
    checkout: Path | None = None,
    head_owner: str | None = None,
    remote: str = "origin",
) -> dict:
    """Lint, sign for `round_id`, push `miner/<hotkey>`, open the PR if none is open. Returns what happened."""
    files, problems = collect(bundle)
    files.pop(attest.FILE, None)  # an old attestation is replaced, never linted
    verdict = check(bundle)
    prose_problems = [p for p in verdict["problems"] if not p.startswith("L10")]
    if prose_problems:
        return {"ok": False, "problems": prose_problems}
    digest = bundle_digest(files)
    hotkey = keypair.ss58_address
    att = attest.sign(keypair, round_id, digest)
    checkout = checkout or Path.cwd()
    branch = f"miner/{hotkey}"
    head = f"{head_owner}:{branch}" if head_owner else branch
    _run(["git", "fetch", "-q", remote, base], cwd=checkout)
    tmp = Path(tempfile.mkdtemp(prefix="sh-submit-"))
    wt = tmp / "wt"
    try:
        _run(["git", "worktree", "add", "-q", "--detach", str(wt), f"{remote}/{base}"], cwd=checkout)
        dest = wt / "submissions" / hotkey
        incumbent = None
        if dest.is_dir():
            inc_files, _ = collect(dest)
            inc_files.pop(attest.FILE, None)
            incumbent = bundle_digest(inc_files)
        if incumbent == digest:
            return {
                "ok": True,
                "hotkey": hotkey,
                "bundle_sha256": digest,
                "skipped": "already the incumbent, byte for byte",
            }
        _run(["git", "checkout", "-q", "-B", branch, f"{remote}/{base}"], cwd=wt)
        shutil.rmtree(dest, ignore_errors=True)
        for rel, data in files.items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (dest / attest.FILE).write_text(json.dumps(att, indent=1) + "\n")
        final = check(dest, hotkey=hotkey, round_id=round_id, require_attestation=True)
        if not final["ok"]:
            return {"ok": False, "problems": final["problems"]}
        _run(["git", "add", f"submissions/{hotkey}"], cwd=wt)
        _run(["git", "commit", "-q", "-m", f"miner: {hotkey} for {round_id}\n\nbundle_sha256 {digest}"], cwd=wt)
        _run(["git", "push", "-q", "-f", "-u", remote, branch], cwd=wt)
        number = _open_pr_for(repo, base, head)
        created = number is None
        if created:
            url = _run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    repo,
                    "--base",
                    base,
                    "--head",
                    head,
                    "--title",
                    f"miner: {hotkey}",
                    "--body",
                    f"Strategy for round `{round_id}` by hotkey `{hotkey}`.\n\n`bundle_sha256` `{digest}`, signed "
                    f"(`attestation.json`). Linted with `python -m sh.cli.lint` — ok.",
                ]
            ).strip()
            number = int(url.rstrip("/").split("/")[-1])
        return {"ok": True, "hotkey": hotkey, "pr": number, "bundle_sha256": digest, "created": created}
    finally:
        _run(["git", "worktree", "remove", "--force", str(wt)], cwd=checkout, check_rc=False)
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spark-Hermes miner")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--base", default=BRANCH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks", help="fetch the open round's tasks")
    t.add_argument("--out", default="tasks")
    s = sub.add_parser("submit", help="sign and submit a strategy for the open round")
    s.add_argument("--bundle", required=True)
    s.add_argument("--key", required=True, help="bittensor hotkey file, mnemonic, or //dev URI")
    s.add_argument("--checkout", default=".", help="a clone whose `origin` you can push to")
    s.add_argument("--head-owner", help="your GitHub user, when pushing to a fork")
    s.add_argument("--round", help="override the round (normally read from the board)")
    s.add_argument("--force", action="store_true", help="submit even if the window is not open")
    st = sub.add_parser("status", help="the round, the window, and your PR")
    st.add_argument("--hotkey")
    a = ap.parse_args(argv)

    if a.cmd == "tasks":
        r = tasks(Path(a.out), a.repo, a.base)
        print(f"round    {r['round_id']} ({r['stage']}; window {r['window']})")
        print(f"tasks    {len(r['tasks'])} -> {r['dir']}")
        return 0
    if a.cmd == "submit":
        live = current(a.repo, a.base)
        if live.get("stage") != "window" and not a.round and not a.force:
            print(
                f"the submission window is not open (round {live.get('round_id')} is at {live.get('stage')})",
                file=sys.stderr,
            )
            return 1
        rid = a.round or live["round_id"]
        kp = attest.load_keypair(a.key)
        r = submit_bundle(
            Path(a.bundle),
            kp,
            round_id=rid,
            repo=a.repo,
            base=a.base,
            checkout=Path(a.checkout),
            head_owner=a.head_owner,
        )
        if not r["ok"]:
            print("not submitted:", file=sys.stderr)
            for p in r["problems"]:
                print(f"  - {p}", file=sys.stderr)
            return 1
        if r.get("skipped"):
            print(f"{r['hotkey']}: {r['skipped']}")
            return 0
        print(
            f"{'opened' if r['created'] else 'updated'} PR #{r['pr']} for {r['hotkey']} in {rid} (digest {r['bundle_sha256'][:12]}…)"
        )
        return 0
    live = current(a.repo, a.base)
    print(f"round    {live['round_id']} · {live['stage']} · window {_remaining(live)}")
    if a.hotkey:
        n = _open_pr_for(a.repo, a.base, f"miner/{a.hotkey}")
        print(f"your PR  {'#' + str(n) if n else 'none open'}")
        st = (live.get("crown") or {}).get("standings", {}).get(a.hotkey)
        if st:
            print(f"crown    rank {st.get('rank', '—')} · Δ {st['delta']:+.3f} on {st['n']} paired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
