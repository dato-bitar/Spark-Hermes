"""The reference arms of a round (spec §5.4, FR-BL-1/2): what the pinned baseline does with no miner files
(`null`) and with the family's own published CANON prose (`canon`).

Neither arm is ever weighted. NULL is the denominator every miner is measured against; CANON is the yardstick
that says how much *any* prose can move this family — when it stops helping, the family has stopped measuring
anything (FR-BL-5). Both arms run through the same runner, boundary and grader as a miner's episode, because a
reference produced by a different path would not be a reference.

    python -m sh.reference.run --round DIR --arm null --arm canon --canon-dir DIR --image IMG \
        --out OUT --network sh-ep --tokens FILE --usage-dir DIR
    python -m sh.reference.run --stats --archive DIR --family posix_report --window r1,r2 --era e0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sh.validator.batch import one
from sh.validator.stats import family_stats, load_episodes

ARMS = ("null", "canon")


def run_arms(
    round_dir: Path,
    arms: list[str],
    canon_dir: Path | None,
    image: str,
    inference: str,
    out: Path,
    *,
    concurrency: int = 1,
    network: str = "host",
    tokens: Path | None = None,
    usage_dir: Path | None = None,
) -> list[dict]:
    tasks = [json.loads(p.read_text()) for p in sorted((round_dir / "tasks").glob("*.json"))]
    withheld = {p.stem: json.loads(p.read_text()) for p in (round_dir / "withheld").glob("*.json")}
    if "canon" in arms and canon_dir is None:
        raise SystemExit("--arm canon needs --canon-dir (the family's published canon/ surface)")
    jobs = [(t, withheld.get(t["task_id"]), arm, canon_dir if arm == "canon" else None) for arm in arms for t in tasks]
    recs, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {
            pool.submit(one, t, w, arm, b, image, inference, out, network, tokens, usage_dir): (arm, t["task_id"])
            for t, w, arm, b in jobs
        }
        for f in as_completed(futs):
            arm, tid = futs[f]
            try:
                r = f.result()
                recs.append(r)
                print(
                    f"{arm:>6} {tid}: verified={r['verified_success']} calls={r['api_calls']} wall={r['wall_s']}",
                    flush=True,
                )
            except Exception as e:
                print(f"{arm:>6} {tid}: ERROR {e!r}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "reference.jsonl").open("a") as f:  # append-only: the archive is the source of truth
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} reference episodes in {time.time() - t0:.0f}s → {out / 'reference.jsonl'}", flush=True)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round")
    ap.add_argument("--arm", action="append", choices=ARMS, default=[])
    ap.add_argument("--canon-dir")
    ap.add_argument("--image")
    ap.add_argument("--inference", default="unused")
    ap.add_argument("--out")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--network", default="host")
    ap.add_argument("--tokens")
    ap.add_argument("--usage-dir")
    ap.add_argument("--stats", action="store_true", help="recompute FamilyStats from an archive instead of running")
    ap.add_argument("--archive", action="append", default=[])
    ap.add_argument("--family")
    ap.add_argument("--window")
    ap.add_argument("--era", default="e0")
    ap.add_argument("--efficiency-reference", default="{}")
    a = ap.parse_args(argv)
    if a.stats:
        if not (a.archive and a.family and a.window):
            raise SystemExit("--stats needs --archive, --family and --window")
        s = family_stats(
            load_episodes(*[Path(p) for p in a.archive]),
            a.family,
            a.window.split(","),
            a.era,
            json.loads(a.efficiency_reference),
        )
        print(json.dumps(s.record(), indent=1))
        return 0
    if not (a.round and a.image and a.out):
        raise SystemExit("running arms needs --round, --image and --out")
    run_arms(
        Path(a.round),
        a.arm or list(ARMS),
        Path(a.canon_dir) if a.canon_dir else None,
        a.image,
        a.inference,
        Path(a.out),
        concurrency=a.concurrency,
        network=a.network,
        tokens=Path(a.tokens) if a.tokens else None,
        usage_dir=Path(a.usage_dir) if a.usage_dir else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
