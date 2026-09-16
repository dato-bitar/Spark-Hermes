"""Read a batch/reference archive and print the per-surface table that sets `B` and `T` (spec §5.8).

python -m sh.validator.report OUT_DIR [--family posix_report --efficiency-reference '{"api_calls": 6}']
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sh.validator.stats import family_stats, load_episodes

COLS = ("n", "verified", "overfit", "dq", "partial", "self_checked", "wall_p50", "wall_p90", "calls_p50", "tokens_in")


def rows(episodes: list[dict]) -> dict[str, dict]:
    out = {}
    for surface in sorted({e.get("surface", "?") for e in episodes}):
        eps = [e for e in episodes if e.get("surface") == surface]
        walls = sorted(float(e.get("wall_s") or 0) for e in eps)
        calls = sorted(float(e.get("api_calls") or 0) for e in eps)
        toks = [((e.get("tokens") or {}).get("prompt_tokens") or 0) for e in eps]

        def pick(xs, q):
            return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else 0

        out[surface] = {
            "n": len(eps),
            "verified": sum(1 for e in eps if e.get("verified_success")),
            "overfit": sum(1 for e in eps if e.get("overfit")),
            "dq": sum(1 for e in eps if e.get("disqualified")),
            "partial": sum(1 for e in eps if e.get("partial") or e.get("timed_out")),
            "self_checked": sum(1 for e in eps if e.get("self_checked")),
            "wall_p50": round(statistics.median(walls), 1) if walls else 0,
            "wall_p90": round(pick(walls, 0.9), 1),
            "calls_p50": round(statistics.median(calls), 1) if calls else 0,
            "tokens_in": round(statistics.mean(toks)) if any(toks) else 0,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--family")
    ap.add_argument("--window", default="")
    ap.add_argument("--era", default="e0")
    ap.add_argument("--efficiency-reference", default="{}")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    eps = load_episodes(*[Path(p) for p in a.paths])
    table = rows(eps)
    if a.json:
        print(json.dumps(table, indent=1))
    else:
        print(f"{'surface':>8} " + " ".join(f"{c:>12}" for c in COLS))
        for s, r in table.items():
            print(f"{s:>8} " + " ".join(f"{r[c]:>12}" for c in COLS))
    if a.family:
        window = a.window.split(",") if a.window else sorted({e.get("round_id", "") for e in eps})
        st = family_stats(eps, a.family, window, a.era, json.loads(a.efficiency_reference))
        print("\n" + json.dumps(st.record(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
