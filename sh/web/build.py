"""The leaderboard (spec §10) — a static page built from a closed round, nothing served.

It is generated from `close.json` alone, which is the point: everything a miner sees here is something they can
recompute from published artefacts. A leaderboard that knows something its readers cannot check is a leaderboard
they have to trust.

    python -m sh.web.build --close DIR/close.json --out site/
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

STYLE = """
:root { --bg:#fbfaf8; --ink:#1b1a18; --muted:#6b6862; --line:#e4e0d8; --good:#1d6a4f; --warn:#8a5a1a; }
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#16151a; --ink:#ece9e3; --muted:#9a968e; --line:#2e2c33; --good:#63c79e; --warn:#d9a441; } }
:root[data-theme="dark"] { --bg:#16151a; --ink:#ece9e3; --muted:#9a968e; --line:#2e2c33; --good:#63c79e; --warn:#d9a441; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, sans-serif;
       padding-block: 2.5rem; padding-left: 20px; padding-right: 20px; max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
.sub { color: var(--muted); margin: 0 0 2rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .55rem .6rem; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; }
.hotkey { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.zero { color: var(--muted); }
.why { color: var(--warn); font-size: .82rem; }
.ok { color: var(--good); }
section { margin-top: 2.5rem; }
h2 { font-size: .95rem; margin: 0 0 .6rem; }
.note { color: var(--muted); font-size: .85rem; }
.wrap { overflow-x: auto; }
"""


def render(close: dict) -> str:
    rows = sorted(close["scores"].values(), key=lambda s: -s["score"])
    weights = close.get("weights", {})
    body = []
    for rank, s in enumerate(rows, 1):
        w = weights.get(s["hotkey"], 0.0)
        reason = f'<div class="why">{html.escape(str(s["reason"]))}</div>' if s.get("reason") else ""
        body.append(
            f'<tr><td><span class="hotkey">{html.escape(s["hotkey"])}</span>{reason}</td>'
            f"<td>{rank}</td><td>{s['n']}</td>"
            f"<td>{s.get('mean_d', 0):+.3f}</td><td>{s['delta_c']:.3f}</td>"
            f"<td>{'yes' if s['gate'] else 'no'}</td>"
            f"<td>{s['overfit_rate']:.2f}</td><td>{s['dq']}</td>"
            f'<td class="{"zero" if w == 0 else ""}">{w:.3f}</td></tr>'
        )

    verified = close.get("commitments_verified", {})
    checked = [t for t, v in verified.items() if v is not None]
    ok = close.get("commitments_ok")
    families = "".join(
        f"<tr><td>{html.escape(f)}</td><td>{r['null']['n']}</td><td>{r['null']['p'] if r['null']['p'] is None else f'{r["null"]["p"]:.2f}'}</td>"
        f"<td>{r['canon']['p'] if r['canon']['p'] is None else f'{r["canon"]["p"]:.2f}'}</td>"
        f"<td>{r['canon'].get('delta_c')}</td><td>{html.escape(str(r['label']))}</td></tr>"
        for f, r in close.get("family_stats", {}).items()
    )

    return f"""<title>Spark-Hermes {html.escape(str(close.get("round_id", "")))}</title>
<style>{STYLE}</style>
<h1>Round {html.escape(str(close.get("round_id", "")))}</h1>
<p class="sub">{close.get("episodes", 0)} episodes · {len(close.get("tasks", []))} instances ·
scoring <code>sh-scoring-v2</code> · era {html.escape(str(close.get("era", "")))}</p>

<div class="wrap"><table>
<thead><tr><th>hotkey</th><th>#</th><th>episodes</th><th>mean d</th><th>&Delta;c</th>
<th>gate</th><th>overfit</th><th>dq</th><th>weight</th></tr></thead>
<tbody>{"".join(body) or '<tr><td colspan="9" class="zero">no miners in this round</td></tr>'}</tbody>
</table></div>
<p class="note"><strong>mean d</strong> is the miner's pass rate minus the family's baseline rate on the same
instances. <strong>&Delta;c</strong> is its one-sided 90&nbsp;% lower bound, which is what pays: beating the
baseline on average is not enough to be paid for beating it. A miner at the baseline earns nothing.</p>

<section>
<h2>Families this round</h2>
<div class="wrap"><table>
<thead><tr><th>family</th><th>null n</th><th>null p</th><th>canon p</th><th>&Delta;c canon</th><th>label</th></tr></thead>
<tbody>{families}</tbody>
</table></div>
</section>

<section>
<h2>Withheld halves</h2>
<p class="note">{len(checked)} of {len(verified)} instances had a sealed withheld half.
Commitments re-verified at close: <span class="{"ok" if ok else "why"}">{"all match" if ok else "MISMATCH"}</span>.
The revealed halves and their salts are published in <code>reveal.json</code>, so anyone can recompute
<code>HMAC(salt, withheld)</code> and confirm the grading criteria were fixed before submissions opened.</p>
</section>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--close", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    page = render(json.loads(Path(a.close).read_text()))
    (out / "index.html").write_text(page)
    print(f"{out / 'index.html'} ({len(page)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
