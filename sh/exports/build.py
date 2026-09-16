"""Training data out of a closed round (spec §8).

The competition exists to produce this. Every row is an episode that a **withheld** half verified, so the data
says "this trajectory actually solved the task", not "this trajectory looked right".

  * **SFT** — one row per verified, non-disqualified episode, in the converter's `{from, value}` shape.
  * **DPO** — a chosen/rejected pair per instance, where the same task was solved by one surface and failed by
    another. The pair is only meaningful within one instance: across instances it would encode difficulty.

Two rules from V4 govern the system turn, and they matter more than they look. The converter emits Hermes'
*generic* function-calling prompt, which is not what the agent actually ran under — so the export replaces it
with the NULL-surface prompt captured at the pin, and carries tool schemas in the row instead. Training on the
converter's own system turn would teach the model a prompt no deployment ever shows it.

    python -m sh.exports.build --round DIR --episodes DIR --close FILE --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA_SFT = "sh-sft-v2"
SCHEMA_DPO = "sh-dpo-v2"
RESERVED = ("null", "canon")


def _leak_scan(text: str, secrets: set[str]) -> list[str]:
    """Nothing withheld may leave in a training row: the withheld half is what makes future rounds gradeable."""
    return sorted({s for s in secrets if s and s in text})


def _rows_for(episode_dir: Path, episode: dict, task: dict, system_prompt: str) -> dict | None:
    trajectory = episode_dir / "trajectory.json"
    if not trajectory.exists():
        return None
    turns = json.loads(trajectory.read_text())
    if not isinstance(turns, list) or not turns:
        return None
    # Rule 1 (V4): the converter's system turn is Hermes' generic function-calling prompt, not the prompt the
    # episode ran under. Replace it; keep everything else the converter produced.
    body = [t for t in turns if t.get("from") != "system"]
    return {
        "schema": SCHEMA_SFT,
        "task_id": episode.get("task_id"),
        "family": episode.get("family"),
        "round_id": episode.get("round_id"),
        "surface": episode.get("surface"),
        "conversations": [{"from": "system", "value": system_prompt}, *body],
        "tools": task.get("tools", []),  # Rule 2 (V4): schemas travel as a field, not as prose
        "verified_success": bool(episode.get("verified_success")),
        "api_calls": episode.get("api_calls"),
        "tool_calls": episode.get("tool_calls"),
        "self_checked": episode.get("self_checked"),
        "pins": episode.get("pins"),
    }


def build(
    round_dir: Path, episodes_dir: Path, close_file: Path, out: Path, *, system_prompt: str | None = None
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tasks = {p.stem: json.loads(p.read_text()) for p in sorted((round_dir / "tasks").glob("*.json"))}
    closed = json.loads(close_file.read_text())

    # Everything withheld, as strings, so a row carrying any of it can be refused rather than uploaded.
    secrets: set[str] = set()
    reveal = close_file.parent / "reveal.json"
    if reveal.exists():
        for entry in json.loads(reveal.read_text()).values():
            secrets.add(entry.get("salt", ""))
            for predicate in entry.get("withheld", {}).get("predicates", []):
                secrets.update(str(a) for a in predicate[1:] if isinstance(a, str) and len(str(a)) >= 16)

    sft, by_task, gates = [], {}, {"no_trajectory": 0, "not_verified": 0, "disqualified": 0, "leaked": 0}
    for episode_json in sorted(episodes_dir.rglob("episode.json")):
        episode = json.loads(episode_json.read_text())
        task = tasks.get(str(episode.get("task_id")), {})
        by_task.setdefault(episode.get("task_id"), []).append((episode, episode_json.parent))
        if episode.get("disqualified"):
            gates["disqualified"] += 1
            continue
        if not episode.get("verified_success"):
            gates["not_verified"] += 1
            continue
        row = _rows_for(
            episode_json.parent,
            episode,
            task,
            system_prompt or "You are Hermes, an agent operating a terminal and a filesystem.",
        )
        if row is None:
            gates["no_trajectory"] += 1
            continue
        if _leak_scan(json.dumps(row), secrets):
            gates["leaked"] += 1
            continue
        sft.append(row)

    # DPO: same instance, one surface solved it and another did not. Across instances a pair would encode
    # difficulty rather than strategy, so pairs never cross a task_id.
    dpo = []
    for task_id, entries in by_task.items():
        task = tasks.get(str(task_id), {})
        prompt = system_prompt or ""
        # Take the first side of each pair that actually yields a row. An episode killed on its timeout has no
        # trajectory at all, and picking only the first loser silently dropped every pair whose first loser
        # happened to be one of those — half the training value of the round, lost to list order.
        chosen = next(
            (
                row
                for e, d in entries
                if e.get("verified_success") and not e.get("disqualified") and (row := _rows_for(d, e, task, prompt))
            ),
            None,
        )
        rejected = next(
            (
                row
                for e, d in entries
                if not e.get("verified_success") and not e.get("void") and (row := _rows_for(d, e, task, prompt))
            ),
            None,
        )
        if chosen and rejected:
            dpo.append(
                {
                    "schema": SCHEMA_DPO,
                    "task_id": task_id,
                    "family": chosen["family"],
                    "round_id": chosen["round_id"],
                    "prompt": task.get("prompt"),
                    "chosen": chosen["conversations"],
                    "rejected": rejected["conversations"],
                    "chosen_surface": chosen["surface"],
                    "rejected_surface": rejected["surface"],
                }
            )

    (out / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sft))
    (out / "dpo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dpo))
    manifest = {
        "schema": "sh-export-manifest-v2",
        "round_id": closed.get("round_id"),
        "sft_rows": len(sft),
        "dpo_pairs": len(dpo),
        "gates": gates,
        "families": sorted({r["family"] for r in sft if r.get("family")}),
        "sft_sha256": hashlib.sha256((out / "sft.jsonl").read_bytes()).hexdigest(),
        "dpo_sha256": hashlib.sha256((out / "dpo.jsonl").read_bytes()).hexdigest(),
        "leak_scan": {"secrets_checked": len(secrets), "rows_refused": gates["leaked"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--close", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system-prompt")
    a = ap.parse_args(argv)
    prompt = Path(a.system_prompt).read_text() if a.system_prompt else None
    print(
        json.dumps(build(Path(a.round), Path(a.episodes), Path(a.close), Path(a.out), system_prompt=prompt), indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
