"""`sh-scoring-v2` — what a miner is paid for (spec §7).

A miner is paid for **beating the baseline on the same instances**, not for passing tasks. Everything here is a
pure function of archived Episode records and the family statistics computed from the reference arms, so
`sh scoring recompute` reproduces a validator's Score records byte for byte.

Three things carry most of the design:

  * **The reference rate is an estimate, not a constant.** Every episode of a family is compared against the same
    measured NULL rate, so that estimate's error does not average away with √n. It is added to the standard
    error explicitly, weighted by each family's share of the window. On the spec's worked example it is as large
    as the miner's own term.
  * **Efficiency is gated on correctness and never pays for cheaper failures.** It is measured only on verified
    successes, against the family's *measured* NULL median — never against `efficiency_reference`, which is an
    author's guess and would otherwise be a reward the author sets.
  * **Nothing is paid on thin evidence.** Below 8 window episodes a miner scores 0; below 4 NULL successes a
    family contributes no efficiency term; a one-sided 90 % lower bound is taken on every mean.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

__all__ = ["PARAMS_V2", "Params", "score", "stat", "weights"]


@dataclass(frozen=True)
class Params:
    z: float = 1.28  # one-sided 90 %
    window: int = 8  # rounds
    min_episodes: int = 8  # fewer than this in the window pays nothing
    min_null_successes: int = 4  # fewer than this and the family's efficiency reference is not a measurement
    min_metric_samples: int = 4
    metrics: tuple[str, ...] = ("api_calls", "tool_calls")
    w_c: float = 0.8
    w_e: float = 0.2
    w_m: dict = field(default_factory=lambda: {"api_calls": 0.5, "tool_calls": 0.5})
    overfit_cutoff: float = 0.25
    copy_penalty: float = 0.5
    bootstrap: int = 200  # resamples for the NULL-median variance


PARAMS_V2 = Params()


@dataclass(frozen=True)
class FamilyReference:
    """What the NULL arm measured for one family over the window — the denominator a miner is judged against."""

    family: str
    n: int
    successes: int
    medians: dict  # metric -> median over NULL verified successes
    samples: dict  # metric -> the values behind that median, for the bootstrap
    requires_self_check: bool = False

    @property
    def p(self) -> float:
        return self.successes / self.n if self.n else 0.0


def stat(episode: dict, reference: FamilyReference, params: Params = PARAMS_V2) -> tuple[float, dict]:
    """One episode's contribution: `d` against the family's NULL rate, and log-ratios per efficiency metric.

    The log ratio is symmetric in the sense that matters here — halving the calls and doubling them are equal and
    opposite — so a single outlier cannot dominate the mean the way a raw ratio would.
    """
    won = bool(episode.get("verified_success"))
    d = (1.0 if won else 0.0) - reference.p
    ratios: dict = {}
    if not won or reference.successes < params.min_null_successes:
        return d, ratios
    for metric in params.metrics:
        ref = reference.medians.get(metric)
        got = episode.get(metric)
        if ref and isinstance(got, (int, float)) and got > 0:
            ratios[metric] = math.log(ref / got)
    # A family that asks for a self-check pays nothing for efficiency without one: spending fewer calls by
    # skipping the verification the task requires is not efficiency.
    if reference.requires_self_check and not episode.get("self_checked"):
        ratios = {m: 0.0 for m in ratios}
    return d, ratios


def _median_variance(samples: list[float], resamples: int, rng: random.Random) -> float:
    """Bootstrap variance of a median. The NULL median is itself an estimate from a handful of episodes."""
    if len(samples) < 2:
        return 0.0
    medians = [statistics.median(rng.choices(samples, k=len(samples))) for _ in range(resamples)]
    return statistics.variance(medians) if len(set(medians)) > 1 else 0.0


@dataclass
class MinerWindow:
    """Everything about one hotkey over the scoring window."""

    hotkey: str
    episodes: list[dict] = field(default_factory=list)
    near_dup: bool = False
    prior_copy: float = 0.0

    @property
    def public_passers(self) -> int:
        return sum(1 for e in self.episodes if e.get("published_pass"))

    @property
    def overfit(self) -> int:
        return sum(1 for e in self.episodes if e.get("overfit"))

    @property
    def dq(self) -> int:
        return sum(1 for e in self.episodes if e.get("disqualified"))


def score(miner: MinerWindow, references: dict, params: Params = PARAMS_V2, *, seed: int = 0) -> dict:
    """The miner's weight contribution, with every term that produced it.

    Returns a record rather than a bare float so a validator's arithmetic can be audited without re-running it.
    """
    rng = random.Random(seed)
    eps = [e for e in miner.episodes if not e.get("void")]
    detail = {
        "hotkey": miner.hotkey,
        "n": len(eps),
        "score": 0.0,
        "delta_c": 0.0,
        "delta_e": {},
        "gate": False,
        "se": None,
        "overfit_rate": 0.0,
        "dq": miner.dq,
        "reason": None,
    }
    if len(eps) < params.min_episodes:
        detail["reason"] = f"{len(eps)} window episodes < {params.min_episodes}"
        return detail

    pairs = [(e, references[e["family"]]) for e in eps if e.get("family") in references]
    if not pairs:
        detail["reason"] = "no episode belongs to a family with reference statistics"
        return detail
    ds, rs = zip(*(stat(e, ref, params) for e, ref in pairs))
    n = len(ds)

    # The family NULL rate is shared by every episode of that family, so its error is not reduced by √n.
    ref_var = 0.0
    for family in {ref.family for _, ref in pairs}:
        ref = next(r for _, r in pairs if r.family == family)
        share = sum(1 for _, r in pairs if r.family == family) / n
        if ref.n:
            ref_var += share**2 * ref.p * (1 - ref.p) / ref.n
    se = math.sqrt((statistics.variance(ds) / n if n > 1 else 0.0) + ref_var)
    mean_d = statistics.mean(ds)
    detail["se"] = round(se, 6)
    detail["mean_d"] = round(mean_d, 6)
    detail["delta_c"] = max(0.0, mean_d - params.z * se)
    detail["gate"] = mean_d + params.z * se >= 0.0

    if detail["gate"]:
        for metric in params.metrics:
            values = [r[metric] for r in rs if metric in r]
            if len(values) < params.min_metric_samples:
                continue
            ref_median_var = 0.0
            for family in {ref.family for _, ref in pairs}:
                ref = next(r for _, r in pairs if r.family == family)
                share = sum(1 for _, r in pairs if r.family == family) / n
                samples = [float(v) for v in ref.samples.get(metric, [])]
                median = ref.medians.get(metric)
                if samples and median:
                    # variance of log(median) ≈ var(median) / median²
                    ref_median_var += share**2 * _median_variance(samples, params.bootstrap, rng) / (median**2)
            se_m = math.sqrt((statistics.variance(values) / len(values) if len(values) > 1 else 0.0) + ref_median_var)
            detail["delta_e"][metric] = max(-1.0, min(1.0, statistics.mean(values) - params.z * se_m))

    raw = params.w_c * detail["delta_c"] + params.w_e * sum(
        params.w_m.get(m, 0.0) * x for m, x in detail["delta_e"].items()
    )

    ofr = miner.overfit / max(1, miner.public_passers)
    detail["overfit_rate"] = round(ofr, 4)
    if miner.public_passers >= params.min_episodes and ofr > params.overfit_cutoff:
        detail["reason"] = f"overfit rate {ofr:.2f} > {params.overfit_cutoff}"
        return detail
    if miner.dq >= 2:
        detail["reason"] = f"{miner.dq} disqualified episodes"
        return detail
    if miner.near_dup:
        detail["reason"] = "bundle is a near-duplicate"
        return detail

    detail["score"] = max(0.0, raw * (1 - ofr) ** 2 * (1 - params.copy_penalty * miner.prior_copy))
    return detail


def weights(scores: dict) -> dict:
    """Normalise to a weight vector. NULL and CANON are never in `scores` — they are references, not competitors."""
    total = sum(max(0.0, v) for v in scores.values())
    if total <= 0:
        return {h: 0.0 for h in scores}
    return {h: max(0.0, v) / total for h, v in scores.items()}


def references_from_stats(family_stats: dict, requires_self_check: dict | None = None) -> dict:
    """Build the scoring references from `FamilyStats` records, keyed by family."""
    out = {}
    for family, record in family_stats.items():
        null = record["null"]
        medians = {m: v["median"] for m, v in (null.get("efficiency") or {}).items()}
        out[family] = FamilyReference(
            family=family,
            n=null["n"],
            successes=null["successes"],
            medians=medians,
            samples=(record.get("null_samples") or {}),
            requires_self_check=bool((requires_self_check or {}).get(family)),
        )
    return out
