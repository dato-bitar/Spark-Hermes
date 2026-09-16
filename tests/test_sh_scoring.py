"""What a miner is paid for. Each test pins one rule the scorer exists to enforce."""

from __future__ import annotations

from sh.scoring.v2 import FamilyReference, MinerWindow, score, stat, weights

REF = FamilyReference(
    family="f",
    n=16,
    successes=8,
    medians={"api_calls": 10, "tool_calls": 10},
    samples={"api_calls": [9, 10, 10, 11, 12, 8, 10, 10], "tool_calls": [9, 10, 10, 11, 12, 8, 10, 10]},
)


def _eps(n, wins, *, calls=10, family="f", **extra):
    return [
        {
            "family": family,
            "verified_success": i < wins,
            "published_pass": i < wins,
            "api_calls": calls,
            "tool_calls": calls,
            "self_checked": True,
            "overfit": False,
            "disqualified": False,
            **extra,
        }
        for i in range(n)
    ]


def test_a_miner_at_the_baseline_rate_is_paid_nothing():
    """The whole design: passing tasks is not the achievement, beating the reference is."""
    s = score(MinerWindow("m", _eps(16, 8)), {"f": REF})
    assert s["score"] == 0.0 and s["delta_c"] == 0.0


def test_a_miner_clearly_above_the_baseline_is_paid():
    s = score(MinerWindow("m", _eps(16, 15)), {"f": REF})
    assert s["score"] > 0 and s["delta_c"] > 0 and s["gate"]


def test_a_thin_window_pays_nothing_however_good_it_looks():
    s = score(MinerWindow("m", _eps(4, 4)), {"f": REF})
    assert s["score"] == 0.0 and "window episodes" in s["reason"]


def test_the_reference_error_is_carried_into_the_standard_error():
    """The spec's central statistical point: the family rate is shared by every episode, so its error does not
    average away. A reference measured on few episodes must pay less than the same result on many."""
    thin = FamilyReference(family="f", n=8, successes=4, medians=REF.medians, samples=REF.samples)
    fat = FamilyReference(family="f", n=200, successes=100, medians=REF.medians, samples=REF.samples)
    on_thin = score(MinerWindow("m", _eps(16, 14)), {"f": thin})
    on_fat = score(MinerWindow("m", _eps(16, 14)), {"f": fat})
    assert on_thin["se"] > on_fat["se"]
    assert on_thin["delta_c"] < on_fat["delta_c"]


def test_efficiency_needs_the_correctness_gate():
    """Spending fewer calls while failing more often is not efficiency."""
    losing = score(MinerWindow("m", _eps(16, 1, calls=2)), {"f": REF})
    assert not losing["gate"] and losing["delta_e"] == {}


def test_efficiency_is_measured_against_a_measurement_not_an_author_s_guess():
    """With fewer than 4 NULL successes the family's median is not a measurement, so no efficiency is paid."""
    unmeasured = FamilyReference(family="f", n=16, successes=2, medians={"api_calls": 10}, samples={})
    d, ratios = stat({"family": "f", "verified_success": True, "api_calls": 5}, unmeasured)
    assert ratios == {}


def test_skipping_a_required_self_check_earns_no_efficiency():
    """The cheapest way to spend fewer calls is to skip the verification the family asks for."""
    checking = FamilyReference(**{**REF.__dict__, "requires_self_check": True})
    _, with_check = stat({"family": "f", "verified_success": True, "api_calls": 5, "self_checked": True}, checking)
    _, without = stat({"family": "f", "verified_success": True, "api_calls": 5, "self_checked": False}, checking)
    assert with_check["api_calls"] > 0
    assert without["api_calls"] == 0.0


def test_an_overfitting_miner_is_zeroed():
    eps = _eps(16, 16)
    for e in eps[:8]:
        e["overfit"] = True
    s = score(MinerWindow("m", eps), {"f": REF})
    assert s["score"] == 0.0 and "overfit rate" in s["reason"]


def test_two_disqualifications_zero_a_miner():
    eps = _eps(16, 16)
    eps[0]["disqualified"] = eps[1]["disqualified"] = True
    s = score(MinerWindow("m", eps), {"f": REF})
    assert s["score"] == 0.0 and "disqualified" in s["reason"]


def test_a_near_duplicate_bundle_is_zeroed():
    s = score(MinerWindow("m", _eps(16, 15), near_dup=True), {"f": REF})
    assert s["score"] == 0.0 and "near-duplicate" in s["reason"]


def test_void_episodes_are_not_evidence():
    """A provider outage must neither help nor hurt."""
    eps = _eps(16, 15) + _eps(4, 0, void=True)
    s = score(MinerWindow("m", eps), {"f": REF})
    assert s["n"] == 16


def test_scoring_is_reproducible():
    """`sh scoring recompute` has to reproduce a validator's record byte for byte."""
    a = score(MinerWindow("m", _eps(16, 14, calls=6)), {"f": REF}, seed=7)
    b = score(MinerWindow("m", _eps(16, 14, calls=6)), {"f": REF}, seed=7)
    assert a == b


def test_weights_normalise_and_ignore_the_references():
    w = weights({"a": 0.3, "b": 0.1, "c": 0.0})
    assert abs(sum(w.values()) - 1.0) < 1e-9 and w["c"] == 0.0
    assert weights({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}  # an all-zero round pays no one


def test_the_worked_example_from_the_spec_reproduces():
    """+0.25 over a NULL rate of 0.4, 64 episodes across 4 families with 16 NULL episodes each."""
    refs, eps = {}, []
    for i in range(4):
        fam = f"f{i}"
        refs[fam] = FamilyReference(family=fam, n=16, successes=6, medians={}, samples={})
        eps += _eps(16, 10, family=fam)  # 10/16 = 0.625 vs 0.375 → d ≈ +0.25
    s = score(MinerWindow("m", eps), refs)
    assert 0.2 < s["mean_d"] < 0.3
    assert 0.05 < s["se"] < 0.12  # the spec's 0.083, to the nearest band
    assert 0.10 < s["delta_c"] < 0.20  # the spec's 0.144
