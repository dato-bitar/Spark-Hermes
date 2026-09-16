"""FamilyStats: the numbers a family is judged on, and the rules that take it out of rotation."""

from __future__ import annotations

import json

from sh.validator.stats import MIN_POOLED, family_stats, load_episodes


def _eps(surface: str, n: int, wins: int, *, family="posix_report", rnd="r1", api=7, **extra) -> list[dict]:
    return [
        {
            "family": family,
            "round_id": rnd,
            "surface": surface,
            "verified_success": i < wins,
            "api_calls": api,
            "tool_calls": api - 1,
            **extra,
        }
        for i in range(n)
    ]


def test_the_arms_pool_over_the_window_and_canon_is_the_yardstick():
    st = family_stats(
        _eps("null", 8, 3, api=9) + _eps("canon", 8, 6, api=6), "posix_report", ["r1"], "e0", {"api_calls": 6}
    )
    r = st.record()
    assert r["null"]["p"] == 0.375 and r["canon"]["delta_c"] == 0.375
    assert r["label"] == "frontier" and r["in_rotation"]
    assert r["canon"]["delta_e"]["api_calls"] > 0  # CANON solved it in fewer calls


def test_statistics_never_span_eras():
    """A promotion is a different baseline, so its episodes describe a different difficulty (FR-BL-6)."""
    old = _eps("null", 8, 8, era="e0")
    new = _eps("null", 8, 1, era="e1")
    assert family_stats(old + new, "posix_report", ["r1"], "e1").null.successes == 1


def test_a_window_without_enough_evidence_is_labelled_unknown():
    st = family_stats(_eps("null", MIN_POOLED - 1, 0), "posix_report", ["r1"], "e0")
    assert st.label() == "unknown" and st.retirement() is None


def test_a_family_the_baseline_always_solves_cheaply_is_retired():
    st = family_stats(
        _eps("null", 16, 16, api=6) + _eps("canon", 16, 16, api=6), "posix_report", ["r1"], "e0", {"api_calls": 6}
    )
    assert st.label() == "trivial"
    assert "no headroom" in st.retirement()


def test_a_trivial_family_that_still_costs_the_baseline_stays():
    """Trivial by pass rate is not trivial by cost: 12 calls against a 6-call reference is still headroom."""
    st = family_stats(
        _eps("null", 16, 16, api=12) + _eps("canon", 16, 16, api=9), "posix_report", ["r1"], "e0", {"api_calls": 6}
    )
    assert st.label() == "trivial" and st.retirement() is None


def test_a_family_nothing_passes_is_retired():
    st = family_stats(_eps("null", 16, 0) + _eps("canon", 16, 0), "posix_report", ["r1"], "e0")
    assert st.label() == "impossible" and "nothing is being measured" in st.retirement()


def test_a_hard_family_is_not_called_impossible_on_thin_evidence():
    """0/8 is a hard family, not a broken one — it keeps running until the window says otherwise."""
    st = family_stats(_eps("null", 8, 0) + _eps("canon", 8, 1), "posix_report", ["r1"], "e0")
    assert st.label() == "frontier"


def test_a_harness_fault_is_reported_as_one_and_takes_precedence():
    """Half the reference episodes falling over is a template bug, not a hard family (FR-BL-3)."""
    st = family_stats(_eps("null", 8, 0, partial=True) + _eps("canon", 8, 0), "posix_report", ["r1"], "e0")
    assert st.partial_rate == 0.5 and "harness or template fault" in st.retirement()


def test_stats_recompute_identically_from_the_archive(tmp_path):
    """The D1 exit: the archive is the source of truth, not any in-memory accumulator."""
    eps = _eps("null", 8, 3, api=9) + _eps("canon", 8, 6, api=6)
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in eps))
    live = family_stats(eps, "posix_report", ["r1"], "e0", {"api_calls": 6}).record()
    archived = family_stats(
        load_episodes(tmp_path / "episodes.jsonl"), "posix_report", ["r1"], "e0", {"api_calls": 6}
    ).record()
    assert live == archived


def test_the_wilson_interval_agrees_with_the_one_it_replaces():
    """`sh/` is standalone — this keeps the inlined copy honest against the implementation it came from."""
    from hermesbench.repeats import wilson as legacy
    from sh.validator.stats import wilson

    for n in (1, 8, 16, 100):
        for k in (0, 1, n // 2, n):
            low, high = wilson(k, n)
            ref = legacy(k, n)
            assert abs(low - max(0.0, ref.low)) < 1e-9 and abs(high - min(1.0, ref.high)) < 1e-9, (k, n)


def _paired(null_calls: dict[str, int], canon_calls: dict[str, int]) -> list[dict]:
    """(task_id -> api_calls) per arm; a task absent from an arm was not solved by it."""
    eps = []
    for surface, calls in (("null", null_calls), ("canon", canon_calls)):
        for task_id, n in calls.items():
            eps.append(
                {
                    "family": "f",
                    "round_id": "r1",
                    "surface": surface,
                    "task_id": task_id,
                    "verified_success": True,
                    "api_calls": n,
                    "tool_calls": n,
                }
            )
    return eps


def test_efficiency_is_compared_only_on_instances_both_arms_solved():
    """The family #2 screen: CANON rescued a hard instance NULL never solved and paid for it in calls. Comparing
    each arm's median over its own successes reported CANON as 43 % less efficient while it was strictly better."""
    eps = _paired(
        null_calls={"t1": 6, "t2": 6},  # null solved the two easy ones cheaply
        canon_calls={"t1": 5, "t2": 5, "t3": 14},
    )  # canon solved those cheaper, plus a hard one
    st = family_stats(eps, "f", ["r1"], "e0", {"api_calls": 6, "tool_calls": 5})
    assert st.delta_e()["api_calls"] > 0  # 12 -> 10 on the shared instances
    assert st.paired_instances() == 2


def test_a_genuine_efficiency_regression_still_shows():
    eps = _paired(null_calls={"t1": 5, "t2": 5}, canon_calls={"t1": 9, "t2": 9})
    st = family_stats(eps, "f", ["r1"], "e0", {"api_calls": 6, "tool_calls": 5})
    assert st.delta_e()["api_calls"] < 0


def test_efficiency_is_silent_when_no_instance_is_shared():
    eps = _paired(null_calls={"t1": 5}, canon_calls={"t2": 5})
    st = family_stats(eps, "f", ["r1"], "e0", {"api_calls": 6, "tool_calls": 5})
    assert st.delta_e() == {} and st.paired_instances() == 0


def test_repeats_of_one_instance_average_before_pairing():
    """Eight repeats of an easy instance must not outvote one repeat of a hard one."""
    eps = _paired(null_calls={}, canon_calls={})
    for i in range(8):
        eps.append(
            {
                "family": "f",
                "round_id": "r1",
                "surface": "null",
                "task_id": "t1",
                "verified_success": True,
                "api_calls": 10,
                "tool_calls": 10,
            }
        )
    eps.append(
        {
            "family": "f",
            "round_id": "r1",
            "surface": "canon",
            "task_id": "t1",
            "verified_success": True,
            "api_calls": 5,
            "tool_calls": 5,
        }
    )
    st = family_stats(eps, "f", ["r1"], "e0", {"api_calls": 6, "tool_calls": 5})
    assert st.delta_e()["api_calls"] == 0.5
