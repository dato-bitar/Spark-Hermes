def test_truncation_is_reported_beside_the_success_rate():
    """`max_steps_hit` was on every episode and aggregated nowhere.

    So the headline reported `success_rate` and `mean_tokens` with no sign that most of the run was
    censored. Measured on two real 19-task runs: 53% and 56% of episodes ended with the harness
    cutting in, and in BOTH runs every single failing episode ended that way -- which makes
    `1 - success_rate` a mix of "could not" and "ran out of budget".
    """
    from hermesbench.metrics import EpisodeMetrics, suite_metrics

    def episode(task_id: str, *, success: bool, capped: bool) -> EpisodeMetrics:
        return EpisodeMetrics(
            task_id=task_id,
            success=success,
            tool_calls=10,
            failed_calls=0,
            hit_failure=False,
            recovered=False,
            mutated=True,
            self_checked=True,
            tokens_used=1000,
            wall_time_s=1.0,
            steps=30,
            max_steps_hit=capped,
            public_passed=success,
        )

    record = suite_metrics(
        [
            episode("a", success=True, capped=False),
            episode("b", success=True, capped=True),
            episode("c", success=False, capped=True),
            episode("d", success=False, capped=False),
        ]
    ).to_record()
    assert record["truncated_episodes"] == 2, "both capped episodes, whether they passed or not"
    assert record["truncated_failures"] == 1, "only the one that was capped AND failed"
    assert record["success_rate"] == 0.5


def test_a_run_that_was_never_truncated_says_zero_rather_than_omitting_it():
    """The field has to be present at zero. An absent key and a zero are the same thing to a reader
    who does not know the key exists, which is how this went unnoticed in the first place."""
    from hermesbench.metrics import EpisodeMetrics, suite_metrics

    clean = EpisodeMetrics(
        task_id="a",
        success=True,
        tool_calls=3,
        failed_calls=0,
        hit_failure=False,
        recovered=False,
        mutated=True,
        self_checked=True,
        tokens_used=10,
        wall_time_s=1.0,
        steps=8,
        max_steps_hit=False,
        public_passed=True,
    )
    record = suite_metrics([clean]).to_record()
    assert record["truncated_episodes"] == 0
    assert record["truncated_failures"] == 0
