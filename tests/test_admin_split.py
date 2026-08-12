"""The one guard in this pipeline that cannot be undone downstream.

Once an evaluation task's rollouts are in the corpus, every later number measured on that task is a
number about the training set, and nothing downstream can recover the distinction. `overfit_rate` --
the instrument that would normally notice -- is measured on those same tasks, so training on them
blinds the check that would report the problem.

Every test here is about refusing.
"""

from __future__ import annotations

import pytest

from admin.split import Contamination, SplitError, build, contamination, eval_task_ids, refuse_eval_tasks


def test_the_real_suite_is_what_gets_protected():
    """Read from disk, not from a list. A task added to the suite is an eval task from the moment it
    exists, and a hard-coded list would silently make the twentieth one trainable."""
    ids = eval_task_ids()
    assert len(ids) == 19
    assert "tc-log-rotation-order" in ids
    assert "recover-from-bad-command" in ids


def test_an_eval_task_in_the_corpus_is_refused_and_named():
    with pytest.raises(SplitError, match="tc-log-rotation-order"):
        refuse_eval_tasks(["gen-0001", "tc-log-rotation-order", "gen-0002"])


def test_the_refusal_says_why_it_matters_rather_than_only_that_it_happened():
    """An operator who reads "refused" and not "this blinds overfit_rate" removes the guard."""
    with pytest.raises(SplitError, match="blinds the instrument"):
        refuse_eval_tasks(["fix-failing-test"])


def test_a_clean_training_set_passes():
    refuse_eval_tasks(["gen-0001", "gen-0002"])


def test_contamination_is_named_not_scored():
    """A similarity number invites a threshold argument. A pair of ids and the shared run is
    something an operator can open and judge in ten seconds."""
    found = contamination(
        {"gen-1": "Write report.txt with the total number of lines across every .log file in logs/"},
        {"recover": "Produce a file named report.txt containing the total number of lines across every .log file"},
    )
    assert len(found) == 1
    assert isinstance(found[0], Contamination)
    assert found[0].train_task == "gen-1" and found[0].eval_task == "recover"
    assert "total number of lines across every" in found[0].shared


def test_unrelated_tasks_are_not_flagged():
    """The check has to be usable. Flagging every pair of engineering prompts would make it noise
    that gets switched off, which is worse than not having it."""
    assert (
        contamination(
            {"gen-1": "Count the distinct hostnames in access.log and write them to hosts.txt"},
            {"other": "Repair the failing CMake target so the build completes"},
        )
        == []
    )


def test_build_refuses_before_it_reports():
    """The split is computed only if it is legal. Returning a Split with a note about contamination
    would let a caller ignore the note."""
    with pytest.raises(SplitError):
        build(train_prompts={"verify-speedup-claim": "anything"})


def test_build_without_eval_prompts_says_it_checked_nothing():
    """Comparing prompts costs a parse the caller may not have done. The limitation is real, so an
    empty `contaminated` on a run that never compared must not read as 'no contamination found'."""
    split = build(train_prompts={"gen-1": "Count the lines"})
    assert split.contaminated == []
    assert len(split.eval_tasks) == 19
    assert split.train_tasks == ("gen-1",)
