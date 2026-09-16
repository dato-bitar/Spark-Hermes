"""C1 contract: a predicate argument is compared, never interpolated; every operator has an
evaluator; a bad predicate is refused before any workspace exists; `judge` names the first failure."""

from __future__ import annotations

import hashlib
import os

import pytest

from sh.predicates import OPS, PredicateError, evaluate, judge, parse, parse_all

HOSTILE = [
    "it's",
    'a"b',
    "$(touch pwned)",
    "`touch pwned2`",
    "a\\b",
    "*",
    "; rm -rf .",
    "--",
    "$PATH",
    "a'; touch pwned3; echo '",
    "${HOME}x",
    "/etc/passwd",
]


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out/r.txt").write_bytes(b"alpha\nbeta\ngamma\n")
    (tmp_path / "out/n.txt").write_text("3.141\n")
    (tmp_path / "out/d.json").write_text('{"a": {"b": 7, "ok": true, "xs": [1, 2]}}')
    os.chmod(tmp_path / "out/r.txt", 0o644)
    return tmp_path


def test_every_operator_in_the_table_has_an_evaluator(ws):
    sample = {
        "file_exists": ["out/r.txt"],
        "file_absent": ["nope"],
        "digest_is": ["out/r.txt", hashlib.sha256(b"alpha\nbeta\ngamma\n").hexdigest()],
        "line_count_is": ["out/r.txt", "3"],
        "text_equals": ["out/n.txt", "3.141"],
        "number_is": ["out/n.txt", "3.14", "0.01"],
        "grep_count": ["out/r.txt", "a", "3"],
        "ordering_is": ["out/r.txt", "alpha", "gamma"],
        "json_field_equals": ["out/d.json", "a.b", "7"],
        "perms_are": ["out/r.txt", "644"],
        "exit_code_is": [0, "0"],
        "stdout_equals": [1, "hello"],
        "custom": ["always", "x"],
    }
    assert set(sample) == set(OPS)
    bits = evaluate(
        [[op, *args] for op, args in sample.items()],
        ws,
        commands=["true", "printf hello"],
        checks={"always": lambda w, *a: True},
    )
    assert all(bits), dict(zip(sample, bits))


@pytest.mark.parametrize("value", HOSTILE)
def test_a_hostile_argument_is_data_and_never_code(value, ws):
    if value.startswith("/") or "${HOME}" in value:
        # paths must be relative; template-looking text is fine as a literal
        if value.startswith("/"):
            with pytest.raises(PredicateError):
                parse(["file_exists", value])
            return
    bits = evaluate([["grep_count", "out/r.txt", value, "0"]], ws)
    assert bits == [True]
    assert sorted(p.name for p in ws.iterdir()) == ["out"]  # nothing executed, nothing created


def test_judge_stops_at_the_first_false_predicate_and_names_it(ws):
    v = judge([["file_exists", "out/r.txt"], ["line_count_is", "out/r.txt", "99"], ["file_exists", "zzz"]], ws)
    assert not v.ok and v.failed_index == 1 and v.message == "out/r.txt does not have 99 lines"
    assert v.bits == (True, False)  # the third predicate was never evaluated


def test_line_count_counts_a_final_unterminated_line(ws):
    (ws / "u.txt").write_bytes(b"one\ntwo")
    assert evaluate([["line_count_is", "u.txt", "2"]], ws) == [True]


def test_json_scalars_compare_as_json_text(ws):
    assert evaluate(
        [
            ["json_field_equals", "out/d.json", "a.ok", "true"],
            ["json_field_equals", "out/d.json", "a.xs.1", "2"],
            ["json_field_equals", "out/d.json", "a.missing", "1"],
        ],
        ws,
    ) == [True, True, False]


def test_a_symlink_that_escapes_the_workspace_is_treated_as_missing(ws):
    (ws / "leak").symlink_to("/etc/hostname")
    assert evaluate([["file_exists", "leak"], ["digest_is", "leak", "0" * 64]], ws) == [False, False]


def test_commands_are_reached_by_index_only_and_run_without_stdin(ws):
    assert evaluate([["exit_code_is", 0, "0"]], ws, commands=["cat"], timeout_s=10) == [True]  # cat on DEVNULL exits 0
    with pytest.raises(PredicateError, match="only 1 are declared"):
        evaluate([["exit_code_is", 3, "0"]], ws, commands=["true"])


def test_custom_dispatches_to_the_family_check_and_a_raising_check_is_false(ws):
    seen = {}

    def rec(w, *a):
        seen["args"] = a

    return True

    def boom(w, *a):
        raise RuntimeError("no")

    assert evaluate([["custom", "rec", "x", "y"], ["custom", "boom"]], ws, checks={"rec": rec, "boom": boom}) == [
        True,
        False,
    ]
    assert seen["args"] == ("x", "y")
    with pytest.raises(PredicateError, match="does not declare"):
        parse_all([["custom", "nope"]], checks=frozenset())


@pytest.mark.parametrize(
    "bad,msg",
    [
        (["digest_is", "f", "NOTHEX"], "sha256"),
        (["line_count_is", "f", "3.5"], "integer"),
        (["file_exists", "../x"], "climbs"),
        (["grep_count", "f", "a\nb", "1"], "single line"),
        (["nope", "f"], "unknown operator"),
        (["file_exists", "a", "b"], "takes 1"),
        (["perms_are", "f", "999"], "octal"),
        (["exit_code_is", "-1", "0"], "command index"),
        (["custom", "Bad-Id"], "check id"),
    ],
)
def test_a_bad_predicate_is_refused_at_construction(bad, msg):
    with pytest.raises(PredicateError, match=msg):
        parse(bad)


def test_text_equals_tolerates_exactly_one_trailing_newline(ws):
    assert evaluate([["text_equals", "out/n.txt", "3.141"]], ws) == [True]
    (ws / "two.txt").write_text("x\n\n")
    assert evaluate([["text_equals", "two.txt", "x"]], ws) == [False]
