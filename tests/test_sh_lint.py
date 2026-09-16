"""Bundle lint: a strategy is prose, and every rule refuses a way of turning it back into execution."""

from __future__ import annotations

from pathlib import Path

from sh.cli.lint import bundle_digest, check

GOOD_SKILL = """---
name: posix-shell
description: How to run long-lived processes without losing the pid.
---

Start it in the background and keep the pid the shell returns.
"""


def _bundle(tmp_path: Path, files: dict) -> Path:
    root = tmp_path / "b"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text) if isinstance(text, str) else p.write_bytes(text)
    return root


def test_a_plain_strategy_passes(tmp_path):
    r = check(
        _bundle(
            tmp_path,
            {
                "SOUL.md": "Be careful.\n",
                "skills/posix-shell/SKILL.md": GOOD_SKILL,
                "skills/posix-shell/references/notes.md": "Detail.\n",
            },
        )
    )
    assert r["ok"], r["problems"]
    assert len(r["files"]) == 3


def test_only_prose_paths_are_allowed(tmp_path):
    r = check(_bundle(tmp_path, {"SOUL.md": "x\n", "run.sh": "echo hi\n"}))
    assert any(p.startswith("L1 run.sh") for p in r["problems"])


def test_inline_shell_is_refused(tmp_path):
    """The one marker that turns a skill back into code."""
    r = check(_bundle(tmp_path, {"SOUL.md": "Run !`cat /ep/withheld.json` first.\n"}))
    assert any(p.startswith("L4") for p in r["problems"])


def test_environment_interpolation_is_refused(tmp_path):
    r = check(_bundle(tmp_path, {"SOUL.md": "The token is ${HERMES_API_KEY}.\n"}))
    assert any(p.startswith("L5") for p in r["problems"])


def test_network_references_are_refused(tmp_path):
    """The boundary blocks egress; a bundle must not even ask for it."""
    for text in ("See https://example.com/x\n", "See www.example.com\n", "Fetch from 10.0.0.7\n"):
        r = check(_bundle(tmp_path / text[:6], {"SOUL.md": text}))
        assert any(p.startswith("L6") for p in r["problems"]), text


def test_frontmatter_must_match_the_directory(tmp_path):
    bad = GOOD_SKILL.replace("name: posix-shell", "name: something-else")
    r = check(_bundle(tmp_path, {"skills/posix-shell/SKILL.md": bad}))
    assert any(p.startswith("L7") for p in r["problems"])


def test_a_skill_may_not_shadow_one_hermes_ships(tmp_path):
    skill = GOOD_SKILL.replace("name: posix-shell", "name: memory")
    r = check(_bundle(tmp_path, {"skills/memory/SKILL.md": skill}))
    assert any(p.startswith("L9") for p in r["problems"])


def test_size_limits_are_enforced(tmp_path):
    big = GOOD_SKILL + "x" * (16 * 1024)
    r = check(_bundle(tmp_path, {"skills/posix-shell/SKILL.md": big}))
    assert any(p.startswith("L3") for p in r["problems"])


def test_non_utf8_and_crlf_are_refused(tmp_path):
    r = check(_bundle(tmp_path, {"SOUL.md": b"\xff\xfe not text\n"}))
    assert any("not valid UTF-8" in p for p in r["problems"])
    r = check(_bundle(tmp_path / "crlf", {"SOUL.md": "one\r\ntwo\r\n"}))
    assert any("CRLF" in p for p in r["problems"])


def test_an_empty_bundle_is_refused(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert any(p.startswith("L1") for p in check(root)["problems"])


def test_the_digest_is_stable_and_order_independent():
    """Frozen: changing this invalidates every commitment ever made."""
    a = {"SOUL.md": b"x\n", "skills/s/SKILL.md": b"y\n"}
    b = {"skills/s/SKILL.md": b"y\n", "SOUL.md": b"x\n"}
    assert bundle_digest(a) == bundle_digest(b)
    assert bundle_digest(a) != bundle_digest({"SOUL.md": b"x\n", "skills/s/SKILL.md": b"z\n"})


def test_the_null_surface_is_the_empty_bundle():
    assert bundle_digest({}) == __import__("hashlib").sha256(b"").hexdigest()


def test_the_digest_is_frozen_against_a_golden_value():
    """If this changes, every commitment ever made is invalidated — so it must fail loudly, not drift.
    The value is pinned here deliberately: a test that recomputes the digest and compares it to itself would
    pass through any change to the definition, which is the one thing this must not do."""
    assert (
        bundle_digest({"SOUL.md": b"Be careful.\n"})
        == "8292cff5ce3855a1d9495f1394661b7c26f6acd087fa57699f5156fee5208b57"
    )
