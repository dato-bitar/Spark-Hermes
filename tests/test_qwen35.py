"""The QWEN35 wire format: rendering, parsing, and the failures it makes possible.

`tests/test_corpus_template_qwen35.py` checks this format against the model's own jinja template.
This file checks the parser against the completions a served model actually returns -- truncated,
mis-spelled, wrapped in the wrong dialect's payload -- because those never reach a template.

The organising fact, repeated here because every test below depends on it: this format's tags ARE
Hermes's. `<tool_call>`, `<tool_response>` and `<think>` all appear in both. Only the payload
differs. So the interesting failures are not "unrecognised markup" but "recognised markup, read by
the wrong parser", and a report that says `invalid JSON` is a report blaming the model for a
harness misconfiguration.
"""

from __future__ import annotations

import pytest

from hermes import qwen35
from hermes.protocol import DIALECTS, ProtocolError, parse_turn

QWEN35 = DIALECTS["qwen35"]

SCHEMAS = {
    "terminal": {
        "type": "object",
        "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}, "quiet": {"type": "boolean"}},
        "required": ["command"],
    },
    "file_write": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "tags": {"type": "array"}},
    },
}


def _parse(text: str, **kwargs):
    return parse_turn(text, dialect=QWEN35, schemas=SCHEMAS, **kwargs)


# --- rendering -----------------------------------------------------------------------


def test_a_call_round_trips():
    rendered = qwen35.render_tool_call("terminal", {"command": "ls -la"})
    turn = _parse(rendered)
    assert turn.malformed == ()
    assert turn.calls[0].name == "terminal"
    assert turn.calls[0].arguments == {"command": "ls -la"}


def test_the_rendered_shape_is_the_templates_shape():
    """Spelled out rather than only round-tripped, so a change to the renderer has to change a
    literal here. A round-trip test passes just as happily against a format both sides invented."""
    assert qwen35.render_tool_call("terminal", {"command": "ls"}) == (
        "<tool_call>\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
    )


def test_parameter_order_follows_the_mapping_not_the_alphabet():
    """The template iterates the mapping. A renderer that sorted would produce training rows the
    model never saw itself generate, teaching a normalisation nothing else in the stack applies."""
    rendered = qwen35.render_tool_call("terminal", {"timeout": 5, "command": "ls"})
    assert rendered.index("<parameter=timeout>") < rendered.index("<parameter=command>")


def test_several_calls_get_a_block_each():
    """The template emits one `<tool_call>` per entry in `tool_calls` rather than nesting several
    `<function=>` in one block. Nesting would parse here and is not what the model produces."""
    rendered = qwen35.render_tool_calls([("terminal", {"command": "ls"}), ("terminal", {"command": "pwd"})])
    assert rendered.count("<tool_call>") == 2
    turn = _parse(rendered)
    assert [c.arguments["command"] for c in turn.calls] == ["ls", "pwd"]
    assert turn.malformed == ()


def test_non_string_values_are_written_as_json_like_the_template_does():
    rendered = qwen35.render_tool_call("terminal", {"quiet": True, "timeout": 30, "command": None})
    assert "<parameter=quiet>\ntrue\n</parameter>" in rendered
    assert "<parameter=timeout>\n30\n</parameter>" in rendered
    assert "<parameter=command>\nnull\n</parameter>" in rendered


def test_a_value_containing_the_closing_tag_cannot_terminate_its_own_parameter():
    """The template escapes nothing, so an unescaped `</parameter>` in a value would end the
    parameter early and turn the rest of the call into markup. An agent working on THIS repository
    writes that string routinely."""
    rendered = qwen35.render_tool_call("file_write", {"content": "see </parameter> above"})
    turn = _parse(rendered)
    assert turn.malformed == ()
    assert len(turn.calls) == 1


def test_a_name_that_would_restructure_the_call_is_refused():
    """`<function=NAME>` is unquoted and terminated by `>`, so there is no escaping available.

    A name containing `>` does not produce a badly-escaped call; it produces a structurally
    different one that parses cleanly as a call to something else. Refused rather than mangled."""
    with pytest.raises(ProtocolError, match="cannot express"):
        qwen35.render_tool_call("terminal>evil", {"command": "ls"})
    with pytest.raises(ProtocolError, match="cannot express"):
        qwen35.render_tool_call("terminal", {"a>b": "x"})


def test_a_call_needs_a_name():
    with pytest.raises(ProtocolError, match="needs a name"):
        qwen35.render_tool_call("", {"command": "ls"})


def test_a_tool_result_is_trimmed_the_way_the_template_trims_it():
    """Tool output almost always ends in a newline. Without the trim the runtime would send one
    more than every training row carries, on essentially every result in every episode."""
    assert qwen35.render_tool_response("terminal", "a\nb\n") == "<tool_response>\na\nb\n</tool_response>"
    assert qwen35.render_tool_response("terminal", {"lines": 3}) == '<tool_response>\n{"lines": 3}\n</tool_response>'


# --- type recovery -------------------------------------------------------------------


def test_declared_types_are_recovered_from_text():
    turn = _parse(qwen35.render_tool_call("terminal", {"command": "ls", "timeout": 30, "quiet": True}))
    assert turn.calls[0].arguments == {"command": "ls", "timeout": 30, "quiet": True}


def test_an_undeclared_parameter_stays_a_string():
    """Guessing is how `{"path": "123"}` becomes `{"path": 123}` and a tool receives an integer
    where it declared a filename. Under-recovering fails at the tool boundary instead of inside."""
    turn = _parse(
        "<tool_call>\n<function=terminal>\n<parameter=surprise>\n123\n</parameter>\n</function>\n</tool_call>"
    )
    assert turn.calls[0].arguments == {"surprise": "123"}


def test_a_string_typed_parameter_holding_a_number_is_not_converted():
    turn = _parse(qwen35.render_tool_call("terminal", {"command": "123"}))
    assert turn.calls[0].arguments == {"command": "123"}


def test_a_value_that_contradicts_its_declared_type_is_left_as_text():
    """Passing it through would move the failure into the tool, where the reason is no longer
    visible."""
    turn = _parse(
        "<tool_call>\n<function=terminal>\n<parameter=timeout>\nsoon\n</parameter>\n</function>\n</tool_call>"
    )
    assert turn.calls[0].arguments == {"timeout": "soon"}


def test_leading_whitespace_in_a_value_is_content_not_framing():
    """The template pads a value with exactly one newline on each side. `.strip()` would also eat a
    value's own indentation -- and the tools a worker calls most write files and run shell
    commands, where leading whitespace is load-bearing."""
    turn = _parse(
        "<tool_call>\n<function=file_write>\n<parameter=content>\n    indented\n\n</parameter>\n</function>\n</tool_call>"
    )
    assert turn.calls[0].arguments == {"content": "    indented\n"}


# --- reasoning -----------------------------------------------------------------------


def test_a_raw_completion_carries_only_the_closing_think_tag():
    """The generation prompt ends with `<think>\\n`, so the opener belongs to the prompt.

    A parser matching a balanced pair finds nothing here, reports the turn as having skipped
    deliberation, and leaves `</think>` in the text the task is graded on."""
    turn = _parse("Deliberating.\n</think>\n\nThe answer is 3.")
    assert turn.scratch_pad == "Deliberating."
    assert turn.text == "The answer is 3."
    assert "</think>" not in turn.text
    assert turn.malformed == ()


def test_a_balanced_pair_is_read_too():
    """What a rendered training row, or a prompt-echoing server, produces."""
    turn = _parse("<think>\nDeliberating.\n</think>\n\nThe answer is 3.")
    assert turn.scratch_pad == "Deliberating."
    assert turn.text == "The answer is 3."


def test_reasoning_supplied_by_the_server_is_preferred_over_the_text():
    turn = _parse("The answer is 3.", reasoning="from the reasoning_content field")
    assert turn.scratch_pad == "from the reasoning_content field"


def test_a_call_inside_the_reasoning_block_is_not_executed():
    """Deliberating about a call and deciding against it is not making it. Same rule Hermes
    applies to a call inside `<think>`, and unlike ATEM -- where the equivalent markup is a
    parser artifact rather than the model reasoning."""
    turn = _parse(
        "<think>\nMaybe I should run "
        "<tool_call>\n<function=terminal>\n<parameter=command>\nrm -rf /\n</parameter>\n</function>\n</tool_call>"
        "\nNo.\n</think>\n\nI will not."
    )
    assert turn.calls == ()
    assert turn.text == "I will not."


def test_a_second_closing_tag_makes_the_boundary_ambiguous_and_says_so():
    """The boundary separates what the model planned from what it did. Guessing at it moves calls
    between "considered" and "made" with nothing recording that a guess happened."""
    turn = _parse("a\n</think>\nb\n</think>\nc")
    assert any("ambiguous" in m for m in turn.malformed)


# --- the failures this format makes possible -----------------------------------------


def test_a_hermes_payload_is_reported_as_a_dialect_problem_not_a_model_problem():
    """The single most valuable message this parser emits.

    `<tool_call>` holding JSON means something configured the model to speak Hermes. Reported as
    that, because "no function in block" sends a reader looking at the model and this sends them
    to the dialect setting."""
    turn = _parse('<tool_call>\n{"name": "terminal", "arguments": {"command": "ls"}}\n</tool_call>')
    assert turn.calls == ()
    assert len(turn.malformed) == 1
    message = turn.malformed[0]
    assert "HERMES form" in message
    assert "hermes-4" in message and "qwen35" in message


def test_the_attribute_spelling_is_named_rather_than_called_empty():
    """`<function name="x">` is one character from correct and is what every other XML-ish tool
    format uses, including this repo's own ATEM. Inside a well-formed `<tool_call>` the near-miss
    scan cannot see it, because that scan runs on text with call blocks already removed."""
    turn = _parse('<tool_call>\n<function name="terminal">\n</function>\n</tool_call>')
    assert turn.calls == ()
    assert any("ATTRIBUTE spelling" in m for m in turn.malformed)


def test_truncation_is_reported_once_not_once_per_tag():
    """`malformed_turns` is bounded by the promotion gate. Counting a single truncated call three
    times -- for the block, the function and the parameter -- makes this dialect look worse than
    Hermes for the identical failure, in the metric that decides whether a model ships."""
    turn = _parse("<tool_call>\n<function=terminal>\n<parameter=command>\nls -la /very/long/pa")
    assert len(turn.malformed) == 1
    assert "truncated" in turn.malformed[0]


def test_an_empty_call_block_is_not_an_abstention():
    turn = _parse("<tool_call>\n\n</tool_call>")
    assert turn.calls == ()
    assert turn.malformed
    assert not turn.abstained, "an opened-and-empty call block is a failure, not a decision"


def test_an_unclosed_parameter_does_not_execute_a_partial_argument_set():
    """The parameters that DID parse look like a complete call. Executing a partial argument set
    is worse than executing nothing."""
    turn = _parse("<tool_call>\n<function=terminal>\n<parameter=command>\nls\n</function>\n</tool_call>")
    assert turn.calls == ()
    assert any("not closed" in m for m in turn.malformed)


def test_a_malformed_turn_executes_nothing():
    """`safe_calls` is what anything executing should read: a turn that confused the parser at all
    executes none of it, because a truncated call can leave a syntactically clean one behind it."""
    turn = _parse(
        qwen35.render_tool_call("terminal", {"command": "ls"})
        + "\n<tool_call>\n<function=terminal>\n<parameter=command>\ntrunc"
    )
    assert len(turn.calls) == 1
    assert turn.malformed
    assert turn.safe_calls == ()


def test_a_plain_answer_is_an_abstention():
    turn = _parse("</think>\n\nNo tool is needed; the answer is 4.")
    assert turn.abstained
    assert turn.text == "No tool is needed; the answer is 4."


def test_an_empty_completion_is_not_an_abstention():
    """An abstention requires something said; an empty completion is a crashed worker."""
    assert not _parse("</think>\n\n").abstained
