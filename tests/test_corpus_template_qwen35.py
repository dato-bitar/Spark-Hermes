"""Does a training row survive the chat template that will render it? -- for the current base.

`tests/test_corpus_template.py` asks this of the ATEM template and is kept, because that dialect is
still implemented. This file asks it of `Qwen/Qwen3.8-27B`, which is the model the pipeline actually
trains, and the answers differ in ways that matter more than the ones they share.

Two of the three defects that file was written for cannot recur here, and one gets WORSE:

  * `tools` holding bare names raised on the ATEM template -- `'str object' has no attribute
    'name'`. This template calls `tool | tojson`, which is perfectly happy with a string, and
    renders `<tools>\\n"terminal"\\n</tools>`. No exception, no warning, and a corpus that
    conditions the model on a tool block containing two quoted words instead of two function
    definitions. The renderer's refusal in `hermes.format._tool_field` is therefore not a
    convenience here -- it is the ONLY thing between this pipeline and a silent corruption, and
    that is what `test_bare_names_would_render_as_garbage_rather_than_raise` pins.

  * `function.arguments` as a JSON string still raises, so that one stays survivable.

  * reasoning in `content` was silently dropped by the ATEM template on tool-calling turns. This
    template renders `content` AND `tool_calls` on the same turn, so nothing is dropped -- but
    `reasoning_in_content` is still False, because `<think>` is rendered from `reasoning_content`
    and a `<think>` block written into `content` would be trained as literal visible prose sitting
    next to a real one.

The rendered text is what gets asserted on, in the same jinja sandbox `transformers` uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from hermes import qwen35
from hermes.format import to_messages_record
from hermes.pin import load_tool_schemas
from hermes.protocol import DIALECTS, ProtocolError, parse_turn
from hermes.trajectory import FINAL, THINKING, TOOL_CALL, TOOL_RESULT, AgentTrajectory, Step

QWEN35 = DIALECTS["qwen35"]
TEMPLATE = Path("hermes/templates/chat-template-qwen35.jinja")
SCHEMAS = Path("hermesbench/harness/tools.json")


def _render(record: dict[str, Any], *, add_generation_prompt: bool = False) -> str:
    """The pinned template, in the sandbox `transformers` renders chat templates in."""
    environment = ImmutableSandboxedEnvironment()

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string(TEMPLATE.read_text(encoding="utf-8"))
    return template.render(
        messages=record["messages"],
        tools=record.get("tools"),
        add_generation_prompt=add_generation_prompt,
    )


@pytest.fixture
def trajectory() -> AgentTrajectory:
    """Reasoning, then a call, then a result, then an answer -- with reasoning on the calling turn."""
    return AgentTrajectory(
        task="Count the lines in logs/one.log.",
        success=True,
        system="You work from evidence the workspace can produce.",
        tools_available=("terminal", "file_read"),
        steps=(
            Step(kind=THINKING, content="wc may not exist here. Read the file instead."),
            Step(kind=TOOL_CALL, tool="file_read", args={"path": "logs/one.log"}, call_id="c0"),
            Step(kind=TOOL_RESULT, content="a\nb\nc\n", call_id="c0", ok=True),
            Step(kind=THINKING, content="Three lines."),
            Step(kind=FINAL, content="3"),
        ),
    )


@pytest.fixture
def schemas() -> dict[str, dict[str, Any]]:
    return load_tool_schemas(SCHEMAS)


def test_a_qwen35_row_renders_at_all(trajectory, schemas):
    """The whole point: a row built for this dialect reaches this model's wire format."""
    text = _render(to_messages_record(trajectory, dialect=QWEN35, tool_schemas=schemas))
    assert "<tool_call>\n<function=file_read>" in text, "the call has to reach the wire format"
    assert "<parameter=path>\nlogs/one.log\n</parameter>" in text, "with its arguments"
    assert "<tool_response>\na\nb\nc\n</tool_response>" in text, "and its result has to come back"


def test_the_rendered_call_is_what_the_parser_reads_back(trajectory, schemas):
    """The round trip that makes the dialect real, closed through the model's OWN template.

    `hermes.qwen35.render_tool_call` and this template are two independent spellings of the same
    format, written from the same source but not from each other. Asserting the parser handles its
    own renderer's output would prove nothing; asserting it handles the TEMPLATE's output is what
    says the corpus and the runtime agree.
    """
    text = _render(to_messages_record(trajectory, dialect=QWEN35, tool_schemas=schemas))
    assistant = text.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]

    turn = parse_turn(assistant, dialect=QWEN35, schemas=schemas)
    assert turn.malformed == (), turn.malformed
    assert len(turn.calls) == 1
    assert turn.calls[0].name == "file_read"
    assert turn.calls[0].arguments == {"path": "logs/one.log"}
    assert "wc may not exist here" in turn.scratch_pad


def test_the_renderer_agrees_with_the_template_byte_for_byte(trajectory, schemas):
    """Stronger than "both parse": the two must produce identical bytes.

    A renderer that merely round-trips could still emit a variant spelling -- different whitespace,
    a different parameter order -- and train the model on a shape its serving template never
    writes. That difference is invisible in every metric until inference.
    """
    text = _render(to_messages_record(trajectory, dialect=QWEN35, tool_schemas=schemas))
    assert qwen35.render_tool_call("file_read", {"path": "logs/one.log"}) in text
    assert qwen35.render_tool_response("file_read", "a\nb\nc\n") in text


def test_the_reasoning_survives_on_a_tool_calling_turn(trajectory, schemas):
    """Rendered from `reasoning_content` into `<think>`, on the turn that reasons toward a call."""
    text = _render(to_messages_record(trajectory, dialect=QWEN35, tool_schemas=schemas))
    assert "<think>\nwc may not exist here. Read the file instead.\n</think>" in text


def test_reasoning_is_not_put_in_content_even_though_content_survives_here(trajectory, schemas):
    """Why `reasoning_in_content` is still False for a template that keeps `content`.

    The ATEM template DROPPED content beside tool_calls, which made this flag a hard requirement.
    This one keeps it -- so a `<think>` block written into `content` would not vanish. It would
    render as literal prose immediately after the real `<think>` block, training the model to
    write its deliberation twice, once as reasoning and once as the answer.
    """
    record = to_messages_record(trajectory, dialect=QWEN35, tool_schemas=schemas)
    assistant = next(m for m in record["messages"] if m.get("tool_calls"))
    assert "reasoning_content" in assistant
    assert "<think>" not in assistant["content"]

    text = _render(record)
    assert text.count("<think>") == text.count("</think>") == 2, "one per assistant turn, no more"


def test_the_pinned_template_still_keeps_content_beside_tool_calls():
    """A guard on the paragraph above. If upstream starts dropping content on a calling turn, this
    dialect's requirements change and it should be visible rather than inferred."""
    text = _render(
        {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "PROSE-BESIDE-A-CALL",
                    "reasoning_content": "deliberation",
                    "tool_calls": [
                        {
                            "id": "c0",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": {"command": "ls"}},
                        }
                    ],
                },
            ],
            "tools": [],
        }
    )
    assert "PROSE-BESIDE-A-CALL" in text


def test_the_pinned_template_still_refuses_a_json_string(schemas):
    """`tool_arguments_json=False` is only correct while the template actually refuses a string.

    This template iterates `tool_call.arguments|items`, which needs a mapping. If upstream starts
    accepting a string the flag becomes a choice rather than a requirement, which is worth knowing.
    """
    record = {
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c0", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
                ],
            },
        ],
        "tools": [],
    }
    with pytest.raises(TypeError, match="Can only get item pairs from a mapping"):
        _render(record)


def test_bare_names_would_render_as_garbage_rather_than_raise():
    """The failure this template makes SILENT, and the reason the renderer's refusal is load-bearing.

    The ATEM template raised on a bare name -- `'str object' has no attribute 'name'` -- so that
    mistake announced itself. This one runs `tool | tojson`, which turns the string "terminal" into
    the JSON string `"terminal"` and renders it inside `<tools>` without complaint.

    A corpus built that way trains the model on a tool block listing two quoted words where two
    function definitions should be. Nothing raises, nothing is empty, and every row looks fine.
    """
    text = _render({"messages": [{"role": "user", "content": "go"}], "tools": ["terminal", "file_read"]})
    block = text.split("<tools>")[1].split("</tools>")[0]
    assert block.strip() == '"terminal"\n"file_read"', (
        "if this ever starts raising, the refusal in hermes.format is a convenience again rather than the only guard"
    )


def test_a_qwen35_row_without_schemas_is_refused_not_written(trajectory):
    """Given the test above, this refusal is what actually prevents the corruption. Refused at the
    renderer, because the alternative here is not a crash hours into a training run -- it is a
    training run that completes and produces a worse model for no visible reason."""
    with pytest.raises(ProtocolError, match="renders tool definitions from the row"):
        to_messages_record(trajectory, dialect=QWEN35)


def test_a_tool_this_harness_has_no_schema_for_is_refused(trajectory, schemas):
    """A stub signature would show the model parameters that do not exist."""
    invented = AgentTrajectory(
        task=trajectory.task,
        success=True,
        system=trajectory.system,
        tools_available=("terminal", "telepathy"),
        steps=trajectory.steps,
    )
    with pytest.raises(ProtocolError, match="no schema for"):
        to_messages_record(invented, dialect=QWEN35, tool_schemas=schemas)


def test_tool_results_come_back_in_a_user_turn(trajectory, schemas):
    """`Dialect.tool_result_role` says `tool`, and the rendered prompt says `user`.

    Both are correct and the template is what reconciles them -- it rewrites the role. Pinned
    because the two disagreeing is exactly the kind of thing that reads as a bug and gets
    "fixed" by changing the flag, which would then stop the template from finding the message.
    """
    record = to_messages_record(trajectory, dialect=QWEN35, tool_schemas=schemas)
    assert [m["role"] for m in record["messages"] if "tool_call_id" in m] == ["tool"]

    text = _render(record)
    assert "<|im_start|>user\n<tool_response>" in text
    assert "<|im_start|>tool" not in text


def test_the_generation_prompt_opens_the_reasoning_block():
    """The fact `hermes.qwen35._split_reasoning` has a whole branch for.

    The prompt ends with `<think>\\n`, so a raw completion begins INSIDE the block and carries only
    the closing tag. A parser matching a balanced pair finds nothing, reports every turn as having
    skipped deliberation, and leaves `</think>` in the text the task is graded on.
    """
    text = _render({"messages": [{"role": "user", "content": "go"}], "tools": []}, add_generation_prompt=True)
    assert text.endswith("<|im_start|>assistant\n<think>\n")

    # The completion that follows such a prompt, parsed the way the harness will see it.
    turn = parse_turn("I will just answer.\n</think>\n\nThe answer is 3.", dialect=QWEN35)
    assert turn.scratch_pad == "I will just answer."
    assert turn.text == "The answer is 3."
    assert turn.malformed == ()
    assert turn.abstained


def test_wire_markup_in_a_pre_fix_trajectory_is_stripped_before_training(schemas):
    """Leftover call markup inside reasoning goes on the channel the template renders `<think>` from.

    Stripped rather than refused, because refusing makes every pre-fix log unaggregatable, and
    counted on the message, because a row that was silently repaired is a row nobody can audit.
    Same guarantee `tests/test_corpus_template.py` makes for ATEM, over this format's markup.
    """
    dirty = AgentTrajectory(
        task="Count the lines in logs/one.log.",
        success=True,
        system="s",
        tools_available=("terminal",),
        steps=(
            Step(
                kind=THINKING,
                content=(
                    "We have logs directory. Let's list logs.\n"
                    "<tool_call>\n<function=terminal>\n<parameter=command>\nls -la logs\n"
                    "</parameter>\n</function>\n</tool_call>"
                ),
            ),
            Step(kind=TOOL_CALL, tool="terminal", args={"command": "ls -la logs"}, call_id="c0"),
            Step(kind=TOOL_RESULT, content="one.log\n", call_id="c0", ok=True),
            Step(kind=FINAL, content="3"),
        ),
    )
    record = to_messages_record(dirty, dialect=QWEN35, tool_schemas=schemas)
    assistant = next(m for m in record["messages"] if m.get("tool_calls"))
    assert "We have logs directory" in assistant["reasoning_content"], "the prose survives"
    assert "<tool_call>" not in assistant["reasoning_content"], "the markup does not"
    assert assistant["reasoning_markup_stripped"] == 1, "and the repair is on the record"


def test_a_clean_trajectory_carries_no_repair_marker(trajectory, schemas):
    """The marker means something only if it is absent when nothing was repaired."""
    record = to_messages_record(trajectory, dialect=QWEN35, tool_schemas=schemas)
    assert all("reasoning_markup_stripped" not in m for m in record["messages"])
