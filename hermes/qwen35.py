"""The QWEN35 wire format, as Qwen3.8-27B natively speaks it.

`hermes.protocol` covers what changed *between Hermes generations* and holds it as data because
those differences are enumerable. `hermes.atem` covers a format that announced itself as foreign
in every tag it wrote. This one is harder than either, and the reason is worth stating before the
code:

**It wears Hermes's tags.** `<tool_call>`, `<tool_response>`, `<think>` -- all three are the
Hermes spelling, and all three appear in this model's own chat template. Only the payload
differs::

    Hermes 4        <tool_call>
                    {"name": "terminal", "arguments": {"command": "ls"}}
                    </tool_call>

    QWEN35          <tool_call>
                    <function=terminal>
                    <parameter=command>
                    ls
                    </parameter>
                    </function>
                    </tool_call>

ATEM could be detected by looking for `<atem:`. This cannot be detected by looking at tags at
all, and the consequence is specific: `hermes.protocol.parse_turn` finds `<tool_call>`, advances
past it, hands `<function=terminal>` to a JSON decoder and raises. Every turn. So the wrong
dialect here does not produce a model that appears to call no tools -- it produces one that
appears to call tools and get the syntax wrong every single time, which is `malformed_turns`
at 100%, which is a metric the promotion gate bounds. The model would be blamed for the
harness's mistake, and the number would look like a training problem.

That is why `Dialect.family` exists and why `hermes.protocol.WIRE_MODULES` dispatches on it
rather than on which tags a completion happens to contain.

## Values are text, so types have to be recovered

One JSON object per call becomes one element per *parameter*, and a parameter's value is text
rather than JSON. There is no arguments object to decode and no place for a JSON syntax error to
occur -- and correspondingly a new failure class that Hermes cannot have, which most of the
parsing below is about.

The template writes non-string values through `tojson` and strings verbatim, so a parameter
reading `true` is indistinguishable from the *string* `"true"` on the wire. That is a property of
the format, not a defect here. `hermes.protocol.coerce_text_arguments` applies the tool's declared
schema to recover what it can and leaves everything else a string; ATEM has the identical problem
and calls the identical function, so the two formats cannot drift apart in how they coerce.

## The completion begins *inside* the reasoning block

The template's generation prompt ends with `<|im_start|>assistant\\n<think>\\n`. The opening tag is
therefore part of the PROMPT, not of the completion, and a server handing back raw generated text
returns::

    ...deliberation...
    </think>

    the answer

with a closing tag and no opening one. A parser matching `<think>(.*?)</think>` finds nothing
here, reports the turn as having skipped deliberation, and leaves `</think>` sitting in the text
that gets graded as the model's answer. Both shapes are handled below: the balanced pair, for a
server that echoes the prompt or a training row rendered by the template, and the leading
unbalanced close, for a raw completion.

## A tool result carries no name

Hermes has no call ids -- nothing correlates a call with its result, and correlation is
positional. This format is one step worse: its `<tool_response>` block has no name attribute and
no JSON body either, just the raw content. So the tool that produced a result is not merely
uncorrelated, it is *unrecoverable* from the wire text. `render_tool_response` still takes a name,
because the dispatch in `hermesbench.policy` passes one, and drops it on the floor deliberately
rather than inventing a slot the model never saw.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hermes.protocol import ParsedCall, ParsedTurn, ProtocolError, coerce_text_arguments

DIALECT_NAME = "qwen35"

# Shared with Hermes, which is the entire difficulty. Spelled out as constants anyway so that a
# reader of this module does not have to hold "same tag, different payload" in their head.
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
RESPONSE_OPEN, RESPONSE_CLOSE = "<tool_response>", "</tool_response>"
FUNCTION_OPEN, FUNCTION_CLOSE = "<function=", "</function>"
PARAM_OPEN, PARAM_CLOSE = "<parameter=", "</parameter>"

# The cheapest substring that means "there might be a call in here", read by callers that want to
# skip a full parse. Every wire module exposes this name so the caller does not have to know which
# constant each one happens to call its opening tag. Note that this one is NOT distinctive between
# formats -- Hermes writes `<tool_call>` too -- so it is a prefilter and never a detector.
CALL_MARKER = CALL_OPEN

# What a rendered tool result starts with. The whole tag, because this format has no attributes on
# it -- and no name inside it either, which is the point made in the module docstring.
RESULT_PREFIX = RESPONSE_OPEN

# Inline, unlike ATEM's separate `self` channel -- but see the module docstring: the opening tag
# lives in the generation prompt, so a raw completion carries only the close.
REASONING_TAG = "think"
REASONING_OPEN, REASONING_CLOSE = "<think>", "</think>"

_BLOCK_RE = re.compile(re.escape(CALL_OPEN) + r"(.*?)" + re.escape(CALL_CLOSE), re.DOTALL)

# `[^>\n]*` and not `[^>]*`: a name is written unquoted and terminated by `>`, so it cannot span a
# line. Allowing newlines let a missing `>` swallow the whole rest of the block into the name and
# report one truncated call as one *successful* call to a tool named after the payload.
_FUNCTION_RE = re.compile(r"<function=([^>\n]*)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]*)>(.*?)</parameter>", re.DOTALL)

_BALANCED_THINK_RE = re.compile(re.escape(REASONING_OPEN) + r"(.*?)" + re.escape(REASONING_CLOSE), re.DOTALL)

# A close with no open, at the head of the text: the completion started inside the block because
# the prompt opened it. Anchored at the start -- a `</think>` appearing later, after content, is a
# stray tag and is reported as one rather than silently re-cutting the turn.
_LEADING_THINK_CLOSE_RE = re.compile(r"\A(.*?)" + re.escape(REASONING_CLOSE), re.DOTALL)

# An opened call block with no close. Truncation by token limit lands here, and this format makes
# it likelier than Hermes does: the element form of a call is several times longer than the
# equivalent JSON, so there is more of it to cut off.
_UNCLOSED_RE = re.compile(re.escape(CALL_OPEN) + r"(?!.*" + re.escape(CALL_CLOSE) + r")", re.DOTALL)

# Tag-shaped and not the tag, looked for only after the real blocks are removed. `<function ...>`
# and `<parameter ...>` in the ATTRIBUTE style are called out by name because they are what every
# other XML-ish tool format uses -- including the one in `hermes.atem` -- so it is the mistake a
# model that has seen mixed training data actually makes, and the one a human editing a fixture
# makes too.
_LOOSE_RE = re.compile(
    r"<\s*/?\s*(?:function|parameter)\b[^>\n]*>?|<\s*/\s*tool_call\s*>",
    re.IGNORECASE,
)

# Hermes's own payload, sitting where this format wants elements. Detected so the report can say
# which format the model actually produced instead of "no function in block" -- the difference
# between a dialect misconfiguration and a model that lost the format is the whole diagnosis.
_HERMES_PAYLOAD_RE = re.compile(r"\A\s*[\[{]")

# `<function name="x">` rather than `<function=x>`, inside an otherwise well-formed call block.
# One character from correct, and the format every other XML-ish tool markup uses -- including
# `hermes.atem`, so a mixed corpus makes it a live risk rather than a hypothetical one.
_ATTRIBUTE_SPELLING_RE = re.compile(r"<\s*(?:function|parameter)\s+[^>\n]*>", re.IGNORECASE)


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """One call, byte-for-byte as the model's own template writes it.

    Parameter order follows the mapping's order rather than being sorted. The template does the
    same, and a renderer that sorted would produce training rows the model never saw itself
    generate -- teaching it a normalisation nothing else in the stack applies.
    """
    if not name:
        raise ProtocolError("a tool call needs a name")
    parts = [CALL_OPEN, "\n", FUNCTION_OPEN, _name(name), ">\n"]
    for key, value in arguments.items():
        parts.append(f"{PARAM_OPEN}{_name(key)}>\n{_value(value)}\n{PARAM_CLOSE}\n")
    parts.append(f"{FUNCTION_CLOSE}\n{CALL_CLOSE}")
    return "".join(parts)


def render_tool_calls(calls: list[tuple[str, dict[str, Any]]]) -> str:
    """Several calls. One `<tool_call>` block each, matching the template.

    The template emits a separate block per entry in `tool_calls` -- it does not nest several
    `<function=>` elements in one block. Nesting would parse here and is not what the model
    produces, so training on it would teach a shape the serving layer never sends.
    """
    return "\n".join(render_tool_call(name, arguments) for name, arguments in calls)


def render_tool_response(name: str, content: Any) -> str:
    """A tool result. `name` is accepted and deliberately discarded -- see the module docstring.

    The template renders a `tool` message as its content between `<tool_response>` tags, with no
    name attribute and no JSON envelope. Adding either would put text in the history that the model
    was never trained on, in the position it reads observations from.

    The body is TRIMMED, because the template trims it (`render_content(...)|trim`). Tool output
    almost always ends in a newline -- `ls`, `cat`, any shell command -- so without this the
    runtime sends `...\\n\\n</tool_response>` where training rows carry `...\\n</tool_response>`,
    on essentially every tool result in every episode. A one-character difference repeated that
    often is a distribution shift, and nothing else in the stack would have reported it.
    """
    body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return f"{RESPONSE_OPEN}\n{body.strip()}\n{RESPONSE_CLOSE}"


def _name(text: str) -> str:
    """A tool or parameter name, which is written unquoted and terminated by `>`.

    There is no escape syntax for this position -- the template interpolates the name raw -- so a
    name containing `>` or a newline does not produce a badly-escaped call, it produces a
    STRUCTURALLY different one: the tag closes early and the remainder becomes the parameter body.
    A crafted tool name is therefore a way to change which call gets parsed, so it is refused here
    rather than mangled into something that parses as a valid call to the wrong thing.
    """
    text = str(text)
    if ">" in text or "\n" in text or "<" in text:
        raise ProtocolError(
            f"name {text!r} contains a character this format cannot express: `<function=NAME>` is "
            "unquoted and terminated by `>`, so there is no escaping available and the call would "
            "silently reparse as a different one"
        )
    return text


def _value(value: Any) -> str:
    """A parameter value, exactly as the template writes it.

    The template's rule is one line: strings verbatim, everything else through `tojson`. So
    booleans arrive as `true`/`false`, `None` as `null`, and anything list- or dict-shaped as
    JSON -- and none of it is marked as having been converted, which is why `coerce_text_arguments`
    exists on the way back.
    """
    if isinstance(value, str):
        # Not escaped beyond the closing tag. The template escapes nothing at all, so escaping
        # more would produce text the model does not write and hand a tool entities instead of
        # characters. The closing tag is escaped because leaving it would let a value terminate
        # its own parameter and turn the rest of the call into markup.
        return value.replace(PARAM_CLOSE, "&lt;/parameter>")
    return json.dumps(value, ensure_ascii=False)


def coerce(arguments: dict[str, str], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Recover declared types from text values. See `hermes.protocol.coerce_text_arguments`."""
    return coerce_text_arguments(arguments, schema)


def _unpad(raw: str) -> str:
    """Undo the newlines the template puts around a value, and only those.

    The template writes `<parameter=k>\\n` + value + `\\n</parameter>`, so exactly one newline on
    each side is framing rather than content. `.strip()` would also eat a value's own leading
    indentation or trailing blank line, which for this project is not hypothetical: the tools a
    worker calls most write files and run shell commands, and both take arguments where leading
    whitespace is load-bearing.
    """
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def parse_turn(content: str, *, reasoning: str = "", schemas: dict[str, dict[str, Any]] | None = None) -> ParsedTurn:
    """Read one assistant turn.

    `reasoning` is taken as an argument for the server that splits it out into
    `reasoning_content`, and recovered from the text otherwise -- both happen, and which one
    depends on serving configuration rather than on the model.

    `schemas` maps tool name to its JSON Schema and is what `coerce` needs. Without it every
    argument stays a string, which is a usable degradation: a tool declaring `{"command":
    "string"}` needs no recovery at all, and most do.
    """
    malformed: list[str] = []
    content, inline_reasoning, stray = _split_reasoning(content)
    malformed.extend(stray)
    if inline_reasoning and not reasoning:
        reasoning = inline_reasoning

    calls, spans, block_problems = _read_blocks(content, schemas)
    malformed.extend(block_problems)
    remainder = _strip(content, spans)

    # An unclosed block ends the turn, so everything from it onward is the truncated call. Cut it
    # away BEFORE the near-miss scan, not after: the truncated text still contains a `<function=`
    # and one or more `<parameter=`, so scanning it reported ONE broken turn as three malformed
    # entries. `malformed_turns` is bounded by the promotion gate, so the over-count made this
    # dialect look worse than Hermes for the identical failure -- in the metric that decides
    # whether a model ships. `hermes.atem` carries the same cut for the same reason.
    truncated = _UNCLOSED_RE.search(remainder)
    if truncated:
        malformed.append(
            f"{CALL_OPEN} was opened and never closed, so the call is truncated. An element-form "
            "call is several times longer than the JSON equivalent, which makes this the likeliest "
            "way a turn breaks under a token limit."
        )
        remainder = remainder[: truncated.start()]

    for loose in _LOOSE_RE.finditer(remainder):
        malformed.append(
            f"{loose.group(0)[:40]!r} is call-shaped but not a call. This format writes "
            f"`{FUNCTION_OPEN}NAME>` and `{PARAM_OPEN}KEY>` with the name in the tag itself; the "
            'attribute spelling `<function name="...">` is a different format and parses as nothing.'
        )

    text = _LOOSE_RE.sub("", remainder).strip()
    return ParsedTurn(calls=tuple(calls), text=text, scratch_pad=reasoning.strip(), malformed=tuple(malformed))


def _split_reasoning(content: str) -> tuple[str, str, list[str]]:
    """(content without the reasoning block, the reasoning, problems found).

    Two shapes, because the opening tag belongs to the generation prompt rather than to the
    completion. A balanced pair is what a rendered training row or a prompt-echoing server gives;
    a leading unbalanced close is what a raw completion gives. Handling only the first leaves
    `</think>` inside the text a task is graded on, and reports every turn as having skipped
    deliberation.
    """
    if (balanced := _BALANCED_THINK_RE.search(content)) is not None:
        reasoning = balanced.group(1)
        rest = content[: balanced.start()] + content[balanced.end() :]
        return rest, reasoning.strip(), _stray_closes(rest)
    if REASONING_CLOSE in content and (leading := _LEADING_THINK_CLOSE_RE.match(content)) is not None:
        rest = content[leading.end() :]
        return rest, leading.group(1).strip(), _stray_closes(rest)
    return content, "", []


def _stray_closes(rest: str) -> list[str]:
    """A second `</think>` after the block already closed.

    Reported rather than removed. It means the reasoning boundary is ambiguous, and the boundary is
    what separates what the model *planned* from what it *did* -- so guessing at it moves calls
    between "considered" and "made" with nothing recording that a guess happened.
    """
    if REASONING_CLOSE not in rest:
        return []
    return [
        f"a second {REASONING_CLOSE} appears after the reasoning block already closed, so the "
        "boundary between deliberation and answer is ambiguous"
    ]


def _read_blocks(
    content: str, schemas: dict[str, dict[str, Any]] | None
) -> tuple[list[ParsedCall], list[tuple[int, int]], list[str]]:
    calls: list[ParsedCall] = []
    spans: list[tuple[int, int]] = []
    malformed: list[str] = []

    for block in _BLOCK_RE.finditer(content):
        spans.append(block.span())
        body = block.group(1)
        functions = list(_FUNCTION_RE.finditer(body))
        if not functions:
            if _HERMES_PAYLOAD_RE.match(body):
                # The single most useful thing this parser can say. A JSON payload here means the
                # dialect is misconfigured -- the model is speaking Hermes because something told
                # it to -- and not that the model forgot the format. Reported as a distinct
                # message so a run that trips it once names the cause in its own log.
                malformed.append(
                    f"{CALL_OPEN} contains a JSON payload, which is the HERMES form of a call. This "
                    f"dialect ({DIALECT_NAME}) expects `{FUNCTION_OPEN}NAME>` elements inside the "
                    "tag. The two formats share the tag and differ only here, so this is very "
                    "likely a dialect set to hermes-4 for a model that speaks qwen35, or a system "
                    "prompt from the wrong one."
                )
            elif (attr := _ATTRIBUTE_SPELLING_RE.search(body)) is not None:
                # The attribute spelling, INSIDE a block. The near-miss scan at the end of
                # `parse_turn` cannot see this: it runs on the text with call blocks already cut
                # out, so a wrong-spelled function nested in a right-spelled `<tool_call>` was
                # reported as "block contains nothing callable" -- true, and useless. The model was
                # one character away from a valid call and the report did not say which.
                malformed.append(
                    f"{attr.group(0)[:40]!r} uses the ATTRIBUTE spelling. This format writes the "
                    f"name in the tag itself -- `{FUNCTION_OPEN}NAME>`, `{PARAM_OPEN}KEY>` -- and "
                    'the `name="..."` form is a different format that parses as nothing here.'
                )
            else:
                # A block with no function is not an abstention and not a call. Left unreported it
                # would resolve to "no calls found", the reading `ParsedTurn` exists to prevent.
                malformed.append(
                    f"{CALL_OPEN} block contains no {FUNCTION_OPEN}; the model opened a call block "
                    "and wrote nothing callable in it"
                )
            continue
        for function in functions:
            name = function.group(1).strip()
            if not name:
                malformed.append(f"{FUNCTION_OPEN}> has an empty name")
                continue
            raw = {p.group(1).strip(): _unpad(p.group(2)) for p in _PARAM_RE.finditer(function.group(2))}
            leftover = _PARAM_RE.sub("", function.group(2))
            if PARAM_OPEN in leftover:
                # An unterminated parameter: its value swallowed the rest of the call. Reported,
                # because the parameters that DID parse look like a complete call -- and executing
                # a partial argument set is worse than executing nothing.
                malformed.append(f"{name}: a {PARAM_OPEN} is not closed, so its value ran to the end of the call")
                continue
            calls.append(ParsedCall(name=name, arguments=coerce(raw, (schemas or {}).get(name))))
    return calls, spans, malformed


def _strip(text: str, spans: list[tuple[int, int]]) -> str:
    out, last = [], 0
    for start, end in spans:
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out)


__all__ = [
    "CALL_CLOSE",
    "CALL_MARKER",
    "CALL_OPEN",
    "DIALECT_NAME",
    "FUNCTION_CLOSE",
    "FUNCTION_OPEN",
    "PARAM_CLOSE",
    "PARAM_OPEN",
    "REASONING_CLOSE",
    "REASONING_OPEN",
    "REASONING_TAG",
    "RESPONSE_CLOSE",
    "RESULT_PREFIX",
    "RESPONSE_OPEN",
    "coerce",
    "parse_turn",
    "render_tool_call",
    "render_tool_calls",
    "render_tool_response",
]
