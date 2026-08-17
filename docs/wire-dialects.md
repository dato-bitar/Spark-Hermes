# Why this repository speaks three wire formats

The project's goals are **a stable harness, faster learning, and less ecosystem breakage**. This
repository is named for Hermes and implements two formats that are not Hermes. Whether that serves
those three goals is a fair question, and for one of them it has measured answers.

`hermes/protocol.py` holds Hermes 3 and Hermes 4. `hermes/atem.py` holds ATEM, which the previous
base spoke. `hermes/qwen35.py` holds the format the **current** base speaks. The current base is
`Qwen/Qwen3.8-27B`.

## The dialect is discovered, not chosen

Every time. A base model ships a `chat_template.jinja`, that template is what conditions the model,
and `hermes/base_model.json` records the dialect as evidence read off it. Hermes is upstream and
fixed; the model is what adapts.

For ATEM the reading was easy. `meta-models/Muse-Glimmer-30B` rendered calls as
`<atem:function_calls>` / `<atem:invoke>` / `<atem:parameter>`, returned results in
`<tool_output name="…">`, addressed deliberation to a `self` recipient, and contained no
`<tool_call>` and no `<think>`. Nothing about it could be mistaken for Hermes.

**Qwen3.8-27B is the hard case, and it is hard in the direction that hurts.** Its template writes
`<tool_call>`, `<tool_response>` and `<think>` — every one of them the Hermes spelling. Only the
payload differs:

```
Hermes 4        <tool_call>
                {"name": "terminal", "arguments": {"command": "ls"}}
                </tool_call>

qwen35          <tool_call>
                <function=terminal>
                <parameter=command>
                ls
                </parameter>
                </function>
                </tool_call>
```

A reviewer checking "does the template write `<tool_call>`?" gets **yes** and concludes Hermes.
That conclusion is wrong and its symptom is not silence. `hermes.protocol` finds `<tool_call>`,
hands `<function=terminal>` to a JSON decoder, and reports a malformed turn — on every turn, at
100%, in `malformed_turns`, which is one of the two numbers the promotion gate bounds. The model
gets blamed for the harness's mistake, and the number looks like a training problem.

This is why `Dialect.family` and `hermes.protocol.WIRE_MODULES` dispatch on a declared family
rather than on which tags a completion happens to contain. Tag-sniffing would be wrong here and
would look right.

## What forcing Hermes cost, measured

On the previous base. Instructing the pinned model in Hermes and parsing its output as Hermes:

| dialect at that base | malformed turns |
|---|---|
| `hermes-4` | **15 in 3 episodes** |
| `atem` | **0** across 19 tasks, 1.69 M tokens, ~300 calls |

At 15-in-3 the gate is measuring the mismatch between harness and model, and every miner surface is
scored through that noise. That is not a stylistic argument against Hermes-at-that-base; it is the
harness failing.

**The equivalent numbers for `qwen35` do not exist yet.** Nobody has served Qwen3.8-27B — see
[`serving-qwen3.8.md`](serving-qwen3.8.md). The reasoning above is read off the template and
verified by rendering real training rows through it
(`tests/test_corpus_template_qwen35.py`), not by generation. The prediction is that
`hermes-4`-at-this-base would score *worse* than 15-in-3, because a shared tag means the parser
gets far enough to produce an error on every turn instead of failing to find anything. It is a
prediction.

## Against each goal, honestly

### Stable harness — net better, and it has cost real debugging

ATEM took the harness from 15 malformed turns in 3 episodes to 0. It also opened two defect classes
Hermes would not have, because every adapter around `<tool_call>` JSON is more battle-tested:

- a serving layer's reasoning parser swallowed call markup, and 8 calls across 11 episodes were
  never executed, never counted, and not malformed either
- the training corpus rendered in the Hermes row shape could not be trained by the model's own
  template at all — and its third fault was silent

Both are fixed and both now have tests that assert on the artifact a consumer actually reads.
[`anatomy-of-an-attempt.md §7`](anatomy-of-an-attempt.md) has the numbers.

`qwen35` has already produced one defect of the same family, caught by a test rather than by a run:
the renderer emitted `\n\n</tool_response>` where the template emits `\n</tool_response>`, because
the template trims the result body and the renderer did not. Tool output almost always ends in a
newline, so that was a one-character divergence on essentially every tool result in every episode —
a distribution shift nothing else in the stack would have reported. It was found by asserting the
renderer's bytes against the template's bytes, which is now a test.

The honest accounting: matching the base's own format is the more stable choice *and* it keeps
costing debugging that Hermes would not.

### Faster learning — the argument holds and is structural

An SFT corpus rendered in a format the base does not speak teaches it to *unlearn* its own
template. On ATEM that was concrete: reasoning was written as `<think>` inside `content`, the
model's template ignores `content` on any turn carrying tool calls, and every reasoning block on
every tool-calling turn was dropped before the trainer saw it.

For `qwen35` the same class of fault exists with a nastier failure mode. Its template renders tool
definitions with `tool | tojson`, so a `tools` field holding **bare names** does not raise — it
renders `<tools>\n"terminal"\n</tools>` and trains the model on a tool block containing two quoted
words. The ATEM template raised on that input. This one does not, which makes the renderer's
refusal in `hermes.format` the only thing standing between the pipeline and a silent corruption.
That is pinned by a test that asserts the garbage, so if upstream ever starts raising, we find out.

### Less ecosystem breakage — this is where the previous base cost, and where this one may not

The real price of ATEM, which should not be minimised:

- a Hermes-runtime consumer could not run those weights
- vLLM 0.27.0 had no native support for the architecture (`vllm#51655`, open), and its
  `--model-impl transformers` fallback returned incoherent output
- SGLang worked from a branch (`sglang#34262`, open)

**This is the goal `qwen35` most plausibly improves**, and the reason is the same tag overlap that
makes it dangerous to detect. A Hermes-runtime consumer meeting `<tool_call>` and `<think>` gets
tags it recognises, and a Qwen release architecture is far likelier to have day-one engine support
than a one-off was. Neither of those is verified. Both are worth checking before being claimed —
`serving-qwen3.8.md` has the checklist.

## The alternative worth taking seriously, and why not

**Train toward Hermes** — keep the native dialect for baseline measurement, but render the SFT and
DPO rows in Hermes so each cycle closes the ecosystem gap instead of widening it.

To train Hermes rows against a base you must replace its chat template. The moment you fork the
template, the fine-tune stops honouring the base's serving contract — and `hermes.conformance`,
which checks that every marker the parser reads is still present in the *pinned* template, is no
longer describing what you serve. You would trade a serving-stack gap for a template fork, and a
template fork is the harder thing to verify.

There is a cheaper version of the same benefit. `Dialect` is data, not branches, and the harness
speaks three formats — so presenting a native-dialect model behind a Hermes-shaped endpoint is a
translation layer at the boundary, not a retraining. That buys ecosystem reach without asking the
weights to unlearn their own format. For `qwen35` that layer is unusually thin: the tags already
match and only the call payload has to be rewritten.

## Where it stands

Keep the base's own contract as both the measurement and the training dialect.
`hermes.promotion.check_graders` already refuses to compare runs across dialects, and the corpus
renders into the channel the model's template reads.

Close the ecosystem gap at the boundary rather than in the weights.

Two caveats worth stating plainly:

`Serving` records precision, device, engine and sampling, and `check_graders` refuses a
cross-dialect comparison — but `Serving` itself does not record the dialect. The refusal works off
the episode metrics rather than off the serving record. That is enough while both come from the
same run, and it is worth tightening before anyone compares runs recorded by different operators.

And the dialect that most needs that tightening is this one. Comparing an `atem` run against a
`hermes-4` run is at least *visibly* a category error. Comparing a `qwen35` run against a
`hermes-4` run is not: the transcripts contain the same tags, and only the payloads differ.
