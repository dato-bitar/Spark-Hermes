# Serving Qwen3.8-27B

**Nothing in this document has been measured.** That is the point of it existing.

[`docs/serving-muse-glimmer.md`](serving-muse-glimmer.md) is the model for what this page should
eventually be: a table of engines that were actually run, the eight startup failures that preceded
the working one, and a KV-pool printout that contradicted a derived constant and corrected it by 4x.
Every line of it was earned against a live server.

This page has none of that yet, because nobody has served
`Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. What follows is what the pinned
config says, what it implies, and — most usefully — which of the previous base's lessons do *not*
transfer.

## What is known, from the pin only

| | value | source |
|---|---|---|
| architecture | `Qwen3_5ForConditionalGeneration` | `config.json` |
| class | `AutoModelForImageTextToText` | it is image-text-to-text; `AutoModelForCausalLM` refuses this config |
| weights | 27.78 B params, 18 safetensors shards, ~55.6 GB bf16 | `model.safetensors.index.json` |
| layers | 64: 48 `linear_attention` + 16 `full_attention`, one full every 4 | `text_config.layer_types` |
| KV heads / head_dim | 4 / 256 | `text_config` |
| native context | 262,144 | `text_config.max_position_embeddings` |
| declared transformers | `5.8.0.dev0` | `config.json` |
| wire dialect | `qwen35` | its own `chat_template.jinja` — see [wire-dialects.md](wire-dialects.md) |

**Derived, not measured:** KV cache is 32,768 B/token at fp8 over the 16 full-attention layers
only, so a 262K peak context is ~8.6 GB beside ~56 GB of bf16 weights. The 48 linear-attention
layers hold a fixed per-sequence recurrent state and do not scale with context, so they are not in
that figure.

That derivation is exactly the kind that was **wrong by 4x on the previous base until a live server
contradicted it**. It has the same shape and it has not been checked. `kv_bytes_measured` in
`hermes/base_model.json` is `null` and should stay null until someone pastes a real number in.

## The first thing to do

Serve it, then read SGLang's own KV pool printout at startup and record it. Two numbers matter:

- the **full-attention pool** — tokens and GiB, which gives implied bytes/token and either confirms
  32,768 or does not
- whatever it reports for the **linear/recurrent state** — which must NOT be added to the per-token
  figure, because it is per-sequence

On the previous base the server accounted for the two halves separately and summing them reproduced
exactly the wrong all-layers number. Expect the same trap in a different shape.

## What does not transfer from Muse-Glimmer

This is the part worth reading before copying a command line.

**The branch build does not apply.** Muse-Glimmer needed SGLang from the `muse-glimmer` branch
because no release had the architecture. Qwen3.5 is a Qwen release architecture and the Hub tags
advertise vLLM and SGLang compatibility. Whether the *installed* version has it is a different
question and is the one to check — but starting from "I need a branch build" is starting from the
previous model's problem.

**Do not reuse `--tool-call-parser muse`.** Those parsers exist to teach SGLang the ATEM wire
format. This model does not speak ATEM. `scripts/serve_agent.sh` now refuses to start for this
dialect unless `SERVE_TOOL_PARSER` is set explicitly, rather than defaulting to something plausible.

That refusal is deliberate and the reason is the worst failure mode on this page: **without a
tool-call parser SGLang does not put the tool definitions in the prompt at all.** The first working
ATEM run came back with `prompt_tokens=77` and the model wondering aloud which command to use,
because it had never been told it had any tools. That does not look like a serving bug. It looks
like a model that will not call tools, and it scores as one.

```bash
$VENV/bin/python -m sglang.launch_server --help | grep -A5 tool-call-parser
```

**Do not reuse the transformers pin.** `docs/serving-muse-glimmer.md` says do not upgrade
transformers, because SGLang pinned 5.12.1 and shipped its own `muse_glimmer` config. This config
declares `5.8.0.dev0`. The two constraints are unrelated and the older advice is about a different
dependency graph.

## What probably does transfer

Mechanical, engine-level lessons rather than model-level ones — worth trying first if the server
will not start:

- `--attention-backend triton --sampling-backend pytorch` avoided flashinfer's startup *and*
  first-request JIT compiles. The server otherwise came up, served its warmup, and died on the
  first real completion inside the sampling kernel.
- the venv's own CUDA toolchain and only that one; a versioned `libcudart.so.13` with no bare `.so`
  is not linkable, which `scripts/serve_agent.sh` works around.
- ninja must be on `PATH`, not merely installed.

## Checklist before this page is worth trusting

- [ ] server starts, and the engine + version is recorded here
- [ ] KV pool printout captured; `kv_bytes_measured` filled in or the derivation corrected
- [ ] a real generation produces a `<tool_call>` / `<function=` call that `hermes.qwen35` parses
- [ ] `prompt_tokens` on a task with tools is large enough that the definitions clearly reached the
      prompt — the `77` check
- [ ] `hermesbench` malformed-turn rate over a real suite, which is the number that says the
      dialect is right
