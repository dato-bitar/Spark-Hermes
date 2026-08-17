# spark-hermes-3.8-27b

`Spark-Hermes-3.8-27B` is the model this pipeline produces. The base is
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), pinned in
[`hermes/base_model.json`](../../base_model.json) at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, and `tests/test_base_model.py` keeps every stage in
this directory in step with that pin.

| stage | what it teaches | from |
|---|---|---|
| `stage-a-reasoning.yaml` | reasoning traces | the pinned base |
| `stage-b-hermes.yaml` | agentic shape, mixed corpus | the pinned base |
| `stage-c-tools.yaml` | tool mechanics, executed trajectories only | the pinned base |
| `stage-d-preference.yaml` | DPO over the competition's own near misses | `outputs/spark-hermes-3.8-27b/stage-c/merged` |

```bash
scripts/train.sh     hermes/recipes/spark-hermes-3.8-27b/stage-c-tools.yaml
scripts/merge_lora.sh hermes/recipes/spark-hermes-3.8-27b/stage-c-tools.yaml
scripts/train.sh     hermes/recipes/spark-hermes-3.8-27b/stage-d-preference.yaml
```

## Three things these recipes get right on purpose

**LoRA over a bf16 base, not QLoRA over a 4-bit one.** The served model is bf16, so adapters
fitted to a quantized copy are adapters for a model nobody runs — and the mismatch is silent,
because they merge back into bf16 either way. QLoRA is the right answer when memory is the
constraint; on a 96 GB card holding ~56 GB of bf16 weights it is not.

**Stage D starts from the merge, not from stage C's adapter.** DPO needs a reference policy, and
with LoRA that comes free — disable the adapter and the base is the reference — but only if the
base *is* the policy you are improving. After A/B/C that policy is the merged model, so chaining
onto stage C's adapter would quietly optimise against a policy three stages out of date.

**`chat_template` names the model's own file.** This base does not speak Hermes, and unlike the
previous one it does not look foreign either — it writes `<tool_call>` and `<think>`, exactly the
Hermes tags, and differs only in what goes *inside* the call tag: `<function=NAME>` with one
`<parameter=KEY>` element each, where Hermes puts a JSON object. That dialect is `qwen35`, it is
implemented in [`hermes/qwen35.py`](../../qwen35.py), and naming a Hermes template here would
train the wrong wire format into the one place a model cannot be corrected from afterwards.

The tag overlap is why this is called out rather than assumed. A reviewer checking "does the
template write `<tool_call>`?" gets yes and concludes Hermes, and every symptom afterwards points
at the model: not silence, but a *malformed* call on every single turn.

## What is not pinned yet

`sequence_len` is 8192 in every stage, which is a floor rather than a measurement. The 190-episode
baseline had a median of 34 turns, so agentic episodes are long and this is the number most likely
to need raising — after measuring headroom on the card, not before.
