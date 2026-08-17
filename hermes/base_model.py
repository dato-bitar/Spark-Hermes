"""The base model, pinned to a revision rather than to a name.

The base is `Qwen/Qwen3.8-27B`. It replaced `meta-models/Muse-Glimmer-30B`, which was never
the target: Muse-Glimmer was a development stand-in taken up while `Qwen/Qwen3.8-27B` was
announced and unpublished, when the only repositories under that name were third-party
derivatives with no official base. That is no longer the situation -- the official repository
published on 2026-08-14 and is what this pin names.

The history is kept here rather than deleted because the reason for the stand-in was a rule,
not an accident: a repository that cannot be pinned or verified should not be trained against.
The rule did not change. The fact it was applied to changed.

**A name is not a pin.** `base_model: Qwen/Qwen3.8-27B` resolves to whatever that repository
holds when someone runs it. `eval.hf_pin` already refuses movable refs on the mining side for
exactly that reason; the base model was the one place still naming a repository without saying
which commit of it. Two runs that agree on every other digest this project computes could still
have trained on different weights.

## What the pin records, and why each field is here

`revision` is the whole point -- a 40-character commit, checked by `eval.hf_pin`.

`hermes_dialect` is recorded with its evidence rather than asserted, and for this base the
evidence is a trap rather than merely inconvenient. The repository's own chat template writes
`<tool_call>` and `<think>` -- both Hermes tags -- so a reader checking for them concludes the
model speaks Hermes and stops. It does not. What sits *inside* `<tool_call>` is one element per
parameter:

    Hermes 4        <tool_call>{"name": "terminal", "arguments": {"command": "ls"}}</tool_call>
    QWEN35          <tool_call>
                    <function=terminal>
                    <parameter=command>
                    ls
                    </parameter>
                    </function>
                    </tool_call>

`hermes.protocol` finds `<tool_call>`, hands the remainder to a JSON decoder, and gets a syntax
error on every single turn. So the failure mode is not "no calls found" but "every call
malformed" -- which reads as a model that cannot follow the format, in the metric the promotion
gate bounds. `hermes/qwen35.py` exists for that reason, the same reason `hermes/atem.py` does,
and the dialect is established by reading the model rather than by choosing for it. The project's
rule is that Hermes is upstream and the model is what adapts.

`multimodal: true` is recorded because it is easy to miss and changes things. This is a
`Qwen3_5ForConditionalGeneration` with a vision tower, so "27B" is not 27B of text
parameters, the text hyperparameters live under `text_config`, and any memory estimate
that reads the top level of `config.json` silently gets `None` for every field.

`kv_bytes_per_token` is derived from those hyperparameters and stored so a budget can be
checked without a network call. **Over the full-attention layers only.** This model is hybrid:
`layer_types` is 48 `linear_attention` + 16 `full_attention` -- one full layer every
`full_attention_interval` (4) -- so only 16 of the 64 layers keep a per-token KV cache and the
other 48 hold a fixed per-sequence recurrent state. Deriving it as
`2 x 64 layers x 4 KV heads x 256 head_dim` -- every layer -- overstates the per-token cost by 4x.

That is more than a footnote because `kv_bytes` sizes a VRAM budget against a card, so a 4x
overstatement rules out hardware that runs this model comfortably: at fp8 and a 200K peak
context the honest number is ~6.6 GB, not ~26 GB. On the previous base the test asserting the
all-layers formula passed the entire time, because it and the pin agreed with each other and
neither described the model -- the failure mode a derived-and-stored constant invites.

**`kv_bytes_measured` is null here, and that is not an oversight.** It was populated on the
previous pin because someone served that model and read the figure off SGLang's own KV pool
printout -- which is what caught the 4x error. Nobody has served this pin. The previous base's
numbers describe a different layer count, KV-head count and head_dim, and carrying them across
would assert evidence that does not exist, in the one field whose entire value is that it was
measured. Stale and fresh look identical once written down. So the derivation stands unconfirmed
and says so.

A budget that says `input_tokens: 262144` without saying whether that is cumulative or peak
context cannot be checked against a card at all -- the two readings differ by more than an
order of magnitude.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PIN_PATH = Path(__file__).resolve().parent / "base_model.json"


class BaseModelError(ValueError):
    """The base-model pin is missing, malformed, or names a ref that cannot be pinned."""


@dataclass(frozen=True)
class BaseModel:
    repository: str
    revision: str
    hermes_dialect: str
    chat_template: str
    multimodal: bool
    kv_bytes_per_token: dict[str, int]
    raw: dict[str, Any]

    @property
    def at_revision(self) -> str:
        """How to name this model anywhere a human will read it."""
        return f"{self.repository}@{self.revision[:12]}"

    def kv_bytes(self, context_tokens: int, *, dtype: str = "fp8") -> int:
        """KV cache bytes for a peak context, so a budget can be checked against a card."""
        per_token = self.kv_bytes_per_token.get(dtype)
        if per_token is None:
            raise BaseModelError(f"no KV size recorded for dtype {dtype!r}; have {sorted(self.kv_bytes_per_token)}")
        return per_token * context_tokens


def load(path: Path | None = None) -> BaseModel:
    """Read the pin and refuse anything that is not actually pinned."""
    from eval.hf_pin import check_revision

    source = path or PIN_PATH
    if not source.is_file():
        raise BaseModelError(f"no base-model pin at {source}")
    record = json.loads(source.read_text(encoding="utf-8"))

    repository = str(record.get("repository") or "")
    if "/" not in repository:
        raise BaseModelError(f"repository {repository!r} is not an org/name Hub id")

    # The same refusal the mining side already makes. A movable ref here would mean two
    # runs agreeing on every other digest and still training on different weights.
    issues = check_revision(record.get("revision"), field="base model revision")
    if issues:
        raise BaseModelError("; ".join(issues))

    return BaseModel(
        repository=repository,
        revision=str(record["revision"]),
        hermes_dialect=str(record.get("hermes_dialect") or ""),
        chat_template=str(record.get("chat_template") or ""),
        multimodal=bool(record.get("multimodal")),
        kv_bytes_per_token={k: int(v) for k, v in (record.get("kv_bytes_per_token") or {}).items()},
        raw=record,
    )


__all__ = ["PIN_PATH", "BaseModel", "BaseModelError", "load"]
