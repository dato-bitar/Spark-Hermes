"""The base model is pinned to a revision, and the recipes agree with the pin."""

import json
from pathlib import Path

import pytest
import yaml

from hermes.base_model import PIN_PATH, BaseModelError, load
from hermes.merge import MERGED_DIRNAME

RECIPE_DIR = Path("hermes/recipes/spark-hermes-3.8-27b")


def _recipes():
    return sorted(RECIPE_DIR.glob("stage-*.yaml"))


def test_the_pin_loads_and_names_a_real_commit():
    pin = load()
    assert pin.repository == "Qwen/Qwen3.8-27B"
    assert len(pin.revision) == 40


def test_a_movable_ref_is_refused(tmp_path):
    """`main` resolves to whatever the repository holds when someone runs it. Two runs
    could then agree on every other digest this project computes and still have trained on
    different weights -- the same refusal eval.hf_pin already makes on the mining side."""
    bad = tmp_path / "pin.json"
    bad.write_text(json.dumps({**json.loads(PIN_PATH.read_text()), "revision": "main"}))
    with pytest.raises(BaseModelError):
        load(bad)


def test_a_short_sha_is_refused(tmp_path):
    bad = tmp_path / "pin.json"
    bad.write_text(json.dumps({**json.loads(PIN_PATH.read_text()), "revision": "6a9e13bd6fc8"}))
    with pytest.raises(BaseModelError):
        load(bad)


# A stage may start from an earlier stage's merged output instead of from the hub -- DPO needs the
# policy it is improving as its reference, and after the SFT stages that policy is the merged model,
# not the raw base. Such a recipe names a local path, so there is nothing to pin it against
# directly; what keeps it honest is that the path is one some other stage in this line actually
# produces, and every one of those is pinned. The chain therefore roots at the pin.
def _is_derived(base_model: str) -> bool:
    return base_model.startswith("outputs/")


def _merge_outputs() -> set[str]:
    """Every path a merge of a stage in this line would write to."""
    out = set()
    for recipe in _recipes():
        config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        if config.get("output_dir") and not _is_derived(str(config["base_model"])):
            out.add(f"{config['output_dir'].rstrip('/')}/{MERGED_DIRNAME}")
    return out


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.name)
def test_every_recipe_matches_the_pin(recipe):
    """A recipe drifting from the pin is the failure the pin exists to prevent, and it
    would be invisible: both files would still name a real model."""
    pin = load()
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    base = str(config["base_model"])
    if _is_derived(base):
        pytest.skip("derived base; covered by test_a_derived_base_stays_inside_this_model_line")
    assert base == pin.repository
    assert config["base_model_revision"] == pin.revision


@pytest.mark.parametrize("recipe", _recipes(), ids=lambda p: p.name)
def test_a_derived_base_stays_inside_this_model_line(recipe):
    """The escape hatch above, held shut.

    `outputs/` as a prefix is not itself an assurance -- it would admit any other line's checkpoint,
    or one built from an unpinned base, and the resulting model would still train and still serve.
    So the path must be one that merging a pinned stage of this same line actually produces:
    that stage's `output_dir` with Axolotl's `merged` appended. A path nothing produces is a
    recipe that cannot run, and one produced by an unpinned stage is off the pin by a hop.

    It must also carry no `base_model_revision`: a local directory has no hub revision, and a
    stamped one would agree with the pin while describing something the pin never produced."""
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    base = str(config["base_model"])
    if not _is_derived(base):
        pytest.skip("hub base; covered by test_every_recipe_matches_the_pin")
    produced = _merge_outputs()
    assert base in produced, f"{base!r} is not produced by any pinned stage here; those write {sorted(produced)}"
    assert "base_model_revision" not in config, "a local path has no hub revision to stamp"


def test_at_most_one_stage_starts_from_a_derived_base():
    """Named rather than counted loosely, so a second recipe going off-pin fails here instead of
    silently joining the exception."""
    derived = {
        r.name for r in _recipes() if _is_derived(str(yaml.safe_load(r.read_text(encoding="utf-8"))["base_model"]))
    }
    assert derived == {"stage-d-preference.yaml"}


def test_the_pin_names_the_vendors_own_repository():
    """This replaced a test forbidding `qwen3.8` in any recipe, which was correct until
    2026-08-14 and is now the opposite of correct: Qwen3.8-27B is the base.

    The RULE that test encoded did not expire with it. What made the name unusable was never the
    version -- it was that the only repositories carrying it were third-party derivatives with no
    official base, which cannot be verified and should not be trained against. So the check moves
    to the thing that actually mattered: the pin names the vendor's own namespace, not a re-upload
    of it. `TheBloke/Qwen3.8-27B-GGUF` and `someone/qwen3.8-27b-merged` both pin to a real
    40-character commit and would sail through every other test in this file.

    Deleting the old test outright would have been the easy move and would have removed the only
    thing standing between this pin and a mirror."""
    pin = load()
    org, _, name = pin.repository.partition("/")
    assert org == "Qwen", (
        f"the pin names {pin.repository!r}. A derivative or a mirror pins just as well as the "
        "original and trains just as silently; the vendor namespace is what says it is the base."
    )
    assert name == "Qwen3.8-27B"


def test_the_dialect_is_recorded_with_its_evidence():
    """Hermes is upstream; the model is what adapts. So the dialect is established by
    reading the model's own template, not chosen for it -- and this base does not speak Hermes.

    For the previous base the evidence could be "it writes `<atem:` and no `<tool_call>`", and
    naming the absent Hermes tags was the decisive half. This base writes `<tool_call>` AND
    `<think>`, so that half of the argument is gone: the tags agree with Hermes and the payload
    does not. Evidence that only listed tags would therefore *support the wrong conclusion* here,
    which is why it has to name the element form explicitly and say that the overlap exists."""
    pin = load()
    assert pin.hermes_dialect == "qwen35"
    evidence = pin.raw["hermes_dialect_evidence"]
    for marker in ("<tool_call>", "<function=", "<parameter=", "<tool_response>", "<think>"):
        assert marker in evidence, marker
    # The distinguishing claim, not just the tag list. Without this a pin could quote the tags,
    # every one of which Hermes also writes, and read as evidence FOR hermes-4.
    assert "does not speak Hermes" in evidence
    assert "one element per parameter" in evidence.lower()


def test_the_dialects_evidence_survives_the_tag_overlap():
    """The specific trap this base sets, asserted rather than trusted to a reader.

    Every tag in the evidence string is one Hermes writes too. So the pin must state that fact --
    a later editor trimming the evidence down to "it writes <tool_call> and <think>" would leave
    something that reads as a Hermes model and passes a tag-based review."""
    evidence = load().raw["hermes_dialect_evidence"].lower()
    assert "tag overlap" in evidence or "shares hermes" in evidence
    assert "json" in evidence, "the evidence has to say what is NOT there, which is the JSON object"


def test_the_dialect_names_one_this_repo_implements():
    from hermes.protocol import DIALECTS

    assert load().hermes_dialect in DIALECTS


def test_multimodal_is_recorded_because_it_is_easy_to_miss():
    """A Qwen3_5ForConditionalGeneration keeps its text hyperparameters under `text_config`;
    the top level of config.json carries only vision and projector keys, so anything reading it
    for hidden_size or num_hidden_layers gets None and computes a memory budget out of nothing.

    Three base models in a row have had this shape, which is why it is asserted rather than noted.
    `AutoModelForCausalLM` also refuses this config outright -- the class is
    `AutoModelForImageTextToText`, and finding that out cost a load attempt."""
    pin = load()
    assert pin.multimodal is True
    assert pin.raw["text_config"]["num_hidden_layers"] == 64
    assert pin.raw["text_config"]["num_key_value_heads"] == 4
    assert pin.raw["text_config"]["head_dim"] == 256
    assert "hidden_size" not in {k for k in pin.raw if k != "text_config"}


def test_kv_bytes_count_only_the_full_attention_layers():
    """This model is hybrid, so its KV cache lives in 16 of 64 layers, not all of them.

    An earlier version of this test asserted `2 * num_hidden_layers * kv_heads * head_dim` and
    passed -- which is exactly how a wrong figure shipped once. The formula and the pin agreed
    with each other, and neither described the model. `layer_types` in the published config is
    48 `linear_attention` + 16 `full_attention`, one full layer every `full_attention_interval`;
    a linear layer holds a fixed-size recurrent state per sequence rather than growing with the
    context, so it belongs in a bounded per-sequence budget and not in a figure that gets
    multiplied by a context length.

    Three bases in a row have been hybrid, by two different mechanisms -- sliding windows on one,
    linear attention on the others. The mechanism differs and the arithmetic lesson is identical,
    which is why this test survives the base model changing underneath it.

    Getting it wrong is not cosmetic: `kv_bytes()` sizes a VRAM budget against a card, and
    counting all 64 layers overstates the per-token cost by 4x, which rules out hardware that in
    fact runs this model comfortably.
    """
    pin = load()
    t = pin.raw["text_config"]
    counts = t["layer_types_counts"]
    full = counts["full_attention"]

    assert sum(counts.values()) == t["num_hidden_layers"]
    assert full == t["num_hidden_layers"] // t["full_attention_interval"]

    # The bounded half needs its bound recorded, whatever the mechanism. For a sliding-window
    # model that was `sliding_window`; here it is the linear layers' own state dimensions, and
    # asserting a field named for one mechanism would have quietly skipped this on the other.
    for field in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"):
        assert t[field] > 0, f"a bounded layer needs its bound recorded; {field} is missing"

    expected = 2 * full * t["num_key_value_heads"] * t["head_dim"]
    assert pin.kv_bytes_per_token["bf16"] == expected * 2
    assert pin.kv_bytes_per_token["fp8"] == expected

    # Name the figure the all-layers formula would produce, so an edit that reintroduces it
    # fails here rather than silently quadrupling every budget again.
    all_layers = 2 * t["num_hidden_layers"] * t["num_key_value_heads"] * t["head_dim"] * 2
    assert pin.kv_bytes_per_token["bf16"] != all_layers
    assert all_layers == pin.kv_bytes_per_token["bf16"] * 4


def test_the_measurement_is_null_until_someone_actually_takes_one():
    """The derivation for this base is UNCONFIRMED, and the pin has to say so.

    On the previous base this field held a real SGLang reading that agreed with the derivation to
    the byte -- and the only reason anyone trusted the derivation was that the reading existed.
    Nobody has served this pin. Carrying the old numbers forward would have been a one-line edit
    that survived every other test in this file, because a measurement of one model and a
    measurement of another are the same shape.

    So the assertion is that the field is null AND that the pin explains why, rather than looking
    like a field somebody forgot to fill in. `kv_bytes()` still works -- the derivation is the
    best available number and is almost certainly right -- but a reader sizing hardware off it
    deserves to know which kind of number it is."""
    pin = load()
    assert pin.raw["kv_bytes_measured"] is None
    why = pin.raw["kv_bytes_measured_note"].lower()
    assert "not been confirmed" in why or "unconfirmed" in why
    # Naming the previous base, so the next person to fill this in cannot do it by copying.
    assert "muse-glimmer" in why


def test_kv_bytes_for_a_peak_context():
    pin = load()
    # 262K tokens -- this model's native context -- of fp8 KV over the 16 full-attention layers is
    # ~8.6 GB, which fits beside ~56 GB of bf16 weights on a 96 GB card. Under an all-64-layers
    # figure it would read ~34 GB, and 56 + 34 does not fit: that is the arithmetic by which a 4x
    # error rules out hardware that in fact runs the model.
    assert round(pin.kv_bytes(262_144, dtype="fp8") / 1e9, 1) == 8.6
    assert round(pin.kv_bytes(131_072, dtype="fp8") / 1e9, 1) == 4.3
    assert pin.kv_bytes(262_144, dtype="bf16") == 2 * pin.kv_bytes(262_144, dtype="fp8")


def test_the_parameter_count_comes_from_the_index_not_the_model_card():
    """ "27B" is a rounded marketing figure; the pin is what a memory budget is computed from.

    Cross-checked against the shard total so an edited digit fails here: bf16 weights are two
    bytes per parameter, so the recorded count and `safetensors_shards` have to be consistent with
    a real 55.6 GB download."""
    pin = load()
    assert pin.raw["parameters"] == 27_781_427_952
    assert pin.raw["safetensors_shards"] == 18
    bf16_bytes = pin.raw["parameters"] * 2
    assert round(bf16_bytes / 1e9) == 56, "the bf16 footprint the recipes' LoRA-not-QLoRA note assumes"


def test_an_unknown_dtype_is_refused_rather_than_assumed():
    with pytest.raises(BaseModelError, match="no KV size recorded"):
        load().kv_bytes(1000, dtype="int2")
