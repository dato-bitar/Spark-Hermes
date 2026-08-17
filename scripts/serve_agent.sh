#!/usr/bin/env bash
# Serve the agent benchmark's model on SGLang.
#
#   scripts/serve_agent.sh /path/to/model [served-name] [port]
#   python -m hermesbench.runner --suite all --model <served-name> --base-url http://127.0.0.1:8001/v1
#
# SGLang rather than vLLM, and that is a measurement rather than a preference -- but the
# measurement was taken against the PREVIOUS base. On 2026-08-11, against Muse-Glimmer-30B,
# vLLM 0.27.0 had no native support for the architecture and its `--model-impl transformers`
# fallback served the model while returning ten tokens of multilingual noise; SGLang from the
# muse-glimmer branch returned correct tool calls. Plain transformers on the same weights,
# revision and card agreed with SGLang, so the fallback was what was broken.
# docs/serving-muse-glimmer.md has that evidence and the eight startup failures that preceded it.
#
# NONE OF THAT HAS BEEN RE-VERIFIED for the current base, Qwen/Qwen3.8-27B. It is a different
# architecture (Qwen3_5ForConditionalGeneration) on a different engine version, so the flags below
# are a starting point and not a recipe. docs/serving-qwen3.8.md records what is known and what is
# not; hermes/base_model.json says the same in its `serving` field.
#
# For a Hermes-dialect model this script works unchanged -- drop the parser flags, which are what
# teach SGLang a non-Hermes wire format.
#
# NOT the TritonBench stack. `scripts/install_serve.sh` pins vLLM 0.25.0+cu129 deliberately: the
# Triton domain score is only comparable across miners if every checkpoint is served by the same
# engine, and every published Triton number was measured on that one. Switching it would invalidate
# them. The two paths serve different benchmarks and are meant to stay apart.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:?usage: scripts/serve_agent.sh <model-path-or-repo> [served-name] [port]}"
SERVED="${2:-$(basename "$MODEL")}"
PORT="${3:-8001}"
VENV="${SGLANG_VENV:-$ROOT/.venv-sglang}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "error: no SGLang venv at $VENV" >&2
  echo "  python3 -m venv $VENV" >&2
  echo "  SGLANG_BUILD_RUST_EXTS=none $VENV/bin/pip install sglang" >&2
  echo "  $VENV/bin/pip install ninja        # SGLang JIT-compiles kernels through it" >&2
  echo "  apt-get install -y ffmpeg          # torchcodec dlopens libavutil" >&2
  echo "See docs/serving-qwen3.8.md for the current base; docs/serving-muse-glimmer.md records" >&2
  echo "the previous one, where the install was a branch build." >&2
  exit 1
fi

# The venv's own CUDA toolchain, and only that one. Pointing CUDA_HOME at a different venv's copy
# made flashinfer's bundled cccl headers disagree with nvcc and every JIT compile failed with
# "CUDA compiler and CUDA toolkit headers are incompatible". CPATH is deliberately not set for the
# same reason: adding the toolkit's includes is what let the bundled headers be found at all.
CUDA_DIR="$(cd "$VENV" && "$VENV/bin/python" -c "
import pathlib, sysconfig
site = pathlib.Path(sysconfig.get_paths()['purelib'])
found = sorted((site / 'nvidia').glob('cu*'))
print(found[-1] if found else '')" 2>/dev/null || true)"

if [ -n "$CUDA_DIR" ] && [ -d "$CUDA_DIR" ]; then
  export CUDA_HOME="$CUDA_DIR"
  export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  # The wheel ships libcudart.so.13 and no bare .so, which `ld -lcudart` cannot find, so every JIT
  # link failed. A versioned library alone is not linkable.
  for lib in "$CUDA_HOME"/lib/libcudart.so.*; do
    [ -e "$CUDA_HOME/lib/libcudart.so" ] || ln -sf "$(basename "$lib")" "$CUDA_HOME/lib/libcudart.so"
    break
  done
fi

# The venv's bin too. Invoking $VENV/bin/python by absolute path leaves the venv's bin off PATH, so
# a pip-installed ninja is present and invisible -- which killed the scheduler three times with
# FileNotFoundError.
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$VENV/bin:$PATH"

ARGS=(
  --model-path "$MODEL"
  --served-model-name "$SERVED"
  --context-length "${SERVE_CONTEXT_LENGTH:-32768}"
  --mem-fraction-static "${SERVE_MEM_FRACTION:-0.88}"
  --host 127.0.0.1 --port "$PORT"
  # Neither backend needs a CUDA toolchain. The default flashinfer paths JIT-compile at startup and
  # again on the first request: attention alone was not enough, because the server came up, served
  # its warmup, and died on the first real completion inside the sampling kernel.
  --attention-backend "${SERVE_ATTENTION_BACKEND:-triton}"
  --sampling-backend "${SERVE_SAMPLING_BACKEND:-pytorch}"
  --trust-remote-code
)

# A tool-call parser is not optional for a non-Hermes dialect, and the reason is worse than it
# sounds: without one SGLang does not render the tool definitions into the prompt AT ALL. The first
# working ATEM run came back with prompt_tokens=77 and the model wondering aloud which command to
# use, because it had never been told it had any tools. That reads as a model that will not call
# tools, and it scores as one.
#
# `muse` is the pair that was verified for ATEM. For qwen35 the correct names are NOT known here --
# they depend on the SGLang build, and guessing produces either a startup failure or, far worse, the
# prompt_tokens=77 silence above. So this refuses rather than defaulting: run
# `$VENV/bin/python -m sglang.launch_server --help` and read the parser choices off the build you
# actually have.
case "${SERVE_DIALECT:-qwen35}" in
  hermes*)
    ;;
  atem)
    ARGS+=(--tool-call-parser "${SERVE_TOOL_PARSER:-muse}" --reasoning-parser "${SERVE_REASONING_PARSER:-muse}")
    ;;
  *)
    if [ -z "${SERVE_TOOL_PARSER:-}" ]; then
      echo "error: SERVE_DIALECT=${SERVE_DIALECT:-qwen35} needs SERVE_TOOL_PARSER set explicitly." >&2
      echo "  Without a tool-call parser SGLang omits the tool definitions from the prompt entirely," >&2
      echo "  and the model looks like one that never calls tools. This has NOT been verified for" >&2
      echo "  Qwen3.8-27B -- see docs/serving-qwen3.8.md. Find the name your build offers:" >&2
      echo "    $VENV/bin/python -m sglang.launch_server --help | grep -A5 tool-call-parser" >&2
      echo "  then re-run with SERVE_TOOL_PARSER=<name> [SERVE_REASONING_PARSER=<name>]." >&2
      exit 2
    fi
    ARGS+=(--tool-call-parser "$SERVE_TOOL_PARSER")
    [ -n "${SERVE_REASONING_PARSER:-}" ] && ARGS+=(--reasoning-parser "$SERVE_REASONING_PARSER")
    ;;
esac

echo "serving $MODEL as $SERVED on 127.0.0.1:$PORT (sglang)" >&2
exec "$VENV/bin/python" -m sglang.launch_server "${ARGS[@]}"
