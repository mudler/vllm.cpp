#!/usr/bin/env bash
# #2864 -- the limb-3 VEHICLE SEARCH, re-runnable.
#
# Limb 3 of the ratified near-tie methodology asks for a strict token-exact pass
# on a bigger DENSE model through the SAME forward code the Qwen3.8-27B Q4_K_M
# ROCm arm runs. A vehicle therefore has to satisfy six conditions at once:
#
#   1. a GGUF that stores K-QUANT tensors  (the tier this arm's kernels serve)
#   2. an architecture OUR gguf dispatch has an arm for
#   3. an architecture the PINNED vLLM registers   (limb 3 is scored against the
#      primary oracle, so a model vLLM cannot load has no denominator)
#   4. DENSE, not MoE
#   5. it FITS the board: 65536 MiB firmware VRAM carve, 59934 MiB free before
#      load, 62 GiB host RAM (docs/bench-evidence/rocm-strix-qwen38-q4km-20260901.md)
#   6. NOT the arm's own artifact, because limb 3 exists to corroborate the arm
#      from outside it
#
# This script measures 1, 2, 3 and 4 and prints 5 as the file size. It runs
# entirely off the fleet's own staged bytes and needs no GPU and no lease: it
# reads GGUF headers, this tree's dispatch table, and the pinned vLLM checkout.
#
# NO MODEL IS RUN and NO NUMBER HERE IS A PERFORMANCE RESULT. This arm's
# declared token gate reads FAIL and AGENTS.md Gates admits no performance
# result from it (#2497 already carries one retraction for exactly that).
set -uo pipefail

TREE="${TREE:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VLLM_PIN_CHECKOUT="${VLLM_PIN_CHECKOUT:-$HOME/_git/vllm}"
VLLM_PIN=5559679229bc961848b121ccdeaa8fa5d79bec98
NAS="${NAS:-/mnt/nas_share}"
HDR="$(dirname "$0")/gguf_header.py"

echo "===== 0. identity ====="
date -u +%FT%TZ
echo "tree             = $TREE"
echo "tree_head        = $(git -C "$TREE" rev-parse HEAD 2>&1)"
echo "vllm_checkout    = $VLLM_PIN_CHECKOUT"
echo "vllm_head        = $(git -C "$VLLM_PIN_CHECKOUT" rev-parse HEAD 2>&1)"
echo "vllm_pin_wanted  = $VLLM_PIN"
if [ "$(git -C "$VLLM_PIN_CHECKOUT" rev-parse HEAD 2>/dev/null)" = "$VLLM_PIN" ]; then
  echo "VLLM_CHECKOUT_IS_THE_PIN = yes"
else
  echo "VLLM_CHECKOUT_IS_THE_PIN = NO -- every registry answer below is about the wrong tree"
fi

echo
echo "===== 1. CONDITION 2: the GGUF architectures OUR build dispatches on ====="
# src/vllm/entrypoints/model_loader.cpp kGgufArchArms, with the four names that
# reach it through a constant resolved from their own headers. Printed from the
# tree so it cannot go stale against a transcription.
sed -n '/^constexpr GgufArchArm kGgufArchArms\[\] = {/,/^};/p' \
  "$TREE/src/vllm/entrypoints/model_loader.cpp"
echo "-- the four constants, resolved --"
grep -rhn 'inline constexpr const char\* k[A-Za-z0-9]*GgufArch = ' "$TREE/include" | sort

echo
echo "===== 2. CONDITION 1 precondition: the ggml type ids our reader ACCEPTS ====="
# GgmlTraits() throws "gguf: unknown ggml type id N" for anything absent, so an
# artifact carrying one unaccepted id is refused WHOLE. This is #2510's real
# predicate.
ACCEPTED=$(grep -oE '^    case [0-9]+: \{' \
  "$TREE/src/vllm/model_executor/model_loader/gguf_reader.cpp" \
  | grep -oE '[0-9]+' | sort -n | tr '\n' ' ')
echo "ACCEPTED_GGML_TYPE_IDS = $ACCEPTED"
for id in 3 5 7 9 15 21 29 31 34 35; do
  case " $ACCEPTED " in
    *" $id "*) echo "  id $id ACCEPTED" ;;
    *)         echo "  id $id REFUSED  (GgmlTraits throws)" ;;
  esac
done

echo
echo "===== 3. CONDITION 3: does the PINNED vLLM register each of our arms? ====="
# The eight kGgufArchArms families, plus the two families that OWN a staged
# artifact without appearing in that table -- Laguna, whose GGUF weight loader
# takes its HfConfig from elsewhere, and Nemotron-H, whose GGUF arm is OWED and
# refuses by its own name.
for a in DeepseekV4ForCausalLM MuseGlimmer Qwen3_5ForConditionalGeneration \
         Qwen3_5MoeForConditionalGeneration Qwen3NextForCausalLM Qwen4Exp \
         Glm5Next GlmMoeDsaForCausalLM Laguna NemotronHForCausalLM; do
  n=$(grep -c "\"$a" "$VLLM_PIN_CHECKOUT/vllm/model_executor/models/registry.py" 2>/dev/null)
  printf 'REGISTRY  %-38s hits=%s\n' "$a" "$n"
done
# muse-glimmer is the ONE candidate that meets every other condition, so its
# oracle reach is probed on all four surfaces that could supply one rather than
# on vLLM's registry alone. vLLM has a Transformers fallback and the plugin has
# a TransformersGGUFWeightsAdapter fallback, so "absent from registry.py" would
# NOT on its own settle it. What settles it is that the `gguf` package the
# plugin depends on -- llama.cpp's own gguf-py -- has no mapping for the
# architecture string the file declares, so the config cannot even be built.
echo "-- can ANY pinned oracle reach muse-glimmer? four surfaces, not one --"
echo "vllm_pin_glimmer_files   = $(grep -rli glimmer "$VLLM_PIN_CHECKOUT/vllm" 2>/dev/null | wc -l)"
echo "vllm_omni_glimmer_files  = $(grep -rli glimmer "$HOME/_git/vllm-omni" --exclude-dir=.git 2>/dev/null | wc -l)"
echo "llama_cpp_glimmer_files  = $(grep -rli glimmer "$HOME/_git/llama.cpp/src" 2>/dev/null | wc -l)"
echo "gguf_py_glimmer_files    = $(grep -rli glimmer "$HOME/_git/llama.cpp/gguf-py" 2>/dev/null | wc -l)"
echo "gguf_py_qwen35_files     = $(grep -rli qwen35 "$HOME/_git/llama.cpp/gguf-py" 2>/dev/null | wc -l)  <- the positive control"
echo "-- and the plugin's own adapter registry, from the staged source archive --"
PLUG_TGZ="${PLUG_TGZ:-/mnt/nas_share/rc/vllm-gfx1151-2740/ggufplugin-src.tar.gz}"
echo "plugin_tgz_sha256 = $(sha256sum "$PLUG_TGZ" 2>&1 | cut -d" " -f1)"
tar -xzOf "$PLUG_TGZ" vllm_gguf_plugin/weights_adapter/__init__.py 2>/dev/null \
  | sed -n '/^_ADAPTER_REGISTRY/,/^]/p'

echo
echo "===== 4. CONDITIONS 1, 4, 5: every GGUF staged on this fleet ====="
# Every candidate, measured. A split model is read from shard 00001, which
# carries the KV block; its per-shard tensor histograms follow it.
mapfile -t FILES < <(find "$NAS/checkpoints" "$NAS/rc/ckpt" -maxdepth 3 -name '*.gguf' 2>/dev/null | sort)
echo "n_gguf_found = ${#FILES[@]}"
for f in "${FILES[@]}"; do
  echo "-------------------------------------------------------------------"
  echo "BYTES              $(stat -c %s "$f")"
  timeout 600 python3 "$HDR" "$f" 2>&1 || echo "HEADER_READ_FAILED rc=$?"
done

echo
echo "===== 5. done ====="
date -u +%FT%TZ
echo "=== VEHICLE_SCAN DONE ==="
