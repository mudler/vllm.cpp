#!/usr/bin/env bash
# The limb-3 VEHICLE PIN CHECK, re-runnable.
#
# #2864 measured the 77 GGUFs already staged on this fleet against six
# pre-registered conditions and found the intersection EMPTY. Its §7 option 2
# says what a vehicle would be: "a second dense `qwen35` checkpoint that is not
# Qwen3.8-27B", and closes "Whether one exists was not established here, because
# fetching it needs recorded authority and none was given."
#
# The authority is now recorded (.agents/developer-preferences.md, 2026-09-04).
# This script establishes the candidate against the SAME six conditions, before
# the download rather than after it, so the choice cannot be fitted to whatever
# arrived.
#
# CORRECTNESS ONLY. No model is run here, no timing is taken, and no number
# below is a performance result.
set -uo pipefail

TREE="${TREE:-$(cd "$(dirname "$0")/../../.." && pwd)}"
VLLM_PIN_CHECKOUT="${VLLM_PIN_CHECKOUT:-$HOME/_git/vllm}"
VLLM_PIN=5559679229bc961848b121ccdeaa8fa5d79bec98
HDR="$(dirname "$0")/../limb3-vehicle-search-20260904/gguf_header.py"
RHDR="$(dirname "$0")/remote_gguf_header.py"

REPO=unsloth/Qwen3.6-27B-GGUF
REV=82d411acf4a06cfb8d9b073a5211bf410bfc29bf
FILE=Qwen3.6-27B-Q4_K_M.gguf
BYTES=16817244384
LFS_SHA=5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0

echo "===== 0. identity ====="
date -u +%FT%TZ
echo "tree             = $TREE"
echo "tree_head        = $(git -C "$TREE" rev-parse HEAD 2>&1)"
echo "vllm_checkout    = $VLLM_PIN_CHECKOUT"
echo "vllm_head        = $(git -C "$VLLM_PIN_CHECKOUT" rev-parse HEAD 2>&1)"
if [ "$(git -C "$VLLM_PIN_CHECKOUT" rev-parse HEAD 2>/dev/null)" = "$VLLM_PIN" ]; then
  echo "VLLM_CHECKOUT_IS_THE_PIN = yes"
else
  echo "VLLM_CHECKOUT_IS_THE_PIN = NO -- every registry answer below is about the wrong tree"
fi
echo "candidate_repo   = $REPO"
echo "candidate_rev    = $REV"
echo "candidate_file   = $FILE"

echo
echo "===== 1. THE PIN IS A REVISION, NOT A REPO ID ====="
# #2497 refused the UD family because `Qwen3.8-27B-UD-Q4_K_XL`'s published bytes
# MOVED IN PLACE under an unchanged name (17,923,394,624 -> 17,559,178,144 B).
# A repo id is therefore not a pin. The size and the LFS sha256 are read at the
# NAMED REVISION and printed here so a later re-publication is detectable.
curl -s -m 60 -X POST \
  "https://huggingface.co/api/models/$REPO/paths-info/$REV" \
  -H 'Content-Type: application/json' -d "{\"paths\":[\"$FILE\"]}" \
  | python3 -c 'import json,sys
d=json.load(sys.stdin)[0]
print("HF_SIZE            %d" % d["size"])
print("HF_GIT_OID         %s" % d["oid"])
print("HF_LFS_SHA256      %s" % d["lfs"]["oid"])'
echo "EXPECT_SIZE        $BYTES"
echo "EXPECT_LFS_SHA256  $LFS_SHA"
echo "NOTE: the LFS oid is what the forge ADVERTISES. The authoritative value is"
echo "      the sha256 measured on the staged bytes after transfer (§5)."

echo
echo "===== 2. CONDITIONS 1, 2, 4, 5: the candidate's OWN header, by RANGE REQUEST ====="
# Read without downloading 16.8 GB, which is this repository's practice for a
# checkpoint manifest. The predicate is evaluated against the artifact's ggml
# type histogram, never against its file name (#2510, #2864 §3).
timeout 900 python3 "$RHDR" \
  "https://huggingface.co/$REPO/resolve/$REV/$FILE" 2>&1

echo
echo "===== 2b. the ARM's own artifact, read by the same instrument ====="
# The comparison the limb turns on: same architecture string, same quant tier,
# a DIFFERENT model. Printed side by side so "same forward code" is checkable
# rather than asserted.
timeout 900 python3 "$HDR" \
  /mnt/nas_share/checkpoints/qwen3.8-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf 2>&1

echo
echo "===== 2c. THE REJECTED SIBLING, read by the same instrument ====="
# The family publishes exactly two dense `qwen35` checkpoints that are not the
# arm. Both `Q4_K_M` files are read here, because the LABEL says they are the
# same thing and the BYTES say they are not: the Qwen3.5 file routes 96 tensors
# through the Q8_0 path, which is a different kernel from the three k-quant
# kernels limb 3 exists to exercise. Printing the rejected candidate is what
# makes the choice auditable rather than asserted.
ALT_REV=3221f178a6b842d04f1fb42f1c413534adcc0a6a
timeout 900 python3 "$RHDR" \
  "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/$ALT_REV/Qwen3.5-27B-Q4_K_M.gguf" 2>&1

echo
echo "===== 3. CONDITION 2: is qwen35 in OUR dispatch? printed from the tree ====="
sed -n '/^constexpr GgufArchArm kGgufArchArms\[\] = {/,/^};/p' \
  "$TREE/src/vllm/entrypoints/model_loader.cpp"
grep -rhn 'inline constexpr const char\* k[A-Za-z0-9]*GgufArch = ' "$TREE/include" | sort
echo "-- the ggml type ids our reader ACCEPTS (one refused id refuses the file WHOLE) --"
ACCEPTED=$(grep -oE '^    case [0-9]+: \{' \
  "$TREE/src/vllm/model_executor/model_loader/gguf_reader.cpp" \
  | grep -oE '[0-9]+' | sort -n | tr '\n' ' ')
echo "ACCEPTED_GGML_TYPE_IDS = $ACCEPTED"
for id in 0 12 13 14; do   # exactly the ids the candidate stores
  case " $ACCEPTED " in
    *" $id "*) echo "  candidate stores id $id -- ACCEPTED" ;;
    *)         echo "  candidate stores id $id -- REFUSED (GgmlTraits throws)" ;;
  esac
done

echo
echo "===== 4. CONDITION 3: the FOUR-SURFACE oracle check, with BOTH controls ====="
# #2864 probed four surfaces for muse-glimmer and carried a POSITIVE control so
# a silent grep failure could not read as an absence. Here the expected answer
# is POSITIVE, so the control that matters is the NEGATIVE one: muse-glimmer is
# re-probed on the identical surfaces, and it must still read 0. A grep that
# matched everything would light both rows and be visible.
echo "-- the candidate declares this HF architecture --"
curl -s -m 60 "https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json" \
  | python3 -c 'import json,sys
c=json.load(sys.stdin)
print("HF_ARCHITECTURES   %s" % c.get("architectures"))
print("HF_MODEL_TYPE      %s" % c.get("model_type"))
t=c.get("text_config",c)
print("HF_LAYERS          %s" % t.get("num_hidden_layers"))
print("HF_HIDDEN          %s" % t.get("hidden_size"))
print("HF_FFN             %s" % t.get("intermediate_size"))
print("HF_EXPERT_KEYS     %s" % [k for k in t if "expert" in k])'
for a in Qwen3_5ForConditionalGeneration MuseGlimmer; do
  n=$(grep -c "\"$a" "$VLLM_PIN_CHECKOUT/vllm/model_executor/models/registry.py" 2>/dev/null)
  printf 'REGISTRY  %-38s hits=%s\n' "$a" "$n"
done
echo "-- surface 1..4, candidate (expect >0) against negative control (expect 0) --"
echo "vllm_pin_qwen35_files      = $(grep -rli qwen3_5 "$VLLM_PIN_CHECKOUT/vllm" 2>/dev/null | wc -l)"
echo "vllm_pin_glimmer_files     = $(grep -rli glimmer "$VLLM_PIN_CHECKOUT/vllm" 2>/dev/null | wc -l)  <- negative control"
echo "vllm_omni_qwen35_files     = $(grep -rli qwen3_5 "$HOME/_git/vllm-omni" --exclude-dir=.git 2>/dev/null | wc -l)"
echo "vllm_omni_glimmer_files    = $(grep -rli glimmer "$HOME/_git/vllm-omni" --exclude-dir=.git 2>/dev/null | wc -l)  <- negative control"
echo "llama_cpp_qwen35_files     = $(grep -rli qwen35 "$HOME/_git/llama.cpp/src" 2>/dev/null | wc -l)"
echo "llama_cpp_glimmer_files    = $(grep -rli glimmer "$HOME/_git/llama.cpp/src" 2>/dev/null | wc -l)  <- negative control"
echo "gguf_py_qwen35_files       = $(grep -rli qwen35 "$HOME/_git/llama.cpp/gguf-py" 2>/dev/null | wc -l)"
echo "gguf_py_glimmer_files      = $(grep -rli glimmer "$HOME/_git/llama.cpp/gguf-py" 2>/dev/null | wc -l)  <- negative control"
echo "llama_cpp_head             = $(git -C "$HOME/_git/llama.cpp" rev-parse HEAD 2>&1)"
echo "-- the PLUGIN's adapter registry, from the staged source archive --"
PLUG_TGZ="${PLUG_TGZ:-/mnt/nas_share/rc/vllm-gfx1151-2740/ggufplugin-src.tar.gz}"
echo "plugin_tgz_sha256 = $(sha256sum "$PLUG_TGZ" 2>&1 | cut -d' ' -f1)"
tar -xzOf "$PLUG_TGZ" vllm_gguf_plugin/weights_adapter/__init__.py 2>/dev/null \
  | sed -n '/^_ADAPTER_REGISTRY/,/^]/p'
echo "-- and the Qwen35 adapter's own architecture predicate --"
tar -tzf "$PLUG_TGZ" 2>/dev/null | grep -i qwen | sed 's/^/plugin_file /'

echo
echo "===== 5. CONDITION 5, and the AUTHORITATIVE sha256 of the STAGED bytes ====="
STAGED="${STAGED:-/mnt/nas_share/rc/ckpt/qwen36-27b-q4km/$FILE}"
if [ -f "$STAGED" ] && [ "$(stat -c %s "$STAGED")" = "$BYTES" ]; then
  # The digest is printed ONLY when the size already matches. A partial file
  # hashes perfectly well and its digest would be a plausible wrong answer, so
  # the size is the gate on whether the question is even asked.
  echo "STAGED_PATH        $STAGED"
  echo "STAGED_BYTES       $(stat -c %s "$STAGED")"
  echo "STAGED_SHA256      $(sha256sum "$STAGED" | cut -d' ' -f1)"
  echo "STAGED_SHA256_MATCHES_FORGE_LFS_OID  $([ "$(sha256sum "$STAGED" | cut -d' ' -f1)" = "$LFS_SHA" ] && echo yes || echo NO)"
elif [ -f "$STAGED" ]; then
  echo "STAGED_PATH        $STAGED"
  echo "STAGED_BYTES       $(stat -c %s "$STAGED")  (expected $BYTES)"
  echo "STAGED_SHA256      NOT_COMPUTED -- the file is not the declared size, and a partial file's digest is a plausible wrong answer"
  echo "BOARD_FREE_MIB     59934   (docs/bench-evidence/rocm-strix-qwen38-q4km-20260901.md)"
  python3 -c "print('FITS_CARVE         %s' % ('yes' if $BYTES/1048576 < 59934 else 'NO'))"
else
  echo "STAGED_PATH        $STAGED"
  echo "STAGED_SHA256      NOT_STAGED_YET"
fi

echo
echo "=== VEHICLE_PIN_CHECK DONE ==="
date -u +%FT%TZ
