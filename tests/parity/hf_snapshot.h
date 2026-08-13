#ifndef VLLM_TESTS_PARITY_HF_SNAPSHOT_H_
#define VLLM_TESTS_PARITY_HF_SNAPSHOT_H_

// Resolving a Hugging Face cache snapshot for a checkpoint-gated test.
//
// Every one of these gates used to take the FIRST entry `directory_iterator`
// yielded under `<repo>/snapshots/`. That is only safe while a repo has exactly
// one cached revision, and `unsloth/Qwen3.6-27B-NVFP4` does not:
//
//   @890bdef7  genuine NVFP4 - `weight_packed` U8 + `weight_scale` F8_E4M3 +
//              `weight_global_scale` F32. Every committed 27B golden was
//              captured against it (see the `oracle.model` field of
//              tests/parity/goldens/qwen36_*_27b/manifest.json).
//   @ccdaab7e  the SAME repo name, silently re-quantized to FP8 W8A8
//              throughout, with every NVFP4-specific `*_global_scale` tensor
//              gone.
//
// So the filesystem decided which model the SACRED gate measured, and a
// token-exact pass against an FP8 model would have been recorded as an NVFP4
// pass. Publishers re-quantize in place; a correctness gate must name the
// revision its golden belongs to.

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace parity {

// The revision the committed 27B goldens were captured against.
inline constexpr const char* kQwen27NvfP4Revision =
    "890bdef7a42feba6d83b6e17a03315c694112f2a";

// The revision the committed Nemotron-3.5-Lightning-30B-A3B-NVFP4 goldens were
// captured against (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4).
// Captured 2026-08-12 on GB10 from the PINNED oracle -- vLLM
// 0.23.1rc1.dev1511+g555967922, transformers 5.14.1, flashinfer 0.6.15.post1 --
// with the identity asserted and the run aborting on mismatch. NOT the
// `vllm-oracle` symlink, which resolves to the v0.25.0 rollback and predates
// NemotronHMoEDecoderLayer entirely.
inline constexpr const char* kNemotron35LightningNvfP4Revision =
    "29f2d1746d8f41e316523194b19018707749b1b1";

// Snapshot directory for `<repo>` at `revision`, or "" when it is not cached
// (the caller then emits its loud SKIP). `env_override`, when set and non-empty,
// names an explicit snapshot directory for a deliberate different-checkpoint
// run and is the ONLY way to gate a revision other than the pinned one -- a
// cache holding some other revision skips rather than silently substituting it.
inline std::string HfSnapshot(const char* repo_dir, const char* revision,
                              const char* env_override) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (env_override != nullptr) {
    const char* over = std::getenv(env_override);
    if (over != nullptr && *over != '\0') {
      if (fs::exists(fs::path(over) / "config.json", ec)) return over;
      return "";
    }
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snap = fs::path(home) / ".cache/huggingface/hub" / repo_dir /
                        "snapshots" / revision;
  if (!fs::exists(snap / "config.json", ec)) return "";
  return snap.string();
}

// The Nemotron-3.5-Lightning gate model (#517). Unlike the Qwen pins above,
// this one is NOT in the HF cache: it is staged on the NAS as a `local_dir`
// snapshot at `$CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4`, so there is
// no `models--org--name/snapshots/<rev>` layout to resolve. The env override is
// therefore the ONLY reachable path, and the cache spelling below exists so the
// revision still names what the golden belongs to. Absent env var => "" => the
// caller emits its loud SKIP, which is the intended behavior off the gate host.
inline std::string Nemotron35LightningSnapshot() {
  return HfSnapshot("models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                    kNemotron35LightningNvfP4Revision,
                    "VT_NEMOTRON35_SNAPSHOT");
}

// The 27B NVFP4 gate model, pinned to the goldens' revision.
inline std::string Qwen27NvfP4Snapshot() {
  return HfSnapshot("models--unsloth--Qwen3.6-27B-NVFP4",
                    kQwen27NvfP4Revision, "VT_QWEN27_SNAPSHOT");
}

// GATE-27B-FP8-TOWER-GOLDEN (issue #466). The OTHER dense 27B -- a different
// model, not a second spelling of the one above. `nvidia/Qwen3.6-27B-NVFP4` is
// `modelopt_mixed`: a W4A16 NVFP4 MLP plus an FP8 W8A8 tower
// (`linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj`, and every
// `self_attn.{q,k,v,o}_proj`) and an NVFP4 head.
//
// It is the checkpoint every fp8-tower lever targets and every recorded 27B
// performance ratio was taken on, and until #466 nothing gated it
// token-for-token. The unsloth revision above lists
// `linear_attn.in_proj_{qkv,z,a,b}` in its `ignore` set and ships ZERO
// `*.input_scale` tensors, so there the fp8 arm is not merely differently tuned
// -- it cannot execute at all, and a gate pinned there can never fail for an
// fp8 defect.
//
// Deliberately NOT named `kQwen27NvfP4Revision<suffix>`:
// tests/tools/test_online_gate_server_binary.py parses that exact identifier
// out of this header and asserts a SINGLE pin equal to
// MODEL_GATE_CONTRACTS["test_qwen27_paged_engine"]["golden_revision"]. That
// assertion is correct and is left untouched.
inline constexpr const char* kQwen27nFp8TowerRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";

// The FP8-tower 27B gate model, pinned to ITS goldens' revision
// (tests/parity/goldens/qwen36_*_27n). An absent cache yields "" and the caller
// emits its loud refusal; a cache holding a DIFFERENT revision of the same repo
// skips rather than being substituted, exactly as above.
inline std::string Qwen27nFp8TowerSnapshot() {
  return HfSnapshot("models--nvidia--Qwen3.6-27B-NVFP4",
                    kQwen27nFp8TowerRevision, "VT_QWEN27N_SNAPSHOT");
}

// GATE-PIN-UNPINNED-SNAPSHOTS (issue #471). The 35B MoE gate model. Pinned to
// the revision its OWN goldens name: `oracle.model` of
// tests/parity/goldens/qwen36_{embed,norm,gdn_layer,fullattn_layer,logits}_35b/
// manifest.json, and `args.model` of goldens/qwen3_5_mtp_head_35b/manifest.json,
// all of which spell out
// `.../models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/491c2f1e.../`.
//
// Only one revision of this repo is cached on the gate host today, so the five
// gates that used to resolve it by `directory_iterator` were latently, not
// actively, wrong. Latent is still unpinned: the goldens name a revision, so
// there was never a reason not to enforce it.
//
// Deliberately NOT named `kQwen27NvfP4Revision<suffix>`, for the same reason
// `kQwen27nFp8TowerRevision` is not:
// tests/tools/test_online_gate_server_binary.py parses that exact identifier out
// of this header and asserts a SINGLE pin. That assertion is correct and is left
// untouched.
inline constexpr const char* kQwen36A3bNvfP4Revision =
    "491c2f1ea524c639598bf8fa787a93fed5a6fbce";

// The 35B-A3B NVFP4 gate model, pinned to its goldens' revision.
inline std::string Qwen36A3bNvfP4Snapshot() {
  return HfSnapshot("models--nvidia--Qwen3.6-35B-A3B-NVFP4",
                    kQwen36A3bNvfP4Revision, "VT_QWEN36_SNAPSHOT");
}

// GATE-PIN-UNPINNED-SNAPSHOTS (issues #471, #472). The DFlash draft checkpoint.
//
// READ THIS BEFORE QUOTING IT. This pin is weaker than the two above and the
// difference is not cosmetic. `kQwen27NvfP4Revision` and
// `kQwen36A3bNvfP4Revision` are RATIFIED: a committed golden manifest names the
// revision it was captured against, so the pin is checkable against the artifact
// it gates. This one is a DETERMINISM pin only.
//
// What was checked:
//   * The DFlash goldens record NO revision. `goldens/dflash_27b/
//     dflash_27b_spec_{on,off}.json` carry `draft="z-lab/Qwen3.6-27B-DFlash"` as
//     a bare repo name; `goldens/dflash_27b_{draft,kvprep}/` name neither repo
//     nor revision.
//   * `goldens/dflash_27b_draft/ckpt_keys.txt` LOOKS like a checkpoint
//     fingerprint and is not one. Compared against this revision on the gate
//     host: 47 golden keys vs 58 snapshot keys, MISMATCH -- the golden list is
//     `model.`-prefixed with `mlp.gate_up_proj` fused and includes
//     `lm_head`/`embed_tokens` the draft checkpoint does not ship. It is vLLM's
//     LOADED-MODULE namespace, not the safetensors key set, so it cannot
//     identify a snapshot.
//   * The revision itself IS committed, at
//     .agents/completed/state-events/0000-00/STATE-LEGACY-000001.md:30698,32279,
//     which records `0919688...` as the snapshot fetched and used for this work.
//
// So this stops the gate silently substituting a future re-quant of the same
// repo -- the failure mode #471 is about -- and it does NOT prove the goldens
// belong to it. #472 owes the re-capture that would make it ratified. Until
// then no DFlash result may be quoted as "gated against a KNOWN checkpoint"; it
// is gated against a NAMED one, which is weaker and is the honest claim.
inline constexpr const char* kQwen27DFlashDraftRevision =
    "0919688658996800f86b895034249700e9481106";

// The z-lab DFlash draft, pinned for determinism (provenance UNRATIFIED, #472).
inline std::string Qwen27DFlashDraftSnapshot() {
  return HfSnapshot("models--z-lab--Qwen3.6-27B-DFlash",
                    kQwen27DFlashDraftRevision, "VT_DFLASH_DRAFT_SNAPSHOT");}

}  // namespace parity

#endif  // VLLM_TESTS_PARITY_HF_SNAPSHOT_H_
