// vllm.cpp original, DELIBERATELY EXCEEDING upstream. Grounded in, and a
// direct response to, vllm/v1/kv_offload/file_mapper.py:49-60,122-139 @
// e24d1b24.
//
// WHY THIS IS NOT A 1:1 PORT — the one place in this feature where "and better"
// is a CORRECTNESS claim rather than a speed claim
// (.agents/specs/kv-persistence-lmcache.md §Risks R3 / §B2).
//
// Upstream's `fs` tier protects a persisted cache with exactly one mechanism: a
// 12-hex-character digest embedded in the DIRECTORY NAME
// (file_mapper.py:128-139), computed over
// {model_name, hash_block_size, gpu_blocks_per_file, tp/pp/pcp/dcp, dtype,
//  kv_cache_groups, inference_engine}.
// It also writes a `config.json` — and NEVER READS IT. There is no reader
// anywhere in the repository (file_mapper.py:16,122-126;
// tiering/fs/manager.py:131-137). So the digest is the whole defence, and it
// omits:
//   * the checkpoint's actual CONTENT — two different finetunes served from the
//     same HF path collide;
//   * the WEIGHT quantization scheme;
//   * the rope / context configuration;
//   * `sliding_window`;
//   * the KV-cache dtype independently of `cache_dtype`;
//   * the block-hash algorithm and the NONE_HASH chain seed.
// Loading a block written under a different model or dtype therefore produces
// PLAUSIBLE TOKENS THAT ARE SIMPLY WRONG — no crash, no warning. On our side
// the exposure is sharper still: our KV dtype is an env-var A/B
// (include/vllm/v1/kv_cache_dtype.h:25-29) and HfConfig carries no
// quantization field at all.
//
// THE RULE, NON-NEGOTIABLE: a mismatch REFUSES. It never warns and proceeds.
// Verification happens at TWO levels and both are mandatory:
//   1. TIER OPEN — `config.json` is read back and compared FIELD BY FIELD, so
//      the refusal names the field that differs.
//   2. EVERY BLOCK OPEN — each block file carries a fixed-size header holding
//      the identity DIGEST, the payload size, and the block's own key. A file
//      that was written under another identity, truncated, or misfiled is
//      refused before a single payload byte is trusted.
//
// FORMAT CONSTRAINT (§Risks R9): the payload is OPAQUE BYTES of exactly
// `page_size_bytes()`, and the spec KIND plus every shape parameter lives in
// the identity. This is what lets ONE code path serve full attention (rank-4,
// interleaved [K|V], a factor of 2 in the block stride) and MLA (rank-3, one
// 576-wide latent, num_kv_heads == 1, NO separate V) without interpreting
// layout at the IO layer. Upstream independently confirms the shapes are not
// interchangeable by excluding MLA from `parallel_agnostic` folder sharing
// (file_mapper.py:85-96).
#ifndef VLLM_V1_KV_OFFLOAD_CACHE_IDENTITY_H_
#define VLLM_V1_KV_OFFLOAD_CACHE_IDENTITY_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vllm::v1::kv_offload {

// Bumped whenever the on-disk encoding changes in a way an older reader would
// misinterpret. Part of the identity, so a version change alone REFUSES an old
// directory rather than misreading it.
inline constexpr uint32_t kCacheFormatVersion = 1;

// Everything a persisted KV block's validity depends on. Every field is part of
// the digest; every field has a negative test asserting REFUSAL.
struct CacheIdentity {
  // --- provenance -----------------------------------------------------------
  // Always "vllm.cpp". Mirrors upstream's `inference_engine` field
  // (file_mapper.py:57) and guarantees our directories can never be mistaken
  // for a stock vLLM's — which matters because our block hashes are
  // sha256_cbor while upstream's DEFAULT is sha256-over-pickle, so the two
  // would key identically-named blocks differently (§Risks R1).
  std::string inference_engine = "vllm.cpp";
  uint32_t format_version = kCacheFormatVersion;

  // --- model ----------------------------------------------------------------
  std::string model_name;
  std::string model_type;
  std::vector<std::string> architectures;
  // SHA-256 (hex) over the canonical dump of HfConfig::raw
  // (include/vllm/transformers_utils/hf_config.h:111). Covers every UNTYPED
  // field too, so a config difference we do not model explicitly still
  // invalidates the cache instead of being ignored.
  std::string hf_config_digest;
  // Weight quantization scheme ("none", "nvfp4", "gguf:q4_k", ...). HfConfig
  // has NO quantization field, so this must be supplied by the loader; an
  // empty string is rejected by validate() rather than silently treated as
  // "unquantized".
  std::string weight_quantization;
  // Optional checkpoint content fingerprint. Upstream's digest covers only the
  // model PATH, so two finetunes at one path collide; when the loader can
  // supply a content fingerprint this closes that hole.
  std::string checkpoint_fingerprint;

  int64_t num_hidden_layers = 0;
  int64_t num_kv_heads = 0;
  int64_t head_size = 0;
  int64_t head_size_v = 0;
  // -1 when the model has no sliding window. NOT covered by upstream's digest.
  int64_t sliding_window = -1;
  // Canonical serialization of the rope configuration (theta, rotary dim,
  // scaling type and its parameters). NOT covered by upstream's digest — and a
  // rope change silently invalidates every cached page.
  std::string rope_config;
  int64_t max_position_embeddings = 0;

  // --- KV cache shape -------------------------------------------------------
  // The KV cache spec KIND ("full_attention", "mla", "sliding_window",
  // "chunked_local_attention", "mamba"). Distinguishes the rank-4 [K|V] page
  // from MLA's rank-3 latent page even when the byte counts happen to agree.
  std::string kv_cache_spec_kind;
  // The EXACT payload size of one block file.
  int64_t page_size_bytes = 0;
  int64_t block_size = 0;
  int64_t hash_block_size = 0;
  // From spec->dtype — NEVER from the VT_KV_CACHE_F32 env var, which describes
  // an A/B knob rather than the authoritative storage dtype.
  std::string kv_dtype;
  // KV quantization mode. Quantized KV cannot be persisted TODAY at all (every
  // spec's real_page_size_bytes() throws when kv_quant_mode != kNone,
  // src/vllm/v1/kv_cache_interface.cpp:18-20), but recording it means a future
  // FP8/NVFP4 KV cache can never silently read bf16 files (§Risks R11).
  std::string kv_quant_mode = "none";

  // --- hashing --------------------------------------------------------------
  // The block-hash algorithm name ("sha256_cbor").
  std::string hash_algo = "sha256_cbor";
  // The NONE_HASH chain seed value, hex. THE most important field for a
  // persisted cache: block hashes CHAIN from it, so a cache written under one
  // seed is meaningless under another. Upstream's digest does not contain it,
  // which is why upstream's cross-process story is "set PYTHONHASHSEED
  // everywhere or get zero hits" rather than a refusal.
  std::string none_hash_hex;
  // How that seed was obtained ("default", "PYTHONHASHSEED", ...). Recorded for
  // auditability; it is NOT compared, because the same seed VALUE reached by
  // two different routes is the same cache.
  std::string none_hash_source;

  // --- parallelism ----------------------------------------------------------
  int64_t tp_size = 1;
  int64_t pp_size = 1;
  int64_t pcp_size = 1;
  int64_t dcp_size = 1;
  int64_t rank = 0;

  // Canonical JSON: sorted keys, no whitespace — mirrors upstream's
  // json.dumps(fields, sort_keys=True, separators=(",", ":"))
  // (file_mapper.py:135). Byte-stable across runs and platforms.
  std::string ToCanonicalJson() const;

  // Parse a canonical JSON document back into an identity. Throws
  // std::runtime_error on malformed input.
  static CacheIdentity FromCanonicalJson(const std::string& json_text);

  // SHA-256 over ToCanonicalJson(), as 32 RAW bytes.
  std::string Digest() const;
  // The first 12 hex characters of Digest(), used in the directory name
  // exactly as upstream's _BASE_PATH_HASH_LEN does (file_mapper.py:16).
  std::string ShortDigestHex() const;

  // Reject an identity that cannot describe a real cache (empty required
  // strings, non-positive sizes). Returns the offending field name, or nullopt
  // when the identity is usable. Catches the "" -> "treated as unquantized"
  // class of bug at construction rather than at read time.
  std::optional<std::string> Validate() const;

  // FIELD-BY-FIELD comparison. Returns the name of the FIRST field that
  // differs, or nullopt when the two identities describe the same cache.
  // `none_hash_source` is deliberately excluded (see its comment).
  static std::optional<std::string> FirstMismatch(const CacheIdentity& a,
                                                  const CacheIdentity& b);
};

}  // namespace vllm::v1::kv_offload

#endif  // VLLM_V1_KV_OFFLOAD_CACHE_IDENTITY_H_
