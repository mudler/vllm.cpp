// Ported from: vllm/config/speculative.py @ e24d1b24
//
// Scope (SPEC-MTP I2, scheduler-half): the T0 SUBSET of SpeculativeConfig the V1
// Scheduler actually reads — the method string, the resolved
// num_speculative_tokens, and the predicates that derive num_lookahead_tokens
// (scheduler.py:275-292). This is the config seam that turns the spec-decode
// scheduler plumbing ON: with no SpeculativeConfig (the production default), the
// scheduler keeps num_lookahead_tokens == 0 and every spec path is inert, so the
// scheduler/engine/input-batch behavior is byte-identical to the pre-spec engine.
//
// The full upstream resolution (draft ModelConfig, quantization, method
// auto-detection from the checkpoint's architectures, dynamic SD schedule,
// disable-by-batch-size, ROCm/attention-backend gating) is DEFERRED to the
// model-loader claims (MODEL-SPEC-qwen3-5-mtp-*); this header carries only what
// the scheduler consumes. The MTP method-resolution rule mirrored here is
// speculative.py:480-489 (model_type qwen3_5|qwen3_5_moe -> method "mtp",
// n_predict = mtp_num_hidden_layers) + speculative.py:865-875 (default
// num_speculative_tokens = n_predict when the user gives no explicit k).
#ifndef VLLM_CONFIG_SPECULATIVE_H_
#define VLLM_CONFIG_SPECULATIVE_H_

#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vllm {

// SPEC-DRAFTER-CHAIN W1 (#1522): one entry of the preference-ordered drafter
// chain — the `vllm_cpp.drafter_chain` extension of `--speculative-config`.
//
// It is a list of OBJECTS and not llama.cpp's comma-separated `--spec-type`
// name string (`common/arg.cpp:3754-3763`), because each entry carries its own
// draft checkpoint and its own k. The fields are exactly the five HONOURED
// top-level keys of the same document, with the same meanings, so an entry is
// read the way a single-method config is read and there is no second spelling
// of anything.
//
// vLLM has NO composition surface to mirror: `SpeculativeMethod` is a single
// `Literal` and `SpeculativeConfig.method` one value of it, verified at the
// parity pin `555967922` and at upstream `origin/main` `c20572610`
// (.agents/specs/drafter-chain.md `## Upstream chain`). llama.cpp is the
// secondary oracle for the SEMANTICS only.
struct SpeculativeChainEntry {
  // "mtp", "dflash", "dspark" or "ngram" — the four this engine implements and
  // the four the row scopes. Never empty on a parsed entry.
  std::string method;
  // The entry's own k (`num_speculative_tokens`). std::nullopt = "not given",
  // resolved per method exactly as the top-level key is.
  std::optional<int> num_speculative_tokens = std::nullopt;
  // The entry's own draft checkpoint (`model`). Required for "dflash" and
  // "dspark"; meaningless for "mtp" (whose draft lives in the target) and for
  // "ngram" (which has no draft model).
  std::optional<std::string> draft_model_path = std::nullopt;
  // The entry's own n-gram window. Only meaningful for "ngram".
  std::optional<int> prompt_lookup_min = std::nullopt;
  std::optional<int> prompt_lookup_max = std::nullopt;
};

// SpeculativeConfig (T0 scheduler subset). Value type; the Scheduler ctor reads
// it (upstream vllm_config.speculative_config).
struct SpeculativeConfig {
  // The speculative method (speculative.py `method`): "mtp", "eagle", "eagle3",
  // "dflash", "dspark", "draft_model", ... Empty string = unset (never
  // constructed on the default path).
  std::string method;

  // num_speculative_tokens (k). Upstream `int | None`; std::nullopt mirrors the
  // "not given -> default to n_predict" resolution done in ResolveMtp / the
  // upstream __post_init__ (speculative.py:865-875).
  std::optional<int> num_speculative_tokens = std::nullopt;

  // SPEC-DFLASH D5: the DFlash draft model path (the `model` key of
  // `--speculative-config`, e.g. "z-lab/Qwen3.6-27B-DFlash" or a local snapshot
  // dir). Unlike MTP (whose draft `mtp.*` tensors live inside the target
  // checkpoint), the DFlash draft is a SEPARATE checkpoint the loader opens; this
  // carries its path. Empty/unset for MTP and non-spec.
  std::optional<std::string> draft_model_path = std::nullopt;

  // n_predict: the draft head depth (MTP: mtp_num_hidden_layers, = 1 for both
  // gate checkpoints). Upstream reads it off the draft model's hf_config
  // (speculative.py:482-486,860-863). 0 when not an n_predict-style method.
  int n_predict = 0;

  // SPEC-NGRAM (ROAD-V1-D3): the n-gram proposer window (speculative.py:157-161
  // prompt_lookup_max/prompt_lookup_min). Only meaningful for method=="ngram";
  // the draft-free matcher looks for a suffix ngram of length in
  // [prompt_lookup_min, prompt_lookup_max]. Defaulted to 5/5 by ResolveNgram when
  // the user gives neither (speculative.py:737-742). std::nullopt for non-ngram.
  std::optional<int> prompt_lookup_min = std::nullopt;
  std::optional<int> prompt_lookup_max = std::nullopt;

  // SPEC-DSPARK W1: parallel_drafting (speculative.py:963-964) — set for the
  // BLOCK drafters ("dflash", "dspark"), which propose the whole draft block in
  // one forward instead of k autoregressive steps. Upstream also uses it to
  // reject P-EAGLE on the V2 runner (config/vllm.py:2168-2177); for us it is the
  // declarative marker that the propose path is block-shaped.
  bool parallel_drafting = false;

  // SPEC-DRAFTER-CHAIN W1 (#1522): the preference-ordered drafter chain, in the
  // order the user gave it. EMPTY is "no chain", which is every document that
  // predates this field and every document vLLM accepts — the additivity the
  // row is built on (.agents/specs/drafter-chain.md D1).
  //
  // W1 lands the field, its validation and its refusals, and NO chain
  // behaviour. `LoadedEngine::ResolveSpecConfig` is the production reader: it
  // refuses a chain BY NAME, before any weight I/O, rather than reducing it to
  // one drafter or to no speculation at all. Resolution is owed by W3.
  std::vector<SpeculativeChainEntry> drafter_chain;

  // use_drafter_chain: whether this config asks for a chain at all. The one
  // predicate every consumer asks, so "empty vector means absent" is stated
  // once instead of at each call site.
  bool use_drafter_chain() const { return !drafter_chain.empty(); }

  // ResolveMtp: build the scheduler-facing SpeculativeConfig for a Qwen3.5 MTP
  // checkpoint. Mirrors speculative.py:480-489 (method "mtp",
  // n_predict = mtp_num_hidden_layers) + :865-875 (default k = n_predict). A user
  // override for k must be a positive multiple of n_predict for MTP-module reuse
  // (speculative.py:869-875); we mirror the divisibility check.
  static SpeculativeConfig ResolveMtp(
      int mtp_num_hidden_layers,
      std::optional<int> user_num_speculative_tokens = std::nullopt) {
    SpeculativeConfig cfg;
    cfg.method = "mtp";
    cfg.n_predict = mtp_num_hidden_layers;
    if (user_num_speculative_tokens.has_value()) {
      const int k = *user_num_speculative_tokens;
      if (mtp_num_hidden_layers > 0 && k > mtp_num_hidden_layers &&
          k % mtp_num_hidden_layers != 0) {
        throw std::invalid_argument(
            "num_speculative_tokens must be divisible by n_predict "
            "(mtp_num_hidden_layers) for MTP module reuse");
      }
      cfg.num_speculative_tokens = k;
    } else {
      // Default to the head depth (speculative.py:867-868).
      cfg.num_speculative_tokens = mtp_num_hidden_layers;
    }
    return cfg;
  }

  // ResolveDflash: build the scheduler-facing SpeculativeConfig for a z-lab DFlash
  // draft (SPEC-DFLASH D4). Mirrors speculative.py method resolution for "dflash"
  // (:1172 use_dflash) + the num_speculative_tokens = block-derived k the loader
  // reads off the draft's dflash_config. Unlike MTP there is no n_predict-module
  // divisibility constraint (the block drafter is non-autoregressive); k is taken
  // as given (the CLI value, or the draft's block-1 default resolved by the loader).
  // n_predict stays 0 (not an n_predict-style method); the extra scheduler
  // lookahead slot comes from use_dflash()/NumLookaheadTokens() (already coded).
  static SpeculativeConfig ResolveDflash(int num_speculative_tokens_k) {
    SpeculativeConfig cfg;
    cfg.method = "dflash";
    cfg.n_predict = 0;
    cfg.num_speculative_tokens = num_speculative_tokens_k;
    cfg.parallel_drafting = true;  // speculative.py:963-964
    return cfg;
  }

  // IsDflash2Draft: whether a DFlash draft is a DFlash2 one, by the architecture
  // upstream itself selects on.
  //
  // BEYOND-PIN (SPEC-DFLASH2 W1, #1314). vllm-project/vllm#52816 MERGED upstream
  // on 2026-08-21 at 05:27:22Z, at head
  // `3406ec1dae9916f920b90f0dbf90dcf54923d042`, merge commit
  // `b389ac29465b33f9e9c534df221ea3c129e9793f`. This comment said OPEN at head
  // `19c9351904df4c63042671bc67a866ca48dc7d6f` -- the FIRST of three heads the
  // pull request carried -- and was doubly stale; corrected 2026-08-21 by the
  // W6 repair wave. The parity pin `555967922` still does not carry the
  // architecture at all, this row does NOT advance the pin, and reconciling the
  // port onto the merged head is owed under #1561.
  // Upstream registers `"DFlash2DraftModel" -> ("qwen3_dflash2",
  // "DFlash2Qwen3ForCausalLM")` (`model_executor/models/registry.py:628`) and
  // asks exactly this question in two places: the speculator selection
  // (`v1/worker/gpu/spec_decode/__init__.py:12`) and `_is_dflash2_draft`
  // (`config/vllm.py:668-676`), both spelled as membership of the string in the
  // draft config's `architectures`. The precedent for mirroring an open pull
  // request ahead of the pin is `SPEC-DSPARK-QWEN3-ROUTING` toward vllm#52197,
  // argued at .agents/specs/dflash2-spec-decode.md D1.
  //
  // It answers a QUESTION and refuses nothing. The refusal — with the missing
  // parts named — belongs to the loader, which is where a user arrives
  // (src/vllm/entrypoints/model_loader.cpp::ResolveSpecConfig).
  static bool IsDflash2Draft(const std::vector<std::string>& architectures) {
    for (const std::string& arch : architectures) {
      if (arch == "DFlash2DraftModel") return true;
    }
    return false;
  }

  // IsDsparkDraft: the DSpark half of upstream's method auto-detection
  // (speculative.py:881-887) — a draft whose model id contains "dspark"
  // (case-insensitively, e.g. deepseek-ai/dspark_qwen3_8b_block7 or
  // RedHatAI/Qwen3.6-35B-A3B-speculator.dspark) OR whose architectures name
  // "Qwen3DSparkModel" / "Gemma4DSparkModel". Kept separate from ResolveDspark so
  // the loader can classify a checkpoint before building any config.
  //
  // BEYOND-PIN (SPEC-DSPARK-QWEN3-ROUTING, #1193): the fourth arm — the
  // "DSparkDraftModel" + model_type "qwen3" PAIR — is vllm-project/vllm#52197
  // hunk 1, merged 2026-08-17 at 7075ddac28c25d4fd2b84bc2a9a6c5ffde0345c8 and
  // absent from the pin 555967922. It is mirrored AHEAD of the pin because the
  // pinned behavior is wrong for a checkpoint that is published and loads here
  // today (RadixArk/Qwen3.8-27B-DSpark declares exactly that pair);
  // .agents/specs/dspark-qwen3-routing.md §2 carries the decision and the three
  // model-matrix precedents for the mark.
  //
  // `model_type` defaults to the empty string so a caller holding only the
  // architecture list asks exactly the pinned question and gets the pinned
  // answer.
  static bool IsDsparkDraft(const std::string& draft_model_id,
                            const std::vector<std::string>& architectures,
                            const std::string& model_type = "") {
    std::string lowered = draft_model_id;
    for (char& c : lowered) {
      c = static_cast<char>(
          std::tolower(static_cast<unsigned char>(c)));  // .lower() upstream
    }
    if (lowered.find("dspark") != std::string::npos) {
      return true;
    }
    for (const std::string& arch : architectures) {
      if (arch == "Qwen3DSparkModel" || arch == "Gemma4DSparkModel") {
        return true;
      }
      // vllm#52197 hunk 1 (BEYOND-PIN): the same architecture string names a
      // DeepSeek-V4 draft when model_type is "deepseek_v4", so the PAIR is the
      // condition and the architecture alone is not.
      if (arch == "DSparkDraftModel" && model_type == "qwen3") {
        return true;
      }
    }
    return false;
  }

  // ResolveDsparkArchitecture: the architecture normalization upstream performs
  // before a DSpark draft loads (speculative.py:934-944 @ 555967922, plus
  // vllm-project/vllm#52197 hunk 2, BEYOND-PIN). Upstream writes the result back
  // onto the draft's hf_config and calls update_arch_(); this returns it,
  // because our loader dispatches on the value instead of mutating a config
  // object it will read again.
  //
  // Upstream's three branches, in upstream's own order:
  //   1. "DSparkDraftModel" + model_type "qwen3" -> ["Qwen3DSparkModel"] (#52197)
  //   2. neither "Qwen3DSparkModel" nor "Gemma4DSparkModel" -> the DeepSeek-V4
  //      rewrite (model_type "deepseek_v4", architectures ["DSparkDraftModel"])
  //   3. "Gemma4DSparkModel" -> key normalization only; the architecture stands
  //
  // The order is the whole of #52197: branch 1 is a guard in front of a
  // catch-all, so the specific case is claimed before the general one rewrites
  // it.
  //
  // TWO TRACKED DIVERGENCES, and both are deliberate.
  //
  // DIVERGENCE 1, at branch 2. Upstream rewrites the config and lets the
  // DeepSeek-V4 model path take it. That path does not exist here:
  // DeepseekV4Model is a stub that fails a VT_CHECK
  // (src/vllm/model_executor/models/deepseek_v4_registry.cpp) and the lane needs
  // two Sparks. Mirroring the rewrite would send the draft into the stub and
  // fail on an internal check instead of naming the missing arm, so branch 2
  // REFUSES by name — which is what AGENTS.md requires of an unimplemented arm.
  // .agents/specs/dspark-qwen3-routing.md §7 R2 records the decision.
  //
  // DIVERGENCE 2, at branch 3. Upstream leaves a Gemma4 draft's architecture
  // ALONE — "Gemma4DSparkModel" stands and only its keys are normalized
  // (speculative.py:945-961) — because upstream has a Gemma4 DSpark model class
  // to dispatch to. This engine does not: LoadQwen3DSpark is the one DSpark
  // draft lane, and both published layouts load through it
  // (src/vllm/entrypoints/model_loader.cpp::LoadDsparkDraft). So branch 3
  // COLLAPSES onto "Qwen3DSparkModel" together with branch 1, and this function
  // answers exactly one lane or throws. The collapse is what makes a
  // lane-dispatch guard at the call site dead code today; it is undone by the
  // change that lands a second lane, not before.
  // .agents/specs/dspark-qwen3-routing.md §3 designs it this way.
  static std::string ResolveDsparkArchitecture(
      const std::vector<std::string>& architectures,
      const std::string& model_type) {
    bool has_draft_model = false;
    bool has_qwen3 = false;
    bool has_gemma4 = false;
    for (const std::string& arch : architectures) {
      if (arch == "DSparkDraftModel") has_draft_model = true;
      if (arch == "Qwen3DSparkModel") has_qwen3 = true;
      if (arch == "Gemma4DSparkModel") has_gemma4 = true;
    }
    if (has_draft_model && model_type == "qwen3") {
      return "Qwen3DSparkModel";  // #52197 hunk 2, the leading branch
    }
    if (!has_qwen3 && !has_gemma4) {
      std::string listed;
      for (const std::string& arch : architectures) {
        if (!listed.empty()) listed += ", ";
        listed += "\"" + arch + "\"";
      }
      throw std::invalid_argument(
          "speculative-config: this DSpark draft routes to the DeepSeek-V4 "
          "DSpark lane, which is not implemented here. Upstream rewrites a "
          "draft whose architectures name neither \"Qwen3DSparkModel\" nor "
          "\"Gemma4DSparkModel\" onto model_type \"deepseek_v4\" and "
          "architectures [\"DSparkDraftModel\"] "
          "(vllm/config/speculative.py:934-944 @ 555967922), and this engine "
          "carries only a DeepseekV4Model stub for that lane, which also needs "
          "two Sparks. Got architectures [" +
          listed + "] with model_type \"" + model_type +
          "\". Owed by row SPEC-DSPARK-QWEN3-ROUTING "
          "(.agents/specs/dspark-qwen3-routing.md).");
    }
    return "Qwen3DSparkModel";
  }

  // ResolveDspark: build the scheduler-facing SpeculativeConfig for a DSpark
  // semi-autoregressive BLOCK draft (SPEC-DSPARK W1). Mirrors the DSpark path of
  // speculative.py __post_init__ exactly:
  //
  //   * `parallel_drafting = True` for ("dflash", "dspark")            :963-964
  //   * `n_predict` defaults k when the draft config carries one       :973-979
  //     (the Gemma4 DSpark branch sets n_predict = block_size, :957-961; a native
  //     Qwen3DSparkModel config carries `block_size` but NOT `n_predict`, so its
  //     k is REQUIRED from the user — which is why the upstream e2e test passes
  //     num_speculative_tokens=7 explicitly)
  //   * k above n_predict must be a multiple of it                     :980-988
  //   * k missing with no n_predict is an error                        :990-994
  //   * k below a DSV4-style `dspark_block_size` is a HARD error       :1003-1027
  //     ("Smaller values produce incorrect output" — garbled, not merely lower
  //     acceptance, because the block/Markov machinery gets an unsupported layout)
  //
  // `n_predict` / `dspark_block_size` are std::nullopt when the draft's HF config
  // carries no such key, mirroring upstream's getattr(..., None).
  //
  // `block_size_key` names the config key `dspark_block_size` was read from, and
  // exists only so the refusal can quote it. Upstream has one key and can name it
  // literally; we have two, because the fallback below reads `block_size` on the
  // published Qwen3 drafts (the divergence argued in
  // .agents/specs/dspark-block-size-guard.md section 2). A message that says
  // `dspark_block_size` when the 7 came from `block_size` sends the user looking
  // for a key their checkpoint does not carry, which on both published drafts is
  // always the case. It defaults to upstream's key, so a caller that supplies the
  // upstream field gets upstream's wording unchanged.
  static SpeculativeConfig ResolveDspark(
      std::optional<int> n_predict, std::optional<int> dspark_block_size,
      std::optional<int> user_num_speculative_tokens,
      std::string_view block_size_key = "dspark_block_size") {
    SpeculativeConfig cfg;
    cfg.method = "dspark";
    cfg.parallel_drafting = true;
    cfg.n_predict = n_predict.value_or(0);

    std::optional<int> k = user_num_speculative_tokens;
    if (n_predict.has_value()) {
      if (!k.has_value()) {
        k = *n_predict;  // "Default to max value defined in draft model config."
      } else if (*k > *n_predict && *n_predict > 0 && *k % *n_predict != 0) {
        throw std::invalid_argument(
            "num_speculative_tokens must be divisible by n_predict "
            "(the DSpark draft block depth)");
      }
    }
    if (!k.has_value()) {
      throw std::invalid_argument(
          "speculative-config: a DSpark draft model was provided, but "
          "\"num_speculative_tokens\" was not provided");
    }
    if (dspark_block_size.has_value() && *k < *dspark_block_size) {
      // speculative.py:1021-1026 verbatim, including the third sentence that
      // says which value to use. It was dropped when the check was first ported
      // and is restored here: the first two sentences tell the user the k is
      // wrong, and only the third tells them what to type.
      throw std::invalid_argument(
          "speculative-config: DSpark requires num_speculative_tokens >= " +
          std::string(block_size_key) + " (" +
          std::to_string(*dspark_block_size) + "); got " + std::to_string(*k) +
          ". Smaller values produce incorrect output. Use "
          "num_speculative_tokens=" +
          std::to_string(*dspark_block_size) + " or larger (e.g. 7).");
    }
    cfg.num_speculative_tokens = k;
    return cfg;
  }

  // ResolveNgram: build the scheduler-facing SpeculativeConfig for the draft-free
  // n-gram proposer (SPEC-NGRAM, ROAD-V1-D3). Mirrors speculative.py:734-762 —
  // num_speculative_tokens (k) is REQUIRED (:1224-1234), and the prompt_lookup
  // window defaults to 5/5 when the user gives neither, or fills the missing bound
  // from the given one (:737-758). There is NO draft model and NO n_predict-module
  // constraint. NumLookaheadTokens() returns 0 for ngram (the scheduler reserves no
  // extra lookahead slots, scheduler.py:250-265 — none of the ngram branches fire),
  // exactly like upstream; the drafts are verified via the scheduled spec tokens.
  static SpeculativeConfig ResolveNgram(int k, std::optional<int> user_min,
                                        std::optional<int> user_max) {
    SpeculativeConfig cfg;
    cfg.method = "ngram";
    cfg.n_predict = 0;
    if (k <= 0) {
      throw std::invalid_argument(
          "speculative-config: method \"ngram\" requires a positive "
          "num_speculative_tokens");
    }
    cfg.num_speculative_tokens = k;
    // Default resolution (speculative.py:737-758).
    if (!user_min.has_value() && !user_max.has_value()) {
      cfg.prompt_lookup_min = 5;
      cfg.prompt_lookup_max = 5;
    } else if (!user_min.has_value()) {
      cfg.prompt_lookup_min = user_max;
      cfg.prompt_lookup_max = user_max;
    } else if (!user_max.has_value()) {
      cfg.prompt_lookup_min = user_min;
      cfg.prompt_lookup_max = user_min;
    } else {
      cfg.prompt_lookup_min = user_min;
      cfg.prompt_lookup_max = user_max;
    }
    if (*cfg.prompt_lookup_min < 1 || *cfg.prompt_lookup_max < 1) {
      throw std::invalid_argument(
          "speculative-config: prompt_lookup_min/max must be >= 1");
    }
    if (*cfg.prompt_lookup_min > *cfg.prompt_lookup_max) {
      throw std::invalid_argument(
          "speculative-config: prompt_lookup_min must be <= prompt_lookup_max");
    }
    return cfg;
  }

  // MaybeOverrideDraftMaxPositionEmbeddings: raise an EAGLE draft's
  // max_position_embeddings up to the target's max_model_len.
  //
  // Ported from: vllm/config/speculative.py
  //   SpeculativeConfig._maybe_override_draft_max_position_embeddings
  //   @ 32e657e68 (vllm#49343 "[BugFix] eagle draft max position embeddings",
  //   underlying issue #48894).
  //
  // EAGLE/eagle3 drafts SHARE the target's positional space: the proposer feeds
  // the draft positions up to the target's max_model_len, while the draft's
  // max_position_embeddings sizes its rotary cos_sin_cache. A draft checkpoint
  // that ships a smaller value (e.g. 2048 for yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
  // vs a 131072-context target) under-sizes that cache, so the gather at a
  // target-scale position reads OUT OF BOUNDS — a device-side assert under
  // graph capture, silent garbage in eager mode. This clamp raises the draft
  // value to the target's max_model_len (never lowers it), for eagle/eagle3 ONLY:
  // an independent AR draft_model may legitimately have a smaller context, so its
  // value must NOT be resized (upstream test_independent_draft_model_keeps_its_
  // own_limit).
  //
  // Signature note: upstream mutates draft_hf_config.max_position_embeddings in
  // place and `getattr(..., None)` treats a missing attribute as "leave alone".
  // Our SpeculativeConfig does not yet carry the draft ModelConfig/hf_config
  // (draft-config resolution is DEFERRED to the eagle/eagle3 model-loader claim),
  // so the value is passed by reference as std::optional<int> — nullopt mirrors
  // "attribute missing" (no override). Returns true iff it raised the value,
  // mirroring upstream's logger.info (the caller emits the log line). The live
  // call site lands with eagle/eagle3 draft-config resolution; this is the tested
  // clamp it will call, and the OOB it prevents is exactly our
  // RotaryEmbeddingBase cos_sin_cache (built to max_position_embeddings_).
  static bool MaybeOverrideDraftMaxPositionEmbeddings(
      std::optional<int>& draft_max_position_embeddings,
      int target_max_model_len) {
    if (!draft_max_position_embeddings.has_value() ||
        *draft_max_position_embeddings >= target_max_model_len) {
      return false;
    }
    draft_max_position_embeddings = target_max_model_len;
    return true;
  }

  // num_speculative_tokens resolved to a concrete k (falls back to n_predict).
  int ResolvedNumSpeculativeTokens() const {
    return num_speculative_tokens.value_or(n_predict);
  }

  // use_ngram (speculative.py method=="ngram"): the draft-FREE proposer. It is NOT
  // a target-hidden-state method, so it is deliberately absent from use_eagle()
  // and contributes 0 to NumLookaheadTokens() (scheduler.py:250-265).
  bool use_ngram() const { return method == "ngram"; }

  // use_eagle (speculative.py:1163-1170): true for the target-hidden-state
  // methods — eagle/eagle3/mtp/dflash/dspark. This is the predicate that sets the
  // scheduler's num_lookahead_tokens = k for MTP (scheduler.py:284-286).
  bool use_eagle() const {
    return method == "eagle" || method == "eagle3" || method == "mtp" ||
           method == "dflash" || method == "dspark";
  }

  // uses_draft_model (speculative.py:1195): true only for the "draft_model"
  // method (a separate small model). Also sets num_lookahead_tokens = k
  // (scheduler.py:287-288).
  bool uses_draft_model() const { return method == "draft_model"; }

  // use_dflash (speculative.py:1172): DFlash needs one EXTRA lookahead slot
  // (scheduler.py:289-292, num_lookahead = k + 1).
  bool use_dflash() const { return method == "dflash"; }

  // use_dspark (speculative.py:1333-1334). DSpark is deliberately NOT part of
  // use_dflash(): the scheduler branches on the two separately and reserves k
  // lookahead slots for DSpark against DFlash's k + 1 (scheduler.py:256-265),
  // because DSpark's anchor query is itself the first prediction rather than a
  // separate bonus query.
  bool use_dspark() const { return method == "dspark"; }

  // NumLookaheadTokens: the scheduler's num_lookahead_tokens for this config
  // (scheduler.py:275-292). This is the value threaded into allocate_slots so the
  // verify slots are reserved ahead of time. 0 for a method the scheduler does
  // not look ahead for.
  int NumLookaheadTokens() const {
    const int k = ResolvedNumSpeculativeTokens();
    if (use_dflash()) {
      return k + 1;  // extra slot for the in-fill last-sampled query.
    }
    if (use_eagle() || uses_draft_model()) {
      return k;
    }
    return 0;
  }
};

// Parse vLLM's `--speculative-config` JSON (SPEC-MTP I5d). Mirrors the subset of
// vllm/engine/arg_utils.py speculative-config handling the CLI needs: the
// `method` string and the optional `num_speculative_tokens`. The accepted
// methods at this pin are "mtp", "dflash" (SPEC-DFLASH D4), "dspark"
// (SPEC-DSPARK W1), "ngram" (SPEC-NGRAM) and "draft_model" (SPEC-DRAFT-MODEL);
// any other method throws. Every key is judged by name and none is dropped
// (#1160). The returned config has n_predict == 0 — the loader resolves it from
// the model's mtp_num_hidden_layers via SpeculativeConfig::ResolveMtp (MTP) or
// the draft's dflash_config via ResolveDflash (DFlash) once the HF config is
// known. Throws std::invalid_argument on a malformed document / unknown method.
//
// SPEC-DRAFTER-CHAIN W1 (#1522): the document also accepts this engine's own
// `vllm_cpp` extension object, whose only key is `drafter_chain` — a JSON array
// of speculator entries in preference order, filling `drafter_chain` above.
// `method` is REQUIRED exactly when that field is ABSENT: the two are mutually
// exclusive, so a chain document leaves `method` empty and a chain-free one is
// unchanged in every respect (spec `.agents/specs/drafter-chain.md` D7). This
// function parses and validates the chain; `LoadedEngine::ResolveSpecConfig`
// refuses one by name, because nothing resolves a chain at this wave.
SpeculativeConfig ParseSpeculativeConfigJson(const std::string& json_text);

}  // namespace vllm

#endif  // VLLM_CONFIG_SPECULATIVE_H_
