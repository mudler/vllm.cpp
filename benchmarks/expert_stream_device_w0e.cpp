// w0e_gen — ENG-EXPERT-STREAM-DEVICE W0e measurement harness (issue #1124).
//
// A THIN CLIENT of the public C ABI (include/vllm.h) only: vllm_engine_load +
// vllm_complete, with a custom logits processor used as a PURE OBSERVER. The
// processor edits nothing, so the argmax the sampler takes is byte-identical to
// a run without it; it exists because it is the only public surface that hands
// back the request's GENERATED TOKEN IDS, which is what G0-CORRECT gates on,
// and because it is invoked once per decode step, which is what gives the
// per-token arrival deltas G0-SPEED needs.
//
// The AUTHORITATIVE id list comes from vllm_complete_tokens (ABI v13), which
// fills a caller buffer with the generated ids. The processor's own token_ids
// view is recorded too, but it is NOT trusted for the gate: the capi suite
// records that under the async scheduler that view can LAG the emitted tokens,
// because the bookkeeping is fed back by update_from_output. Pass --prompt-ids
// to take that path; without it the harness falls back to vllm_complete, whose
// only id evidence is the processor's lagging view.
//
// It builds as the `expert-stream-device-w0e` target, and links `vllm::shared`
// rather than `vllm::vllm`, so it exercises the packaged C ABI and nothing else
// and cannot reach an internal header even by accident. It was originally left
// out of CMake beside marlin_moe_standalone.cpp, on the reading that a
// gate instrument is not a shipped capability. That is the wrong trade for THIS
// file: an instrument nothing compiles rots silently against the very ABI it
// measures, and the numbers in `.agents/benchmark-record.md` cannot be
// reproduced from a file that no longer builds. So the project builds it.
//
//   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
//     -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a
//     -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass -DVLLM_CPP_TRITON=ON
//   cmake --build build --target expert-stream-device-w0e -j 6
//
// One arm, with the lane on and its statistics line at every step so that the
// after-prefill snapshot and the final one are both in the log. (Written on one
// line: a backslash continuation inside a `//` comment is what -Werror=comment
// rejects, and this file is compiled with the project's flags now.)
//
//   VT_GGUF_PREFAULT=0 VT_MOE_EXPERT_STREAM=1 VT_MOE_EXPERT_STREAM_SLOTS=8000
//   VT_MOE_EXPERT_STREAM_STATS_EVERY=1 ./build/examples/expert-stream-device-w0e
//   --model <shard-1>.gguf --device cuda --max-tokens 32 --max-num-seqs 1
//   --prompt-ids 760,6511,314,9338,369
//
// `VT_GGUF_PREFAULT=0` is load-bearing for a model larger than memory, and
// STATS_EVERY=1 is load-bearing for a short run: at the default 16 a healthy
// 32-token run prints no periodic line at all, which looks exactly like a dead
// lane.
#include "vllm.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Obs {
  std::vector<double> t;                    // epoch seconds at each LP call
  std::vector<std::vector<int32_t>> ids;    // ids seen at each LP call
};

double Now() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// PURE OBSERVER: records and returns. It must not touch `logits`.
void ObserveLp(const int32_t* token_ids, int32_t n, float* logits,
               int32_t vocab, void* ud) {
  (void)logits;
  (void)vocab;
  Obs* o = static_cast<Obs*>(ud);
  o->t.push_back(Now());
  o->ids.emplace_back(token_ids, token_ids + (n > 0 ? n : 0));
  std::fprintf(stderr, "[w0e] lp_call=%zu t=%.6f n_ids=%d\n", o->t.size(),
               o->t.back(), n);
  std::fflush(stderr);
}

long ProcKb(const char* key) {
  std::FILE* f = std::fopen("/proc/self/status", "r");
  if (f == nullptr) return -1;
  char line[512];
  long v = -1;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, key, std::strlen(key)) == 0) {
      v = std::atol(line + std::strlen(key));
      break;
    }
  }
  std::fclose(f);
  return v;
}

void PrintMem(const char* tag) {
  std::printf("W0E_MEM %s VmRSS_kB=%ld VmHWM_kB=%ld\n", tag, ProcKb("VmRSS:"),
              ProcKb("VmHWM:"));
  std::fflush(stdout);
}

const char* NextArg(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "w0e_gen: missing value after %s\n", argv[i]);
    std::exit(2);
  }
  return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, prompt = "The capital of France is", tag = "run";
  std::vector<int32_t> prompt_ids;
  int32_t device = 0;  // 0 auto, 1 cpu, 2 cuda
  int max_tokens = 33;
  long long kv_bytes = 0;
  int max_num_seqs = 1;

  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    if (f == "--model") {
      model = NextArg(argc, argv, i);
    } else if (f == "--prompt") {
      prompt = NextArg(argc, argv, i);
    } else if (f == "--tag") {
      tag = NextArg(argc, argv, i);
    } else if (f == "--prompt-ids") {
      const std::string s = NextArg(argc, argv, i);
      size_t p = 0;
      while (p < s.size()) {
        size_t q = s.find_first_of(", ", p);
        if (q == std::string::npos) q = s.size();
        if (q > p) prompt_ids.push_back(std::atoi(s.substr(p, q - p).c_str()));
        p = q + 1;
      }
    } else if (f == "--max-tokens") {
      max_tokens = std::atoi(NextArg(argc, argv, i));
    } else if (f == "--kv-cache-memory") {
      kv_bytes = std::atoll(NextArg(argc, argv, i));
    } else if (f == "--max-num-seqs") {
      max_num_seqs = std::atoi(NextArg(argc, argv, i));
    } else if (f == "--device") {
      const std::string d = NextArg(argc, argv, i);
      if (d == "auto") {
        device = 0;
      } else if (d == "cpu") {
        device = 1;
      } else if (d == "cuda") {
        device = 2;
      } else {
        std::fprintf(stderr, "w0e_gen: bad --device %s\n", d.c_str());
        return 2;
      }
    } else {
      std::fprintf(stderr, "w0e_gen: unknown flag %s\n", f.c_str());
      return 2;
    }
  }
  if (model.empty()) {
    std::fprintf(stderr, "w0e_gen: --model required\n");
    return 2;
  }

  std::printf("W0E_TAG=%s\nW0E_DEVICE=%d\nW0E_ABI=%d\nW0E_MAX_TOKENS=%d\n",
              tag.c_str(), static_cast<int>(device),
              static_cast<int>(vllm_abi_version()), max_tokens);
  std::printf("W0E_PROMPT=%s\n", prompt.c_str());
  std::fflush(stdout);
  PrintMem("start");

  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = model.c_str();
  mp.device = device;
  mp.max_num_seqs = max_num_seqs;
  // 0.0 is the ABI's "unset" spelling; vllm-cli does the same so a plain run
  // does not look like an explicit ask (#1165).
  mp.gpu_memory_utilization = 0.0;
  if (kv_bytes > 0) mp.kv_cache_memory_bytes = kv_bytes;

  const double t_load0 = Now();
  std::printf("W0E_LOAD_START=%.6f\n", t_load0);
  std::fflush(stdout);
  vllm_engine* eng = nullptr;
  vllm_status st = vllm_engine_load(&mp, &eng);
  const double t_load1 = Now();
  if (st != VLLM_OK) {
    std::printf("W0E_LOAD_STATUS=%d\nW0E_LOAD_ERROR=%s\n", static_cast<int>(st),
                vllm_last_error());
    std::printf("W0E_LOAD_SECS=%.3f\nW0E_RESULT=LOAD_FAILED\n",
                t_load1 - t_load0);
    std::fflush(stdout);
    return 1;
  }
  std::printf("W0E_LOAD_SECS=%.3f\nW0E_LOAD_END=%.6f\n", t_load1 - t_load0,
              t_load1);
  std::fflush(stdout);
  PrintMem("after_load");

  Obs obs;
  vllm_sampling_params sp = vllm_sampling_params_default();
  sp.temperature = 0.0f;  // greedy
  sp.max_tokens = max_tokens;
  sp.logits_processor = &ObserveLp;
  sp.logits_processor_user_data = &obs;

  vllm_completion out{};
  std::vector<int32_t> out_ids(256);
  int32_t n_out_ids = 0;
  const bool tokens_path = !prompt_ids.empty();
  std::printf("W0E_ENTRY=%s\n", tokens_path ? "vllm_complete_tokens" : "vllm_complete");
  if (tokens_path) {
    std::printf("W0E_PROMPT_IDS_N=%zu\nW0E_PROMPT_IDS=", prompt_ids.size());
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
      std::printf("%s%d", i == 0 ? "" : ",", prompt_ids[i]);
    }
    std::printf("\n");
  }
  std::fflush(stdout);
  const double t_gen0 = Now();
  if (tokens_path) {
    st = vllm_complete_tokens(eng, prompt_ids.data(),
                              static_cast<int32_t>(prompt_ids.size()), &sp,
                              out_ids.data(),
                              static_cast<int32_t>(out_ids.size()), &n_out_ids,
                              &out);
  } else {
    st = vllm_complete(eng, prompt.c_str(), &sp, &out);
  }
  const double t_gen1 = Now();
  std::printf("W0E_GEN_STATUS=%d\nW0E_GEN_SECS=%.3f\n", static_cast<int>(st),
              t_gen1 - t_gen0);
  if (st != VLLM_OK) {
    std::printf("W0E_GEN_ERROR=%s\nW0E_RESULT=GEN_FAILED\n", vllm_last_error());
    std::fflush(stdout);
    vllm_engine_free(eng);
    return 1;
  }
  std::printf("W0E_PROMPT_TOKENS=%d\nW0E_COMPLETION_TOKENS=%d\nW0E_FINISH=%s\n",
              out.prompt_tokens, out.completion_tokens,
              out.finish_reason != nullptr ? out.finish_reason : "(none)");
  std::printf("W0E_TEXT_BEGIN\n%s\nW0E_TEXT_END\n",
              out.text != nullptr ? out.text : "");

  // Per-step arrival: LP call k fires when step k's logits are ready, i.e. at
  // the moment token k becomes available. k == 1 is the end of prefill (TTFT).
  std::printf("W0E_LP_CALLS=%zu\n", obs.t.size());
  for (size_t k = 0; k < obs.t.size(); ++k) {
    const double prev = (k == 0) ? t_gen0 : obs.t[k - 1];
    std::printf("W0E_STEP k=%zu t=%.6f dt=%.6f n_ids=%zu\n", k + 1, obs.t[k],
                obs.t[k] - prev, obs.ids[k].size());
  }
  // AUTHORITATIVE: the ids vllm_complete_tokens wrote.
  if (tokens_path) {
    std::printf("W0E_OUT_IDS_N=%d\nW0E_OUT_IDS=", n_out_ids);
    for (int32_t i = 0; i < n_out_ids; ++i) {
      std::printf("%s%d", i == 0 ? "" : ",", out_ids[static_cast<size_t>(i)]);
    }
    std::printf("\n");
  }
  // ADVISORY: the longest id prefix the processor's view showed. Recorded so a
  // reader can see the lag rather than infer it; never the gate's evidence.
  if (!obs.ids.empty()) {
    size_t best = 0;
    for (size_t k = 0; k < obs.ids.size(); ++k) {
      if (obs.ids[k].size() >= obs.ids[best].size()) best = k;
    }
    const std::vector<int32_t>& ids = obs.ids[best];
    std::printf("W0E_LP_IDS_N=%zu\nW0E_LP_IDS=", ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      std::printf("%s%d", i == 0 ? "" : ",", ids[i]);
    }
    std::printf("\n");
  }
  PrintMem("after_gen");
  std::printf("W0E_RESULT=OK\n");
  std::fflush(stdout);

  vllm_completion_free(&out);
  vllm_engine_free(eng);
  PrintMem("after_free");
  std::fflush(stdout);
  return 0;
}
