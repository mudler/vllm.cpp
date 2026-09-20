// Stateful public API double for the #3075 CLI, not a model oracle.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <vector>
#include <nlohmann/json.hpp>
using llama_token = int32_t;
using llama_pos = int32_t;
using llama_seq_id = int32_t;
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_BF16 };
enum llama_flash_attn_type { LLAMA_FLASH_ATTN_TYPE_DISABLED, LLAMA_FLASH_ATTN_TYPE_AUTO };
enum ggml_log_level { GGML_LOG_LEVEL_INFO };
inline bool fault(const char* s) { return std::getenv("PROBE_FAULT") && std::strcmp(std::getenv("PROBE_FAULT"), s) == 0; }
inline void event(nlohmann::json j) { std::ofstream(std::getenv("PROBE_EVENTS"), std::ios::app) << j.dump() << '\n'; }
struct llama_model {};
struct llama_vocab {};
struct llama_model_params { int n_gpu_layers = 0; };
struct llama_context_params {
    uint32_t n_ctx = 0, n_batch = 2048, n_ubatch = 512, n_seq_max = 1;
    int n_threads = 4, n_threads_batch = 4;
    bool offload_kqv = true, op_offload = true, kv_unified = true;
    ggml_type type_k = GGML_TYPE_F32, type_v = GGML_TYPE_F32;
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
};
struct llama_batch {
    int32_t n_tokens;
    llama_token* token; float* embd; llama_pos* pos; int32_t* n_seq_id;
    llama_seq_id** seq_id; int8_t* logits;
};
struct llama_context {
    llama_context_params p;
    int id; bool synced = true;
    std::map<int, int> outputs, next;
    std::vector<float> borrowed = std::vector<float>(151936);
};
inline void llama_log_set(void (*)(ggml_log_level, const char*, void*), void*) {}
inline void llama_backend_init() { event({{"op", "backend_init"}}); }
inline void llama_backend_free() { event({{"op", "backend_free"}}); }
inline bool llama_supports_gpu_offload() { return !fault("gpu_unavailable"); }
inline llama_model_params llama_model_default_params() { return {}; }
inline llama_context_params llama_context_default_params() {
    llama_context_params p;
    if (fault("defaults")) p.n_batch = 32;
    if (fault("ubatch")) p.n_ubatch = 32;
    if (fault("threads")) p.n_threads = 8;
    if (fault("threads_batch")) p.n_threads_batch = 8;
    if (fault("offload")) p.offload_kqv = false;
    if (fault("op_offload")) p.op_offload = false;
    if (fault("flash")) p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return p;
}
inline llama_model* llama_model_load_from_file(const char*, llama_model_params p) {
    event({{"op", "model_load"}, {"layers", p.n_gpu_layers}});
    return fault("model") ? nullptr : new llama_model;
}
inline void llama_model_free(llama_model* p) { event({{"op", "model_free"}}); delete p; }
inline const llama_vocab* llama_model_get_vocab(const llama_model*) { static llama_vocab v; return &v; }
inline int llama_vocab_n_tokens(const llama_vocab*) { return fault("vocab") ? 10 : 151936; }
inline int llama_tokenize(const llama_vocab*, const char* text, int, llama_token* out, int cap, bool bos, bool special) {
    const char* prompts[] = {"The capital city of France is", "The three primary colors are", "Water boils at a temperature of", "The Pythagorean theorem states that", "In 1969, humans first walked on", "A prime number is a natural number"};
    int lengths[] = {6, 5, 6, 7, 8, 9};
    if (bos || !special) return -1;
    for (int s = 0; s < 6; ++s) if (std::strcmp(text, prompts[s]) == 0) {
        int n = lengths[s]; if (cap < n) return -n;
        for (int i = 0; i < n; ++i) out[i] = 100 * s + i + 1;
        if (fault("tokenize")) out[0]++;
        event({{"op", "tokenize"}, {"sequence", s}});
        return n;
    }
    return -1;
}
inline llama_context* llama_init_from_model(llama_model*, llama_context_params p) {
    static int id = 0;
    event({{"op", "context_new"}, {"context", id}, {"ctx", p.n_ctx}, {"slots", p.n_seq_max}, {"unified", p.kv_unified}, {"k", p.type_k}, {"v", p.type_v}});
    if (fault("context")) return nullptr;
    return new llama_context{p, id++};
}
inline void llama_free(llama_context* p) { event({{"op", "context_free"}, {"context", p->id}}); delete p; }
inline uint32_t llama_n_ctx(const llama_context* c) { return fault("resolved_ctx") ? 4096 : c->p.n_ctx; }
inline uint32_t llama_n_ctx_seq(const llama_context*) { return fault("resolved") ? 1024 : 2048; }
inline uint32_t llama_n_seq_max(const llama_context* c) { return fault("resolved_slots") ? 2 : c->p.n_seq_max; }
inline uint32_t llama_n_batch(const llama_context* c) { return fault("resolved_batch") ? 1024 : c->p.n_batch; }
inline uint32_t llama_n_ubatch(const llama_context* c) { return fault("resolved_ubatch") ? 256 : c->p.n_ubatch; }
inline int llama_n_threads(llama_context* c) { return fault("resolved_threads") ? 2 : c->p.n_threads; }
inline int llama_n_threads_batch(llama_context* c) { return fault("resolved_threads_batch") ? 2 : c->p.n_threads_batch; }
inline llama_batch llama_batch_init(int n, int, int) {
    event({{"op", "batch_new"}});
    if (fault("batch")) return {};
    llama_batch b{0, new int[n], nullptr, new int[n], new int[n], new int*[n+1], new int8_t[n]};
    for (int i = 0; i < n; ++i) b.seq_id[i] = new int[1];
    b.seq_id[n] = nullptr;
    return b;
}
inline void llama_batch_free(llama_batch b) {
    event({{"op", "batch_free"}});
    if (b.seq_id) { for (int i = 0; b.seq_id[i]; ++i) delete[] b.seq_id[i]; delete[] b.seq_id; }
    delete[] b.token; delete[] b.pos; delete[] b.n_seq_id; delete[] b.logits;
}
inline int llama_decode(llama_context* c, llama_batch b) {
    nlohmann::json rows = nlohmann::json::array();
    c->outputs.clear();
    for (int i = 0; i < b.n_tokens; ++i) {
        int s = b.seq_id[i][0];
        rows.push_back({{"token_id", b.token[i]}, {"position", b.pos[i]}, {"sequence_id", s}, {"logits_requested", bool(b.logits[i])}});
        if (b.n_seq_id[i] != 1 || b.pos[i] != c->next[s]++ || b.token[i] != s*100+b.pos[i]+1) return -7;
        if (b.logits[i]) c->outputs[i] = s;
    }
    event({{"op", "decode"}, {"context", c->id}, {"rows", rows}});
    c->synced = false;
    std::fill(c->borrowed.begin(), c->borrowed.end(), -9999);
    return fault("decode") ? -1 : 0;
}
inline void llama_synchronize(llama_context* c) { c->synced = true; event({{"op", "sync"}, {"context", c->id}}); }
inline float* llama_get_logits_ith(llama_context* c, int i) {
    event({{"op", "get"}, {"context", c->id}, {"index", i}});
    if (fault("null") || !c->outputs.count(i)) return nullptr;
    int s = c->outputs.at(i);
    for (int t = 0; t < 151936; ++t) c->borrowed[t] = float(-t) / 8 + s;
    c->borrowed[42] = c->borrowed[43] = 100 + s;
    if (fault("nonfinite")) c->borrowed[151935] = std::numeric_limits<float>::infinity();
    return c->borrowed.data();
}
