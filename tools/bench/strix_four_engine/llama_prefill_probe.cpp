// #3075: public-API first-prefill diagnostic, llama.cpp @ 10bf611e533d81f7.
// This synchronizing capture is not a correctness gate or throughput benchmark.
#include <llama.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
constexpr int kVocab = 151936;
const std::vector<std::string> kPrompts = {
    "The capital city of France is", "The three primary colors are",
    "Water boils at a temperature of", "The Pythagorean theorem states that",
    "In 1969, humans first walked on", "A prime number is a natural number"};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void emit(json record) {
    record["schema"] = 1;
    std::cout << record.dump() << '\n';
    std::cout.flush();
    require(bool(std::cout), "output failure");
}

json read_config(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream), "cannot open config");
    std::string bytes(1048577, '\0');
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(!stream.bad(), "config read failure");
    bytes.resize(static_cast<size_t>(stream.gcount()));
    require(bytes.size() <= 1048576, "config exceeds 1 MiB");
    std::set<std::string> keys;
    auto config = json::parse(bytes, [&](int, json::parse_event_t event, json& value) {
        if (event == json::parse_event_t::key)
            require(keys.insert(value.get<std::string>()).second, "duplicate config key");
        return true;
    });
    require(config.is_object() && config.size() == 4 && config.contains("schema") &&
            config.contains("gguf") && config.contains("prompts") && config.contains("prompt_ids"),
            "config keys mismatch");
    require(config["schema"].is_number_integer() && config["schema"] == 1, "invalid schema");
    require(config["gguf"].is_string() && !config["gguf"].get<std::string>().empty(), "invalid GGUF path");
    require(config["prompts"] == json(kPrompts), "six canonical raw prompts required");
    require(config["prompt_ids"].is_array() && config["prompt_ids"].size() == 6, "six token arrays required");
    for (const auto& ids : config["prompt_ids"]) {
        require(ids.is_array() && !ids.empty() && ids.size() <= 2048, "invalid prompt IDs");
        for (const auto& id : ids)
            require(id.is_number_integer() && id >= 0 && id < kVocab, "token ID out of range");
    }
    const int lengths[] = {6, 5, 6, 7};
    for (int i = 0; i < 4; ++i)
        require(config["prompt_ids"][i].size() == static_cast<size_t>(lengths[i]), "experiment lengths changed");
    return config;
}

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};
struct Batch {
    llama_batch value = llama_batch_init(24, 0, 1);
    ~Batch() { llama_batch_free(value); }
    void check() const {
        require(value.token && value.pos && value.n_seq_id && value.seq_id && value.logits, "batch allocation failed");
        for (int i = 0; i < 24; ++i) require(value.seq_id[i] != nullptr, "batch sequence allocation failed");
    }
};
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;

json resolved(llama_context* ctx) {
    json result = {{"n_ctx", llama_n_ctx(ctx)}, {"n_ctx_seq", llama_n_ctx_seq(ctx)},
                   {"n_seq_max", llama_n_seq_max(ctx)}, {"n_batch", llama_n_batch(ctx)},
                   {"n_ubatch", llama_n_ubatch(ctx)}, {"n_threads", llama_n_threads(ctx)},
                   {"n_threads_batch", llama_n_threads_batch(ctx)}};
    require(result == json({{"n_ctx", 8192}, {"n_ctx_seq", 2048}, {"n_seq_max", 4},
                           {"n_batch", 2048}, {"n_ubatch", 512}, {"n_threads", 4},
                           {"n_threads_batch", 4}}), "resolved context mismatch");
    return result;
}

void capture(llama_context* ctx, const json& submitted, const json& ids,
             const char* mode, int repetition, int index, const json& row) {
    // llama-context.cpp:3744: this getter borrows context-owned output storage.
    const float* borrowed = llama_get_logits_ith(ctx, index);
    require(borrowed != nullptr, "missing logits");
    std::vector<float> scores(borrowed, borrowed + kVocab);
    for (float x : scores) require(std::isfinite(x), "nonfinite logits");
    std::vector<int> order(kVocab);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin()+10, order.end(), [&](int a, int b) {
        return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
    });
    json top = json::array();
    for (int i = 0; i < 10; ++i) top.push_back({{"token_id", order[i]}, {"logit", scores[order[i]]}});
    int sequence = row.at("sequence_id").get<int>();
    // nlohmann's float serialization round-trips the copied FP32 values.
    emit({{"type", "scores"}, {"mode", mode}, {"repetition", repetition},
          {"sequence_id", sequence}, {"original_terminal_index", row.at("original_index")},
          {"batch_terminal_index", index}, {"prompt_ids", ids.at(sequence)},
          {"submitted_batches", submitted}, {"logits", scores}, {"argmax", order[0]}, {"top10", top}});
}

void experiment(const json& config) {
    llama_log_set([](ggml_log_level, const char* text, void*) { std::fputs(text, stderr); }, nullptr);
    Backend backend;
    require(llama_supports_gpu_offload(), "GPU offload unavailable");
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    Model model(llama_model_load_from_file(config.at("gguf").get<std::string>().c_str(), mp), llama_model_free);
    require(bool(model), "model allocation failed");
    const auto* vocab = llama_model_get_vocab(model.get());
    require(vocab && llama_vocab_n_tokens(vocab) == kVocab, "vocabulary mismatch");
    const auto& ids = config.at("prompt_ids");
    for (int s = 0; s < 6; ++s) {
        std::vector<llama_token> actual(2048);
        int count = llama_tokenize(vocab, kPrompts[s].c_str(), static_cast<int32_t>(kPrompts[s].size()),
                                   actual.data(), static_cast<int32_t>(actual.size()), false, true);
        require(count > 0 && count <= 2048, "tokenization failed");
        actual.resize(count);
        require(json(actual) == ids.at(s), "tokenization mismatch");
    }
    auto cp = llama_context_default_params();
    require(cp.n_batch == 2048 && cp.n_ubatch == 512 && cp.n_threads == 4 && cp.n_threads_batch == 4 &&
            cp.offload_kqv && cp.op_offload && cp.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO,
            "public context defaults changed");
    cp.n_ctx = 8192;
    cp.n_seq_max = 4;
    cp.kv_unified = false;
    cp.type_k = GGML_TYPE_BF16;
    cp.type_v = GGML_TYPE_BF16;
    json original = json::array();
    std::vector<std::vector<int>> separate(4);
    std::vector<int> together;
    for (int s = 0; s < 4; ++s) for (size_t p = 0; p < ids.at(s).size(); ++p) {
        int i = static_cast<int>(original.size());
        separate[s].push_back(i);
        together.push_back(i);
        original.push_back({{"original_index", i}, {"token_id", ids[s][p]}, {"position", p},
                            {"sequence_id", s}, {"logits_requested", p+1 == ids[s].size()}});
    }
    const std::vector<std::vector<int>> partitions = {
        {0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21}, {5}, {16,22}, {23}};
    bool header = false;
    int score_records = 0;
    for (const char* mode : {"independent", "combined", "partitioned"}) {
        for (int repetition = 0; repetition < 3; ++repetition) {
            bool independent = std::string(mode) == "independent";
            for (int part = 0; part < (independent ? 4 : 1); ++part) {
                Context ctx(llama_init_from_model(model.get(), cp), llama_free);
                require(bool(ctx), "context allocation failed");
                auto settings = resolved(ctx.get());
                if (!header) {
                    emit({{"type", "header"}, {"vocab_size", kVocab}, {"repetitions", 3},
                          {"score_semantics", "raw_logits"}, {"resolved", settings},
                          {"requested", {{"n_gpu_layers", -1}, {"n_ctx", 8192}, {"n_seq_max", 4},
                                         {"kv_unified", false}, {"type_k", "bfloat16"}, {"type_v", "bfloat16"},
                                         {"flash_attn", "auto"}, {"n_batch", 2048}, {"n_ubatch", 512},
                                         {"n_threads", 4}, {"n_threads_batch", 4}}}});
                    header = true;
                }
                std::vector<std::vector<int>> groups = independent ? std::vector<std::vector<int>>{separate[part]} :
                    std::string(mode) == "combined" ? std::vector<std::vector<int>>{together} : partitions;
                Batch batch;
                batch.check();
                json submitted = json::array();
                for (const auto& group : groups) {
                    auto& b = batch.value;
                    b.n_tokens = static_cast<int32_t>(group.size());
                    json rows = json::array();
                    for (int i = 0; i < b.n_tokens; ++i) {
                        const auto& row = original.at(group[i]);
                        b.token[i] = row.at("token_id").get<int>();
                        b.pos[i] = row.at("position").get<int>();
                        b.n_seq_id[i] = 1;
                        b.seq_id[i][0] = row.at("sequence_id").get<int>();
                        b.logits[i] = row.at("logits_requested").get<bool>();
                        rows.push_back(row);
                    }
                    require(llama_decode(ctx.get(), b) == 0, "decode failed");
                    llama_synchronize(ctx.get());
                    submitted.push_back(rows);
                    for (int i = 0; i < b.n_tokens; ++i) if (b.logits[i]) {
                        capture(ctx.get(), submitted, ids, mode, repetition, i, rows.at(i));
                        ++score_records;
                    }
                }
            }
        }
    }
    require(score_records == 36, "incomplete capture");
    emit({{"type", "diagnostic_complete"}, {"score_records", score_records}});
}

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);  // Turn a closed output pipe into a checked write failure.
    try {
        require(argc == 3 && std::string(argv[1]) == "--config", "usage: llama-prefill-probe --config CONFIG");
        experiment(read_config(argv[2]));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "llama-prefill-probe: " << error.what() << '\n';
        return 1;
    }
}
