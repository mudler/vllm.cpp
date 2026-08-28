// The NATIVE-layout EXL3 reader — QUANT-EXL3 W1b (#2181).
//
// The reader's job is to turn `{prefix}.{trellis,suh,svh}` into an `Exl3Weight`
// with the right geometry AND the right codebook. The second half is what this
// suite exists for.
//
// THE CODEBOOK IS SELECTED BY TENSOR PRESENCE, AND THE POLARITY IS THE OPPOSITE
// OF THE OBVIOUS GUESS. `LinearEXL3` sets `self.mcg = (self.mcg_tensor is not
// None)` and passes that BOOLEAN to `ext.reconstruct` (`exl3.py:74-77,197,223`),
// so a checkpoint shipping NO `mcg` tensor is NOT MCG — it is cb 0, the
// original QTIP 3INST. The first draft of this reader read absence as MCG.
//
// That mistake is invisible to every check a loader can make. The wrong
// multiplier yields a codebook with the SAME DISTRIBUTION and no relation to
// the right one, so the weight decodes to the correct RMS, every shape check
// passes, and the model emits fluent nonsense. MEASURED on
// `turboderp/Llama-3.2-1B-Instruct-exl3` @ 3.0bpw, layer 0 `q_proj`, against the
// unquantized `Llama-3.2-1B-Instruct` tensor fetched by range request:
//
//     cb 1 (mcg, WRONG here):  RMS 0.038454   cosine -0.0006
//     cb 0 (3INST, correct):   RMS 0.035941   cosine +0.9896
//     reference:               RMS 0.036056
//
// Same distribution, opposite verdict. Only a correlation against real
// exllamav3-produced data separates them, which is why the fixtures below gate
// the SELECTION and `test_exl3_dequant` gates the decode.
#include <doctest/doctest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"

namespace {

using vllm::StTensor;
using vllm::dense_loaders::IsExl3Projection;
using vllm::dense_loaders::LoadExl3;

// A hermetic stand-in for a shard: names -> tensors, with bytes this fixture
// owns. No file, no mmap — the reader takes a resolver and a probe, so the
// suite can hand it exactly the tensor set a checkpoint would carry.
struct FakeShard {
  std::map<std::string, StTensor> t;
  std::vector<std::vector<uint8_t>> storage;

  void Add(const std::string& name, const std::string& dtype,
           const std::vector<int64_t>& shape, size_t bytes) {
    storage.push_back(std::vector<uint8_t>(bytes, 0x5A));
    StTensor s;
    s.dtype = dtype;
    s.shape = shape;
    s.data = storage.back().data();
    s.nbytes = bytes;
    t[name] = s;
  }
  // One EXL3 projection at [k, n] and `bits`, without any codebook marker.
  void AddProjection(const std::string& proj, int64_t k, int64_t n, int bits) {
    Add(proj + ".trellis", "I16", {k / 16, n / 16, 16 * bits},
        static_cast<size_t>(k / 16) * (n / 16) * 16 * bits * 2);
    Add(proj + ".suh", "F16", {k}, static_cast<size_t>(k) * 2);
    Add(proj + ".svh", "F16", {n}, static_cast<size_t>(n) * 2);
  }
  vllm::TensorResolver Get() const {
    return [this](const std::string& n) -> const StTensor& {
      auto it = t.find(n);
      VT_CHECK(it != t.end(), "fake shard: tensor not found: " + n);
      return it->second;
    };
  }
  std::function<bool(const std::string&)> Has() const {
    return [this](const std::string& n) { return t.find(n) != t.end(); };
  }
};

}  // namespace

TEST_CASE("exl3 native loader: NO marker means codebook 0, not MCG") {
  FakeShard s;
  s.AddProjection("model.layers.0.mlp.gate_proj", 2048, 8192, 3);
  REQUIRE(IsExl3Projection(s.Has(), "model.layers.0.mlp.gate_proj"));

  const vllm::Exl3Weight w = LoadExl3(s.Get(), s.Has(), "model.layers.0.mlp.gate_proj");
  // THE ASSERTION THIS FILE EXISTS FOR. Reading absence as MCG is the defect
  // that decoded a real checkpoint to fluent nonsense.
  CHECK(w.codebook == 0);
  CHECK(w.InFeatures() == 2048);
  CHECK(w.OutFeatures() == 8192);
  CHECK(w.Bits() == 3);
}

TEST_CASE("exl3 native loader: an mcg marker means codebook 1") {
  FakeShard s;
  s.AddProjection("lm_head", 2048, 128256, 6);
  s.Add("lm_head.mcg", "I32", {1}, 4);

  const vllm::Exl3Weight w = LoadExl3(s.Get(), s.Has(), "lm_head");
  CHECK(w.codebook == 1);
  // The same fixture pins the per-tensor width: this head is SIX-bit, which is
  // what the published 3.0bpw artifact ships over a 3-bit body.
  CHECK(w.Bits() == 6);
}

TEST_CASE("exl3 native loader: an unported codebook REFUSES rather than decoding") {
  FakeShard s;
  s.AddProjection("p", 128, 128, 3);
  s.Add("p.mul1", "I32", {1}, 4);
  // cb 2 is upstream's dp4a byte-sum variant. Decoding it as 0 or 1 would be
  // silently wrong in exactly the way this suite's header documents, so the
  // reader refuses by name instead.
  CHECK_THROWS(LoadExl3(s.Get(), s.Has(), "p"));
}

TEST_CASE("exl3 native loader: the storage predicate is upstream's, all three tensors") {
  FakeShard s;
  s.Add("p.trellis", "I16", {8, 8, 48}, 8 * 8 * 48 * 2);
  // `Linear.is_exl3_storage` requires trellis WITH suh|su AND svh|sv
  // (`modules/linear.py:385-389`). A trellis alone is not EXL3 storage, and
  // answering yes here would route a half-written projection into this reader
  // instead of letting it fall through to the dense loader.
  CHECK_FALSE(IsExl3Projection(s.Has(), "p"));
  s.Add("p.suh", "F16", {128}, 256);
  CHECK_FALSE(IsExl3Projection(s.Has(), "p"));
  s.Add("p.svh", "F16", {128}, 256);
  CHECK(IsExl3Projection(s.Has(), "p"));
}

TEST_CASE("exl3 native loader: a transposed sign vector REFUSES BY NAME") {
  // suh is the INPUT side and svh the OUTPUT side. Swapping them loads, runs
  // and returns a confidently wrong answer on a square projection, so the
  // lengths are checked against the trellis geometry rather than each other.
  FakeShard s;
  s.Add("p.trellis", "I16", {8, 32, 48}, 8 * 32 * 48 * 2);  // k=128, n=512
  s.Add("p.suh", "F16", {512}, 1024);                       // swapped
  s.Add("p.svh", "F16", {128}, 256);                        // swapped
  std::string what;
  try {
    LoadExl3(s.Get(), s.Has(), "p");
    FAIL("exl3 native loader: swapped suh/svh did NOT throw");
  } catch (const std::exception& e) {
    what = e.what();
  }
  INFO("refusal: " << what);
  CHECK(what.find("suh") != std::string::npos);
}
