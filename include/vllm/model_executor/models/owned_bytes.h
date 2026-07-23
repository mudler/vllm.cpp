// vllm.cpp original. Byte storage for `OwnedTensor` (qwen3_5_weights.h) with
// TWO residencies, mirroring llama.cpp's loader, which supports both:
//
//   * OWNED    — a heap `std::vector<uint8_t>`. Today's behavior, the default,
//                and what every non-GGUF weight path produces.
//   * BORROWED — a read-only `(pointer, size)` view into memory this tensor does
//                NOT own, plus a `shared_ptr` keep-alive on whatever does. Two
//                producers, both from `QUANT-GGUF-KEEPQ-LOADER` L5:
//                  - the GGUF file's read-only mmap, so a kept quant weight is
//                    consumed IN PLACE instead of being copied into an owned
//                    buffer (llama.cpp `src/llama-model-loader.cpp:1385`
//                    `load_data_for`, which mmaps raw block bytes rather than
//                    copying when `use_mmap` is set);
//                  - one bf16 expansion shared by two tensors, which is how a
//                    TIED `token_embd`/`lm_head` stops being materialized twice
//                    (llama.cpp reaches the same end through
//                    `TENSOR_DUPLICATED`, which aliases the head onto the
//                    embedding rather than duplicating its bytes).
//
// WHY A CONTAINER RATHER THAN EXTRA FIELDS ON `OwnedTensor`: the read API below
// is deliberately the exact `std::vector<uint8_t>` subset the tree already uses
// (`data/size/empty/capacity/begin/end/==`), so all ~300 existing `.bytes`
// readers compile and behave unchanged, while every MUTATING entry point
// (`resize`, `assign`, non-const `data`/`begin`/`end`) hard-fails on a borrowed
// buffer instead of silently writing through a read-only mapping. The lifetime
// hazard is therefore a compile-time-shaped, run-time-checked seam rather than a
// raw pointer someone must remember not to outlive: the keep-alive handle is
// carried BY the bytes, so a borrowed `OwnedTensor` cannot outlive its mapping.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm {

class OwnedBytes {
 public:
  OwnedBytes() = default;
  explicit OwnedBytes(std::vector<uint8_t> v) : owned_(std::move(v)) {}

  // Borrow `size` bytes at `data`, holding `owner` for as long as this buffer
  // (or any copy of it) lives. `owner` MUST keep `data` valid; it is the mmap
  // handle or the shared expansion. A null owner is rejected — a borrowed view
  // with no keep-alive is precisely the dangling-pointer bug this type exists
  // to make unrepresentable.
  static OwnedBytes Borrow(const uint8_t* data, size_t size,
                           std::shared_ptr<const void> owner) {
    VT_CHECK(data != nullptr || size == 0, "OwnedBytes::Borrow: null data");
    VT_CHECK(owner != nullptr, "OwnedBytes::Borrow: a borrowed buffer requires a keep-alive owner");
    OwnedBytes b;
    b.borrowed_ = data;
    b.borrowed_size_ = size;
    b.owner_ = std::move(owner);
    return b;
  }

  bool borrowed() const { return owner_ != nullptr; }

  // --- read API (identical to std::vector<uint8_t>'s, both residencies) ------
  const uint8_t* data() const { return borrowed() ? borrowed_ : owned_.data(); }
  size_t size() const { return borrowed() ? borrowed_size_ : owned_.size(); }
  bool empty() const { return size() == 0; }
  size_t capacity() const { return borrowed() ? borrowed_size_ : owned_.capacity(); }
  const uint8_t* begin() const { return data(); }
  const uint8_t* end() const { return data() + size(); }

  // Non-const element access. It does NOT reject a borrowed buffer, because
  // most of this tree reads a weight through a non-const handle and constness
  // therefore does not distinguish a read from a write — rejecting here would
  // fire on hundreds of legitimate readers. What it means is: *writing* through
  // this pointer is the caller's contract to keep. For the mmap residency the
  // contract is enforced by the kernel (the mapping is PROT_READ, so a write
  // faults immediately and loudly); for a shared expansion it rests on the fact
  // that the ONLY sharing producer is the tied `token_embd`/`lm_head` pair,
  // which nothing writes to after the loader builds it.
  //
  // The mutations that would actually INVALIDATE a borrow — reallocating or
  // resizing the buffer out from under the owner — are rejected below, and they
  // are the ones no contract can cover.
  uint8_t* data() { return const_cast<uint8_t*>(std::as_const(*this).data()); }
  uint8_t* begin() { return data(); }
  uint8_t* end() { return data() + size(); }

  // --- structural mutation (OWNED only; loud on a borrowed buffer) -----------
  void resize(size_t n) { RequireOwned("resize()"); owned_.resize(n); }
  void resize(size_t n, uint8_t fill) { RequireOwned("resize()"); owned_.resize(n, fill); }
  template <typename It>
  void assign(It first, It last) { RequireOwned("assign()"); owned_.assign(first, last); }
  void assign(size_t n, uint8_t fill) { RequireOwned("assign()"); owned_.assign(n, fill); }

  // Turn an OWNED buffer into a shared read-only one, WITHOUT copying (the
  // vector is moved into a refcounted holder), and return that holder so a
  // SECOND OwnedBytes can `Borrow` the very same bytes. This is how one bf16
  // expansion comes to serve two tensors — a tied `token_embd`/`lm_head` — in
  // place of materializing the vocab matrix twice.
  std::shared_ptr<const void> Share() {
    RequireOwned("Share()");
    auto holder = std::make_shared<const std::vector<uint8_t>>(std::move(owned_));
    std::vector<uint8_t>().swap(owned_);  // moved-from vector: force it to release
    borrowed_ = holder->data();
    borrowed_size_ = holder->size();
    owner_ = holder;
    return owner_;
  }

  // Drop the payload: frees an owned vector's capacity outright, and releases a
  // borrowed view's keep-alive (which may drop the last reference to the mmap or
  // the shared expansion). Both leave an empty, OWNED buffer.
  void Reset() {
    std::vector<uint8_t>().swap(owned_);
    borrowed_ = nullptr;
    borrowed_size_ = 0;
    owner_.reset();
  }

  // VALUE equality — residency is not part of the value, so a borrowed buffer
  // compares equal to the owned copy of the same bytes. That is what makes the
  // "nothing moved" tests meaningful across the residency change.
  friend bool operator==(const OwnedBytes& a, const OwnedBytes& b) {
    return a.size() == b.size() &&
           (a.size() == 0 || std::memcmp(a.data(), b.data(), a.size()) == 0);
  }
  friend bool operator!=(const OwnedBytes& a, const OwnedBytes& b) { return !(a == b); }

 private:
  void RequireOwned(const char* what) const {
    VT_CHECK(!borrowed(), std::string("OwnedBytes::") + what +
                              " on a BORROWED buffer: it views read-only memory this "
                              "tensor does not own (a GGUF mmap or a shared expansion)");
  }

  std::vector<uint8_t> owned_;
  const uint8_t* borrowed_ = nullptr;
  size_t borrowed_size_ = 0;
  // Non-null IFF borrowed. Type-erased: the producer decides what it keeps alive
  // (a GgufFile mapping, a shared bf16 expansion).
  std::shared_ptr<const void> owner_;
};

}  // namespace vllm
