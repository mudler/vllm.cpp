// vllm.cpp original. `vllm-server` — the OpenAI-compatible server, as a THIN
// CLIENT of the public C ABI.
//
// ARCH-ONE-SURFACE (.agents/specs/one-surface-abi.md, developer-directed
// 2026-08-07): every unit under examples/ is a thin client of the public surface
// — the flat C ABI `vllm.h`, the ONLY header `make install` ships — never of the
// internal C++ tree. This file used to be the DEEPEST breach of that rule: 1046
// lines reaching into 36 internal headers to construct the engine, the serving
// layers, metrics, the video seam and the ASR seam by hand, which is why it
// carried an entry in scripts/example-abi-allowlist.txt.
//
// The construction moved into the library verbatim
// (src/vllm/entrypoints/openai/server_main.cpp, declared in
// vllm/entrypoints/openai/server_main.h) and is published as `vllm_server_main`
// at ABI v17. HTTP and FFI therefore cannot drift: the server the ABI runs IS
// the server this binary runs.
//
// Flag parsing lives in the library with the construction it feeds, because the
// two are one contract (vLLM's cli_args.py). Mirroring ~57 flags into a C struct
// would have put that churn in the ABI, where every field is permanent.
#include "vllm.h"

int main(int argc, char** argv) { return vllm_server_main(argc, argv); }
