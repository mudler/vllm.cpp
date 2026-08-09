// vllm.cpp original. The OpenAI server's ENTRY POINT, owned by the library.
//
// ARCH-ONE-SURFACE: `examples/server` used to construct the engine, the serving
// layers, metrics, the video seam and the ASR seam directly from 36 internal
// headers -- the deepest breach of the ONE SURFACE directive
// (.agents/specs/one-surface-abi.md), and the reason the example carried an
// entry in scripts/example-abi-allowlist.txt. The construction moved HERE
// verbatim; the example is now a thin client of the C ABI's `vllm_server_main`,
// which wraps this.
//
// argv rather than a params STRUCT is deliberate. The server takes ~57 flags and
// grows more with every serving feature; a mirrored C struct would put that
// churn in the ABI, where every field is permanent. The flag surface is already
// specified by vLLM's cli_args.py, which is the contract this mirrors, so argv
// IS the stable interface. Embedders that want programmatic control keep the
// granular entry points (vllm_engine_load / vllm_chat / vllm_video_generate /
// vllm_transcribe) -- this one exists to RUN THE SERVER.
#ifndef VLLM_ENTRYPOINTS_OPENAI_SERVER_MAIN_H_
#define VLLM_ENTRYPOINTS_OPENAI_SERVER_MAIN_H_

namespace vllm {
namespace entrypoints {
namespace openai {

// Parse `argv` and run the OpenAI-compatible server until it exits. Returns the
// process exit code (0 on clean shutdown). Prints usage and returns 0 for
// `--help`, and returns non-zero after printing the reason on a bad argument or
// a startup failure -- it does NOT throw across the C ABI boundary.
int VllmServerMain(int argc, char** argv);

}  // namespace openai
}  // namespace entrypoints
}  // namespace vllm

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVER_MAIN_H_
