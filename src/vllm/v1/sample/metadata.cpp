// Ported from: vllm/v1/sample/metadata.py @ e24d1b24
//
// SamplingMetadata is a header-only value carrier (see
// include/vllm/v1/sample/metadata.h). This translation unit exists so the header
// is compiled standalone (self-containment check) and has a home in the build.
// It is built by InputBatch::make_sampling_metadata()
// (src/vllm/v1/worker/gpu/input_batch.cpp — the port of
// gpu_input_batch.py::_make_sampling_metadata).
#include "vllm/v1/sample/metadata.h"
