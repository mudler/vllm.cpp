// Syntax-check stub. See ../README.md.
//
// DELIBERATELY EMPTY, and that is exact rather than lazy:
// `metal_mlx_provider.mm` imports <Foundation/Foundation.h> and names no
// Foundation type. Every Metal handle it touches arrives as a `void*`, because
// `metal_buffers.h` and `metal_context.h` are plain C++ on purpose
// (metal_context.h:22). An empty stub therefore models the provider's real use
// of Foundation with no loss.
#ifndef VT_METAL_STUBS_FOUNDATION_FOUNDATION_H_
#define VT_METAL_STUBS_FOUNDATION_FOUNDATION_H_
#endif  // VT_METAL_STUBS_FOUNDATION_FOUNDATION_H_
