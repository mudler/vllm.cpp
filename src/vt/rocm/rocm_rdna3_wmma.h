/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
// Adapted from rocWMMA 2.2.1 rocwmma_impl.hpp::mma_sync (lines 250-279).
// Triton at the pinned vLLM e126687a9a duplicates gfx11's packed inputs across
// the two half-waves. rocWMMA's upper-half K rotation changes FP32 rounding.
// Keep the library's transforms and accumulator layout; canonicalize only
// the packed operands on gfx1100. gfx12 retains the public rocWMMA operation.
#pragma once
#include <rocwmma/rocwmma.hpp>

namespace vt::rocm {
template <class A, class B, class C>
__device__ inline void AttentionMmaSync(C& output, const A& a, const B& b, C& input) {
#if defined(__gfx1100__)
  using Config = rocwmma::GetMmaConfig_t<A, B, C, C>;
  using XA = typename Config::PreMmaXFormA;
  using XB = typename Config::PreMmaXFormB;
  using XC = typename Config::PreMmaXFormC;
  using XD = typename Config::PostMmaXFormD;
  using PackA = typename Config::PackA;
  using PackB = typename Config::PackB;
  using PackC = typename Config::PackC;
  using PackD = typename Config::PackD;
  using Mma = typename Config::Mma;
  auto pa = PackA::pack(XA::exec(a.mAccess));
  auto pb = PackB::pack(XB::exec(b.mAccess));
  auto pc = PackC::pack(XC::exec(input.mAccess));
  static_assert(sizeof(typename rocwmma::VecTraits<decltype(pa)>::DataT) == 4);
  static_assert(sizeof(typename rocwmma::VecTraits<decltype(pb)>::DataT) == 4);
  const int source = static_cast<int>(threadIdx.x) & 15;
#pragma unroll
  for (unsigned i = 0; i < rocwmma::VecTraits<decltype(pa)>::size(); ++i)
    rocwmma::to_native_vector(pa)[i] = __shfl(rocwmma::to_native_vector(pa)[i], source, 32);
#pragma unroll
  for (unsigned i = 0; i < rocwmma::VecTraits<decltype(pb)>::size(); ++i)
    rocwmma::to_native_vector(pb)[i] = __shfl(rocwmma::to_native_vector(pb)[i], source, 32);
  output.mAccess = XD::exec(PackD::unpack(Mma::exec(pa, pb, pc)));
#else
  rocwmma::mma_sync(output, a, b, input);
#endif
}
}  // namespace vt::rocm
