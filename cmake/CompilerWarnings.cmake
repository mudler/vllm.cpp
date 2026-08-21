function(vllm_cpp_set_warnings target)
  # The instrumentation lanes (VLLM_CPP_SANITIZE) keep the warnings but drop
  # -Werror. This has to happen HERE rather than as a global -Wno-error, because
  # these are PRIVATE target options and therefore land after anything
  # add_compile_options contributed — a global -Wno-error would simply be
  # overridden. Sanitizer instrumentation changes inlining enough that GCC's
  # range and initialization analyses fire inside LIBSTDC++ on correct code (a
  # one-element `std::vector<int32_t> v = {x}` draws "forming offset 4 is out of
  # the bounds [0, 4]" for a 4-byte read of a 4-byte array; <regex> draws 26
  # -Wmaybe-uninitialized reports from std::function internals). None is project
  # code, and the plain build — which IS the one that enforces -Werror — is
  # clean. Letting a false positive in a system header stop the RUNTIME detectors
  # would defeat the point of the lane.
  set(_vllm_cpp_werror -Werror)
  if(NOT VLLM_CPP_SANITIZE STREQUAL "OFF")
    set(_vllm_cpp_werror "")
  endif()

  # GCC >= 15 reports -Warray-bounds inside LIBSTDC++ and the vendored nlohmann
  # json for code that is correct, so the diagnostic stays VISIBLE but stops
  # being fatal on those compilers only. Everything <= 14 is unchanged and still
  # fails the build on a real out-of-bounds.
  #
  # The bound was 16 until gcc 15.2.0 reproduced the same class from the OTHER
  # libstdc++ container: `std::vector<float>`'s inlined copy-construct plus
  # `_M_allocate`, reported against `ltx2_samplers.cpp:161,163` as "array
  # subscript -1 is outside array bounds of 'float [2305843009213693951]'". That
  # bound is SIZE_MAX/4, the giveaway that the allocator's unconstrained size
  # range reached the subscript check rather than a real object. The subscripts
  # are `sigmas.back()` guarded by an explicit `VT_CHECK(sigmas_in.size() >= 2)`
  # three lines above, which GCC does not propagate through the inlined copy. So
  # `main` did not build on gcc 15.x at all, and no CI lane covers that release:
  # `build-newest-gcc` runs 16 (already exempt) and `build-test-cpu` runs older.
  #
  # It is the same false-positive class this file already documents above for
  # the sanitizer lanes, and it is not something the calling code can avoid:
  # `_Sp_counted_base::_M_release()` is identical machine code for every
  # shared_ptr type, so after inlining GCC attributes ONE instantiation's
  # destructor to ANOTHER instantiation's allocation size. On gcc 16.1.1 that
  # presents as "array subscript 'std::mutex[0]' is partly outside array bounds
  # of 'unsigned char [32]'" pointing at a json.dump() call, in a translation
  # unit that contains no shared_ptr<std::mutex> at all.
  #
  # libstdc++ carries its own `#pragma GCC diagnostic` suppressions around that
  # very destructor (bits/shared_ptr_base.h), i.e. the standard library treats
  # this as a warning to silence rather than a bug to fix in user code. Upstream
  # is GCC PR tree-optimization/122197; Eigen, assimp and CMSSW all disable the
  # check the same way on the affected releases.
  set(_vllm_cpp_array_bounds "")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
    set(_vllm_cpp_array_bounds -Wno-error=array-bounds)
  endif()
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra ${_vllm_cpp_werror} ${_vllm_cpp_array_bounds}>
    # OBJCXX (.mm — the Metal backend) is a SEPARATE COMPILE_LANGUAGE from CXX,
    # so the CXX genex above does not reach it. Without this line the Metal TUs
    # would be the only unwarned code in the tree (BACKEND-METAL-MLX W0).
    $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -Wextra -Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
  endif()
endfunction()
