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

  if(MSVC)
    # CMake's VS generator can still surface TreatWarningAsError=true from
    # higher-level defaults even when we do not pass /WX explicitly. Force the
    # target property off and add /WX- so native Windows builds keep warnings
    # visible without stopping the port on unrelated warning-cleanup work.
    set_property(TARGET ${target} PROPERTY COMPILE_WARNING_AS_ERROR OFF)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CXX>:/WX->
      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
      $<$<COMPILE_LANGUAGE:CXX>:/wd4324>
      $<$<COMPILE_LANGUAGE:CXX>:/wd4458>
      $<$<COMPILE_LANGUAGE:OBJCXX>:/W4>
      $<$<COMPILE_LANGUAGE:OBJCXX>:/WX>
      $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
    # Native Windows/MSVC is not warning-clean yet. Keep /W4 so diagnostics stay
    # visible, but do not promote all C++ warnings to errors or the port never
    # reaches the remaining real build blockers.
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra ${_vllm_cpp_werror}>
      # OBJCXX (.mm — the Metal backend) is a SEPARATE COMPILE_LANGUAGE from CXX,
      # so the CXX genex above does not reach it. Without this line the Metal TUs
      # would be the only unwarned code in the tree (BACKEND-METAL-MLX W0).
      $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -Wextra -Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
  endif()
endfunction()
