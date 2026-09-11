ID: ISSUE-GH-192
Title: vllm_c.cpp:1401:13: error: 'OrEmpty' has C-linkage specified, but returns user-defined type       'std::string' (aka 'basic_string<char>') which is incompatible with C [-Werror,-Wreturn-type-c-linkage]
Row: SERVE-C-ABI
State: CLOSED
Kind: bug
GitHub: 192
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-09
Updated: 2026-08-09
Closed: 2026-08-09

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ```
> -- The CXX compiler identification is Clang 21.1.8
> -- Detecting CXX compiler ABI info
> -- Detecting CXX compiler ABI info - done
> -- Check for working CXX compiler: /usr/lib/llvm/21/bin/clang++ - skipped
> -- Detecting CXX compile features
> -- Detecting CXX compile features - done
> -- Looking for a CUDA compiler
> -- Looking for a CUDA compiler - NOTFOUND
> -- The C compiler identification is Clang 21.1.8
> -- Detecting C compiler ABI info
> -- Detecting C compiler ABI info - done
> -- Check for working C compiler: /usr/lib/llvm/21/bin/clang - skipped
> -- Detecting C compile features
> -- Detecting C compile features - done
> -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
> -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
> -- Found Threads: TRUE
> -- Found Python3: /usr/bin/python3.14 (found version "3.14.4") found components: Interpreter
> -- Configuring done (0.8s)
> -- Generating done (0.4s)
>
> ...
>
> [ 24%] Building CXX object CMakeFiles/vllm.dir/src/vllm/entrypoints/openai/tool_parsers/abstract.cpp.o
> [ 24%] Building CXX object CMakeFiles/vllm.dir/src/vllm/entrypoints/openai/tool_parsers/structural_tags.cpp.o
> /mnt/storage/llm/vllm.cpp/src/capi/vllm_c.cpp:1401:13: error: 'OrEmpty' has C-linkage specified, but returns user-defined type
>       'std::string' (aka 'basic_string<char>') which is incompatible with C [-Werror,-Wreturn-type-c-linkage]
>  1401 | std::string OrEmpty(const char* s) { return s == nullptr ? std::string() : std::string(s); }
>       |             ^
> [ 25%] Building CXX object CMakeFiles/vllm.dir/src/vllm/entrypoints/openai/tool_parsers/detect.cpp.o
> [ 25%] Building CXX object CMakeFiles/vllm.dir/src/vllm/entrypoints/openai/tool_parsers/utils.cpp.o
> [ 25%] Building CXX object CMakeFiles/vllm.dir/src/vllm/entrypoints/openai/tool_parsers/hermes.cpp.o
> [ 25%] Building CXX object CMakeFiles/vllm.dir/src/vllm/entrypoints/openai/tool_parsers/mistral.cpp.o
> [ 25%] Building CXX object CMakeFiles/vllm.dir/src/vllm/entrypoints/openai/tool_parsers/llama.cpp.o
> [ 25%] Building CXX object CMakeFiles/vllm.dir/src/vllm/entrypoints/openai/tool_parsers/llama4_pythonic.cpp.o
> 1 error generated.
> gmake[2]: *** [CMakeFiles/vllm.dir/build.make:3621: CMakeFiles/vllm.dir/src/capi/vllm_c.cpp.o] Error 1
> gmake[2]: *** Waiting for unfinished jobs....
> gmake[1]: *** [CMakeFiles/Makefile2:1314: CMakeFiles/vllm.dir/all] Error 2
> gmake: *** [Makefile:146: all] Error 2
> ```
>
> git rev-parse HEAD
> f921062ba4fbf3341b02d2ac826bb3d84cc91a64
>
>

## Resolution

Commit `7534da6554f0dcea5d27e85590a0bc1c7e9e3c4e` dated 2026-08-09 hoists `OrEmpty` out of `extern "C"` and names the Apple Clang failure. GitHub closed issue #192 on 2026-08-09.
