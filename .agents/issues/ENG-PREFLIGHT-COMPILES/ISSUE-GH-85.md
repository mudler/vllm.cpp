ID: ISSUE-GH-85
Title: Error when linking
Row: ENG-PREFLIGHT-COMPILES
State: CLOSED
Kind: bug
GitHub: 85
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-06
Updated: 2026-08-09
Closed: 2026-08-09

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> When building on Debian 13, x86_64
>
> > cmake .
> ...
> > make
> [ 66%] Built target vllm_capi_c_check
> /usr/bin/ld: cannot open output file tokenize: Is a directory
> collect2: error: ld returned 1 exit status
>
> [ 72%] Linking CXX executable test_encoder_cache_manager
> /usr/bin/ld: cannot open output file dequant_nvfp4: Is a directory
> collect2: error: ld returned 1 exit status
> make[2]: *** [examples/CMakeFiles/dequant_nvfp4.dir/build.make:103: examples/dequant_nvfp4] Error 1
> make[1]: *** [CMakeFiles/Makefile2:12751: examples/CMakeFiles/dequant_nvfp4.dir/all] Error 2
>
> [ 76%] Linking CXX executable dump_container
> /usr/bin/ld: cannot open output file dump_container: Is a directory
> collect2: error: ld returned 1 exit status
> make[2]: *** [examples/CMakeFiles/dump_container.dir/build.make:103: examples/dump_container] Error 1
> make[1]: *** [CMakeFiles/Makefile2:12685: examples/CMakeFiles/dump_container.dir/all] Error 2

## Resolution

GitHub's timeline records commit `4cba292de1e8e1fedfc0552bde6410e88eb0e0e6` on 2026-08-09 immediately before closure. That commit refuses the reported in-source `cmake .` build before the linker collision.
