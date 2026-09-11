ID: ISSUE-GH-1531
Title: **Both sanitizer lanes died at LINK on every branch based after `a50c57d69`, so neither reported anything about the code.** `vllm_sanitize_test_shared` forwards `vllm`'s INTERFACE include directories, system include directories and compile definitions, and never forwarded its INTERFACE link libraries. Once W5 put `CPPHTTPLIB_OPENSSL_SUPPORT` on `vllm` PUBLIC, that forwarded define reached every test including the vendored httplib header while `OpenSSL::SSL` did not, and the only libssl near the link line was a `DT_NEEDED` of the shim, which GNU ld refuses to resolve through (`DSO missing from command line`). Eight targets: `test_minimax_music3_e2e_real`, `test_tls_transport`, `test_hf_hub`, `test_model_resolver`, `test_downloader`, `test_serve_hf_model`, `test_openai_conformance`, `test_openai_api_server`. Sanitizer configurations only -- the default build links `vllm::vllm`, whose PUBLIC OpenSSL propagates normally. Fixed in flow by forwarding `INTERFACE_LINK_LIBRARIES` beside the other three forwards
Row: ENG-HF-MODEL-DOWNLOAD
State: UNKNOWN
Kind: bug
GitHub: 1531
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:541`

### Frozen archive evidence

> | [#1531](https://github.com/mudler/vllm.cpp/issues/1531) | `ENG-HF-MODEL-DOWNLOAD` | **Both sanitizer lanes died at LINK on every branch based after `a50c57d69`, so neither reported anything about the code.** `vllm_sanitize_test_shared` forwards `vllm`'s INTERFACE include directories, system include directories and compile definitions, and never forwarded its INTERFACE link libraries. Once W5 put `CPPHTTPLIB_OPENSSL_SUPPORT` on `vllm` PUBLIC, that forwarded define reached every test including the vendored httplib header while `OpenSSL::SSL` did not, and the only libssl near the link line was a `DT_NEEDED` of the shim, which GNU ld refuses to resolve through (`DSO missing from command line`). Eight targets: `test_minimax_music3_e2e_real`, `test_tls_transport`, `test_hf_hub`, `test_model_resolver`, `test_downloader`, `test_serve_hf_model`, `test_openai_conformance`, `test_openai_api_server`. Sanitizer configurations only -- the default build links `vllm::vllm`, whose PUBLIC OpenSSL propagates normally. Fixed in flow by forwarding `INTERFACE_LINK_LIBRARIES` beside the other three forwards | bug |

## Resolution

-
