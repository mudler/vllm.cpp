ID: ISSUE-GH-396
Title: `test_lmcache_connector` data race under TSan: `MockLmcacheServer` writes non-atomic `listen_fd_` before joining its accept thread
Row: KV-EXTERNAL-CACHE
State: UNKNOWN
Kind: bug
GitHub: 396
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:98`

### Frozen archive evidence

> | [#396](https://github.com/mudler/vllm.cpp/issues/396) | `KV-EXTERNAL-CACHE` | `test_lmcache_connector` data race under TSan: `MockLmcacheServer` writes non-atomic `listen_fd_` before joining its accept thread | bug |

## Resolution

-
