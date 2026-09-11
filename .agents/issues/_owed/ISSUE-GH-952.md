ID: ISSUE-GH-952
Title: Mirror gap: `--max-num-seqs` does double duty as our HTTP concurrency ceiling. `server_main.cpp:1202-1204` passes it as `max_concurrent_streams` and `api_server.cpp:35-52` sizes a fixed cpp-httplib pool at `max_num_seqs + 4`, while an SSE stream holds its worker for the whole generation; upstream serves through uvicorn over asyncio with NO `limit_concurrency`, no semaphore and no max-concurrent anything (`entrypoints/launcher.py:71,76` at the pin), so `max_num_seqs` bounds only the scheduler batch and the overflow queues INSIDE the engine. REFUTED as the cause of [#931](https://github.com/mudler/vllm.cpp/issues/931): the pool was 36 against an offered concurrency of 8. Filed because raising `--max-num-seqs` for throughput silently raises the HTTP ceiling with it
Row: -
State: UNKNOWN
Kind: bug
GitHub: 952
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:259`

### Frozen archive evidence

> | [#952](https://github.com/mudler/vllm.cpp/issues/952) | — | Mirror gap: `--max-num-seqs` does double duty as our HTTP concurrency ceiling. `server_main.cpp:1202-1204` passes it as `max_concurrent_streams` and `api_server.cpp:35-52` sizes a fixed cpp-httplib pool at `max_num_seqs + 4`, while an SSE stream holds its worker for the whole generation; upstream serves through uvicorn over asyncio with NO `limit_concurrency`, no semaphore and no max-concurrent anything (`entrypoints/launcher.py:71,76` at the pin), so `max_num_seqs` bounds only the scheduler batch and the overflow queues INSIDE the engine. REFUTED as the cause of [#931](https://github.com/mudler/vllm.cpp/issues/931): the pool was 36 against an offered concurrency of 8. Filed because raising `--max-num-seqs` for throughput silently raises the HTTP ceiling with it | bug |

## Resolution

-
