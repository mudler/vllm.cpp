# Spike: mirror uvicorn/asyncio TCP_NODELAY on the SSE serving transport

Row: `SERVE-HTTP-TRANSPORT` (engine matrix, serving section, next to
`SERVE-ASYNC-LLM`). Owner claim: `CLAIM-SERVE-TRANSPORT-1`. This is the durable
spike gate for lever #1 of the 2026-07-14
[parity rescan](parity-rescan-2026-07-14.md): the OpenAI HTTP server serves
per-token SSE frames with Nagle enabled, while vLLM's effective serving
transport (uvicorn over asyncio) disables Nagle on every accepted socket.

## Scope

Mirror vLLM's serving-socket transport behavior for the streaming OpenAI
endpoints. Concretely: enable `TCP_NODELAY` on every accepted connection socket
of our `httplib::Server`, so each per-token `text/event-stream` frame is put on
the wire immediately instead of being coalesced/held by Nagle's algorithm
against the peer's delayed ACK.

In scope for this checkpoint (CODE + TESTS + RECORD):

- One production one-liner: `server.set_tcp_nodelay(true)` in the `ApiServer`
  setup, before `listen`, with a comment citing the mirror grounding.
- A behavioral, socket-level regression test that observes the accepted
  server socket's `TCP_NODELAY` option.
- The full record surface set (README, BENCHMARKS, roadmap, engine matrix,
  coordination claim, state log, parity ledger).

Explicitly OUT of scope (future `SERVE-HTTP-TRANSPORT` work, noted not done):
keep-alive idle timeout parity, read/write timeout parity, `SO_REUSEADDR`/
backlog parity, and HTTP/2 — tracked as later transport-parity items. Also out
of scope and prohibited here: anything GPU, and the c16 diagnostic track
(`scripts/dgx-gdn-packed-component.sh`, `tools/bench/gdn_packed_component.py`).

## Upstream chain

vLLM's effective serving transport is **uvicorn over CPython asyncio**, and the
Nagle-disable is applied by asyncio, not by uvicorn or vLLM directly:

- vLLM serves through uvicorn:
  `vllm/entrypoints/launcher.py:71,76` builds `uvicorn.Config(app, ...)` +
  `uvicorn.Server(config)` inside `serve_http`; `vllm/entrypoints/openai/
  api_server.py:591,630` passes the `uvicorn_kwargs`. There is no raw socket
  server in vLLM's OpenAI entrypoint.
- uvicorn accepts connections via asyncio's `create_server`:
  `uvicorn/server.py:144,151` `await loop.create_server(create_protocol,
  sock=sock, ...)` (verified against the locally installed uvicorn `0.49.0`;
  the pinned-env uvicorn version is not installed locally, but the
  `create_server` → asyncio-transport path is invariant across uvicorn
  versions). uvicorn's own `config.py:571` only sets `SO_REUSEADDR` on the
  **listening** socket — it does not touch the accepted socket's `TCP_NODELAY`.
- CPython asyncio enables `TCP_NODELAY` on **every accepted TCP stream
  socket**: `asyncio/base_events.py:192-197` defines `_set_nodelay(sock)` →
  `sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)` for
  AF_INET/AF_INET6 SOCK_STREAM sockets, and `asyncio/selector_events.py:950`
  calls it from `_SelectorSocketTransport.__init__`, i.e. once per accepted
  connection (verified in local CPython `3.12.3`; behavior is present since
  Python 3.6 and is version-invariant for TCP sockets).

Net upstream contract we mirror: vLLM's SSE stream is delivered over sockets
with Nagle disabled. Grounding for the parity relevance (per-token SSE writes,
the c2–c8-fail / c16–c32-pass shape, the fat c8/c32 ITL tails, and the
GPU-faster/wall-slower contradiction) is the rescan report, lever #1.

## Our baseline

Our OpenAI server is a bare `httplib::Server` (cpp-httplib, vendored), which
defaults Nagle **on** for accepted sockets:

- `third_party/httplib/httplib.h:142` — `#define CPPHTTPLIB_TCP_NODELAY false`
  is the default; `httplib.h:2002` initializes `Server::tcp_nodelay_` from it.
- `third_party/httplib/httplib.h:12083` — in the accept loop, the server sets
  `set_socket_opt(sock, IPPROTO_TCP, TCP_NODELAY, 1)` on the accepted socket
  **only if** `tcp_nodelay_` is true; otherwise the accepted socket keeps the
  kernel default (Nagle enabled).
- `third_party/httplib/httplib.h:11340-11342` — `Server::set_tcp_nodelay(bool)`
  sets `tcp_nodelay_`; it is read at accept time, so calling it any time before
  `listen`/`serve` takes effect for all subsequent connections.
- No `set_tcp_nodelay(true)` call exists anywhere under `src/`, `include/`, or
  `examples/` today (grep-verified in the rescan); the SSE path in
  `src/vllm/entrypoints/openai/api_server.cpp` (chunked content provider,
  `register_routes`) therefore writes per-token frames under Nagle.

## Port map

Production change (one line + comment) in the `ApiServer` setup, before the
accept loop starts:

- `src/vllm/entrypoints/openai/api_server.cpp` — in `ApiServer::Impl`
  construction (the existing server setup that also installs the fixed
  `new_task_queue`), call `server.set_tcp_nodelay(true);` with a comment citing
  the uvicorn/asyncio mirror grounding. Nothing else about the server changes.

There is no header change: `set_tcp_nodelay` is an existing cpp-httplib
`Server` method and the `httplib::Server` is already the pimpl member. There is
no vLLM `.py` to translate literally (the behavior lives in asyncio), so this is
a behavior mirror, recorded as such, not a line-for-line source port.

## Tests to port

vLLM has no unit test that asserts `TCP_NODELAY` (it is inherited from
asyncio), so there is no upstream test module to re-express. We add a
behavioral socket-level regression instead, in the existing real-server socket
layer of `tests/vllm/entrypoints/openai/test_api_server.cpp` (which already
stands up the `ApiServer` on a loopback ephemeral port via `bind_to_any_port`
+ a background `serve()` thread):

- New case: after connecting a client to the running `ApiServer` and driving one
  request, assert (server-side) that the server's **accepted** socket has
  `TCP_NODELAY == 1`. Because the test client and the server run in the SAME
  process, the accepted socket fd is discoverable in-process: scan
  `/proc/self/fd`, match the socket whose `getsockname()` local port is the
  server port and whose `getpeername()` peer port is the client's local
  ephemeral port (this uniquely identifies the accepted socket, distinct from
  the listening socket — which fails `getpeername` — and from the client socket
  — whose local port differs), then `getsockopt(fd, IPPROTO_TCP, TCP_NODELAY)`.
  This is a true behavioral assertion on the production `ApiServer`, requires no
  change to the vendored httplib and no test-only production hook, and is
  deterministic on loopback. It is Linux-guarded (`/proc/self/fd`); the CPU test
  tier runs on Linux.

Chosen approach = **(a) behavioral socket-level assertion**, not (b) the
source-text contract pin, because the accepted-socket fd is reachable
in-process on Linux and a behavioral assertion is strictly stronger: it fails
RED before the production call is added (accepted socket `TCP_NODELAY == 0`) and
passes GREEN after (`== 1`), proving the observable transport behavior rather
than the presence of a source line.

## Gates

- **Correctness / behavioral (binding here):** the new socket-level case is
  RED before the production one-liner (accepted `TCP_NODELAY == 0`) and GREEN
  after (`== 1`); the full `test_openai_api_server` target stays green; clean
  rebuild under `-Werror` is clean. `scripts/check-agent-record.py`,
  `tests/scripts/test_agent_record.py`, `tests/scripts/test_doc_checkpoint.py`
  pass, and `scripts/check-doc-checkpoint.py` passes over the commit.
- **Performance (non-binding, PENDING):** exactly one non-binding localhost SSE
  A/B stream sizing on dgx to observe the loopback magnitude — explicitly
  DEFERRED because the GPU is held by the c16 diagnostic reproduction; do NOT
  run anything on dgx for this row now. This measurement is illustrative only.
- **Axis credit rule:** this row earns parity-axis credit ONLY through the
  single authorized exact-grid rerun under `CLAIM-SERVE-GATE-1`; no localhost
  or component number binds any of the 124 axes.

## Dependencies

- Depends on the vendored cpp-httplib (`VLLM_CPP_SERVER=ON`) and its
  `Server::set_tcp_nodelay`. No new third-party dependency.
- Independent of every GPU/kernel component and of the c16 diagnostic track;
  can land in parallel with them (rescan "Recommended sequence" step 2).
- The eventual performance sizing depends on an idle GPU window, which is
  currently unavailable (c16 holds it) — hence PENDING.

## Work breakdown

Single row-sized unit (this checkpoint), executed test-first:

1. Spike (this document).
2. RED behavioral socket-level test in `test_api_server.cpp`; run, observe the
   right-reason failure (accepted `TCP_NODELAY == 0`).
3. Production one-liner `server.set_tcp_nodelay(true)` in `api_server.cpp`.
4. GREEN: focused target + clean `-Werror` rebuild.
5. Record surfaces + record-checker gates, one commit.

Later `SERVE-HTTP-TRANSPORT` transport-parity items (separate future rows/claims,
not this change): keep-alive/idle-timeout parity, read/write-timeout parity,
listening-socket option parity, HTTP/2; and the authorized exact-grid sizing.

## Risks and decisions

- **Decision — behavioral over contract-pin:** the accepted-socket fd is
  reachable in-process on Linux, so we assert observable `TCP_NODELAY`, which is
  strictly stronger than pinning the source text.
- **Risk — fd disambiguation:** matching only on the client's ephemeral peer
  port + the server local port uniquely selects the accepted socket; the test
  stops the server per case so no stale accepted fd from a sibling case leaks.
  Mitigation: match on BOTH ports and keep the client connection open (HTTP
  keep-alive) while scanning.
- **Risk — Linux-only observation:** `/proc/self/fd` is Linux; the assertion is
  `#if defined(__linux__)`-guarded. The CPU test tier is Linux, so RED/GREEN is
  observed there; other platforms compile the case without the behavioral
  assertion.
- **Decision — coordination state:** the record checker requires a coordination
  claim to reference a `SPIKE`/`ACTIVE` row, so while `CLAIM-SERVE-TRANSPORT-1`
  is open the `SERVE-HTTP-TRANSPORT` row is `ACTIVE` (implemented + CPU-tested;
  the non-binding GPU localhost sizing is the remaining open work under the
  claim, parked because the GPU is held by c16). It converges to `GATING` when
  the claim releases after the sizing (or an explicit deferral).
- **Non-risk — correctness:** `TCP_NODELAY` changes only when bytes leave the
  socket buffer, never their content; SSE framing and all existing
  `test_api_server` behavior are unaffected.
</content>
</invoke>
