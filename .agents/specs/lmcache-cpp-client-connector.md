# Spike: a from-scratch C++ LMCache CLIENT (connect over the wire, no `lmcache` PyPI package)

**Rows touched (analysis only, no state past `SPIKE`):** `KV-EXTERNAL-CACHE`
(primary), `KV-CONNECTORS` (the seam it reuses). **Claim:**
`CLAIM-LMCACHE-CPP-CLIENT`. **ANALYSIS / SCOPING ONLY — no implementation, no
build, no GPU.**

Portfolio row: `ROAD-V1-D4` (external KV-cache provider interoperability).
This spec **reopens** the LMCache disposition that
[kv-persistence-lmcache.md](kv-persistence-lmcache.md) deferred as
"external PyPI, go/no-go, no specified wire protocol to implement against"
(its §Risks R2 and W6). It reopens it on the user's explicit hypothesis
(2026-07-23): *"we DO want to connect to LMCache, so we should NOT need a pypi
package — IIRC it works with zmq."* The question this spec answers with source:
**does vLLM talk to a RUNNING LMCache instance over a network/IPC wire protocol
a from-scratch C++ client could speak, without importing `lmcache` in our
process?**

**Headline verdict — the user is RIGHT, and the prior spike's "no specified wire
protocol" claim is REFUTED by reading the LMCache source.** There are TWO
distinct, fully-specified, language-agnostic wire protocols vLLM/LMCache use to
talk to a *running* LMCache server, and a third in-process mode that is the only
genuinely non-portable one:

1. **Remote-store protocol** (`lm://` → `lmcache.v1.server.LMCacheServer`):
   **plain TCP**, a **fixed `struct.pack` binary header + raw KV bytes**. No
   ZMQ, no msgpack, no pickle, no CUDA-IPC. **This is the cleanest possible C++
   client target and it is unambiguously FEASIBLE.**
2. **Multiprocess (MP) protocol** (`tcp://…:5555` → the LMCache MP server, the
   mode the user remembers as "zmq"): **ZMQ DEALER↔ROUTER**, control plane
   **`msgspec.msgpack`** (portable), data plane **CUDA-IPC** (portable from C++
   via `cudaIpcGetMemHandle`/`OpenMemHandle`, but co-located and layout-coupled),
   plus **one pickle blob** in the one-time cache registration. FEASIBLE with
   effort; matches the user's ZMQ recollection.
3. **In-process** (`LMCacheConnectorV1`, the default): the `LMCacheEngine` runs
   *inside* the process (`lmcache_connector.py:107-113`). This is Python-only and
   is **not a wire boundary** — correctly rejected, as before.

Both (1) and (2) **need zero `lmcache` in our process** and both **sidestep the
prior spike's R1 hash-compatibility blocker**, because LMCache keys on its OWN
token hashing (blake3 rolling, chunk_size 256), never on vLLM's `sha256`/`cbor`
block hash. The user's belief is confirmed with source.

### Scope

The complete path by which vLLM connects to a running LMCache instance, read on
BOTH sides of the boundary and cited `file:line`: the vLLM-vendored connector
glue (pin `e24d1b24`, `/home/mudler/_git/vllm`) AND the external LMCache package
itself (`LMCache/LMCache` @ `8570aad`, cloned to `/home/mudler/_git/lmcache-src`
for this analysis — not installed on any project box, consistent with the prior
finding). It maps, for each of the three modes, the transport (ZMQ socket type /
TCP / CUDA-IPC), the serialization on the wire (msgpack / fixed `struct` / raw
bytes / pickle), the message types, the key format and whether it forces us to
match a vLLM hash, and the KV block byte layout the server expects. It then gives
the **feasibility verdict**, the **showstopper analysis** (pickle? CUDA-IPC?
torch serialization?), **how much of the landed W4/W5 `KVConnector` seam an
LMCache client reuses**, and a row-sized W-plan for the recommended target.

Out of scope (dispositioned, not scheduled): the CB/blend RPCs
(`protocols/blend*.py`), P2P (`protocols/p2p.py`), the HTTP control frontend
(`multiprocess/http_server.py`), the observability/Prometheus RPCs, and every
non-`lm`/non-MP storage backend (Redis/S3/Mooncake/Infinistore/GDS) — each is a
separate store protocol with its own external dependency, enumerated in
§Risks D2 but not planned. No implementation is in scope: this is a spec.

### Upstream chain

Two repositories, both read at a fixed commit:

| Repo | Pin | Verify |
|---|---|---|
| vLLM (connector glue) | `e24d1b24` | `git -C /home/mudler/_git/vllm log --oneline -1` |
| LMCache (the package) | `8570aad` | `git -C /home/mudler/_git/lmcache-src log --oneline -1` → `8570aad chore(mp_coordinator)…` |

**Correction to the record.** The prior spike concluded (R2, W6) that "there is
no specified wire protocol to implement against … a C++ LMCache client means
reimplementing an unpinned Python package's ZMQ message types and CUDA-IPC
handshake … the opposite of the mechanical-port property." Reading the LMCache
source refutes the strong form of that claim: the wire IS specified, in two
places, and one of them (the remote-store protocol) is a ~90-line fixed-`struct`
byte protocol with nothing Python-specific on the wire. The prior spike never
read the LMCache package (it was not installed and was not cloned); this spec
does, and the disposition changes from "reverse-engineering a moving target" to
"a specified, portable protocol with a versioning risk."

#### The full connection architecture (vLLM → connector → LMCache → wire)

```
                                vLLM process                     │  LMCache server process
                                                                 │  (a separate OS process/container)
  ┌─────────────────────────────────────────────────────────┐   │
  │ Scheduler ── KVConnectorBase_V1 (7 abstract methods) ────┼── │
  │ Worker    ──         (base.py:171,454,489,510,…)         │   │
  └───────────────┬─────────────────────────────────────────┘   │
                  │ pick ONE connector implementation:           │
                  │                                              │
   (default) LMCacheConnectorV1 ──► LMCacheEngine IN-PROCESS ───────► speaks a STORAGE BACKEND
     lmcache_connector.py:72,107-113   (Python, needs lmcache)  │      (Redis / S3 / lm:// / …)
                  │                                              │            │
                  │                                              │            ▼  MODE (1) if backend = lm://
   (MP)  LMCacheMPConnector ──► LMCacheMP{Scheduler,Worker}Adapter ──ZMQ──►  MODE (2) MP server
     lmcache_mp_connector.py:10,464    multi_process_adapter.py:141,348  │   multiprocess/server.py
                  │   ZMQ DEALER, msgspec.msgpack control        │            (ROUTER)
                  │   CUDA-IPC data (register once), event IPC    │            reads our GPU KV directly
                  ▼                                              │
        ═══════════ THE WIRE ═══════════                        │
   MODE (1) lm://  : plain TCP  + fixed struct header + raw KV bytes  → lmcache.v1.server.LMCacheServer
   MODE (2) MP     : ZMQ DEALER↔ROUTER + msgspec.msgpack + CUDA-IPC   → lmcache.v1.multiprocess.server
```

**vLLM side (pin `e24d1b24`).** All 16 connectors are `KVConnectorBase_V1`
subclasses selected by name in `factory.py:152-238`.

- `LMCacheConnectorV1` (`lmcache_connector.py:72`): the default. `__init__`
  (`:83-113`) reads `use_native`; either way it instantiates a Python
  `LMCacheConnectorV1Impl` — vendored (`:101-103`) or from the installed package
  (`:107-111`) — and delegates **every** hook to it (`:113`, then `:120-354`).
  This engine runs the cache logic in-process. **Not a wire boundary.**
- `LMCacheMPConnector` (`lmcache_mp_connector.py:1226`, class body
  `:464-1192`): the "recommended standalone-server mode." `import zmq` at
  `:10`. `__init__` (`:476-524`) reads `lmcache.mp.host` (default
  `tcp://localhost`), `lmcache.mp.port` (default `5555`), builds a
  `zmq.Context.instance()` and a scheduler- or worker-side adapter
  (`:504-520`). Delegates the 7 hooks to the adapters (`:559-903`). At module
  load it PREFERS the external package's connector and falls back to this
  builtin (`:1200-1226`) — so even the "builtin" path imports
  `lmcache.v1.multiprocess.*` (`:11-13,26-49`).
- The MP adapters live in the vendored
  `lmcache_integration/multi_process_adapter.py`: `import zmq` (`:10`), imports
  `lmcache.v1.multiprocess.{custom_types,mq,protocol}` (`:12-18`).
  `MessageQueueClient(server_url, context)` is the ZMQ client (`:161,357`);
  `send_lmcache_request` (`:42-62`) calls `mq_client.submit_request(request_type,
  payloads, response_class)`. `register_kv_caches` wraps our GPU tensors as
  CUDA-IPC handles (`wrap_kv_caches` `:23-25` → `CudaIPCWrapper`) and sends
  `RequestType.REGISTER_KV_CACHE` ONCE (`:413-429`). Per-op `submit_store`/
  `submit_retrieve` (`:431-497`) send keys + `instance_id` + `block_ids` +
  `event.ipc_handle()` (a CUDA event IPC handle) under `RequestType.STORE`/
  `RETRIEVE`. Lookup (`:191-259`) sends `RequestType.LOOKUP` with keys.

**LMCache side (pin `8570aad`).** The two servers a running instance exposes:

MODE (2) — the MP server (`lmcache/v1/multiprocess/`):
- Transport: `MessageQueueClient` opens a `zmq.DEALER` and `connect`s
  (`mq.py:270-271`); the shared poll loop notes "all clients' **DEALER**
  sockets" (`mq.py:128`), so the server is the ROUTER peer. Requests are
  `send_multipart([b_request_uid, b_request_type] + b_payloads)`
  (`mq.py:323`); responses are `recv_multipart()` framed
  `[uid, type, *response]` (`mq.py:334-353`).
- Serialization: **`msgspec.msgpack`** end to end — `msgspec_encode`/
  `msgspec_decode` (`mq.py:82-105`), which encode each dataclass/`msgspec.Struct`
  payload as a msgpack map keyed by field name (`custom_types.py:53-56`
  documents the forward-compat-by-field-name wire contract). `RequestType`
  enum → msgpack int; `RequestUID` → msgpack (`mq.py:46-50`).
- Message types: `protocols/base.py:27-90` (the `RequestType` enum — engine ops
  `REGISTER_KV_CACHE`, `UNREGISTER_KV_CACHE`, `STORE`, `RETRIEVE`, `LOOKUP`,
  `FREE_LOOKUP_LOCKS`, `END_SESSION`, `PREPARE/COMMIT_{STORE,RETRIEVE}`;
  controller `GET_CHUNK_SIZE`, `CLEAR`, `PING`; plus blend/P2P). Each type's
  ordered payload classes + response class are declared in
  `protocols/engine.py:87-…` (e.g. `REGISTER_KV_CACHE` payload =
  `[int, KVCache, str, int, EngineType, LayoutHints, list[EngineGroupInfo]]`,
  `protocols/engine.py:96-115`). `protocol.py:32-67` resolves them.
- Key format: `IPCCacheServerKey` (`custom_types.py:26-116`) — a `@dataclass`
  msgpack map: `model_name, world_size, worker_id|None, token_ids: tuple[int],
  start, end, request_id, cache_salt`. **Two key modes:** *token mode* sends the
  raw `token_ids` and the SERVER computes chunk hashes via its `TokenHasher`
  (`token_hasher.py:54-79`, blake3, chunk_size 256, rolling prefix hash
  `blake3(prefix_hash ‖ tokens)`); *hash mode* strides vLLM block hashes
  client-side (`multi_process_adapter.py:28-39`, takes the last block-hash of
  each chunk). **Token mode needs no vLLM-hash match at all.**
- KV data plane: **CUDA-IPC.** `KVCache = list[DeviceIPCWrapper]`
  (`custom_types.py:120`). The wrapper crosses the msgpack wire as an Ext blob,
  **code 1** (`custom_types.py:212-234`), whose payload is **`pickle.dumps(obj)`**
  (`platform/base/ipc_wrapper.py` `Serialize`/`Deserialize`). The default
  `CudaIPCWrapper` (`platform/cuda/ipc_wrapper.py:30-89`) publishes a PyTorch
  caching-allocator handle via `storage._share_cuda_()`; `RawCudaIPCWrapper`
  (`:92-187`) is the **non-PyTorch** path — plain `cudaIpcGetMemHandle` on the
  data pointer (`:132`) and `cudaIpcOpenMemHandle` on the receiver (`:170`).
  Per-op transfers carry a CUDA **event** IPC handle (`event.ipc_handle()`,
  `multi_process_adapter.py:461,495`), and the server, holding our registered
  memory handle, does the copy itself on its own stream after waiting the event.
- The server touches our GPU KV **directly**; it is therefore **co-located**
  (same node, same visible GPU). This is a KV-*movement* offload where LMCache
  owns the DMA, not a network cache.

MODE (1) — the remote-store server (`lmcache/v1/server/`, backend `lm://`):
- Transport: **plain TCP** `socket.SOCK_STREAM`. Server:
  `server/__main__.py:24-32` (`bind`/`listen`), `handle_client` loop `:43-135`.
  Client: `storage_backend/connector/lm_connector.py:28-177`
  (`LMCServerConnector`, `socket.connect` `:47-48`), wired by `lm://` in
  `lm_adapter.py:17-28`.
- Serialization: a **fixed `struct.pack` binary header, then raw KV bytes** —
  `protocol.py`. `ClientMetaMessage.serialize` (`:228-251`) =
  `struct.pack("iiiiiiiii150s", command, length, fmt, dtype_int, location_int,
  shape0..3, key.ljust(150))` → `packlength()` = `4*9 + 150 = 186` bytes.
  `ServerMetaMessage` (`:288-321`) = `struct.pack("iiiiiiiii", code, length,
  fmt, dtype_int, shape0..3, location_int)` → 36 bytes. **No pickle, no msgpack,
  no torch on the wire.**
- Message types: `ClientCommand` (`:28-33`) = `PUT, GET, EXIST, LIST, HEALTH`.
  PUT = header then `sock.sendall(kv_bytes)` (`lm_connector.py:123-136`,
  server `__main__.py:55`). GET = header → server replies `ServerMetaMessage`
  then raw bytes, client `recv_into` a preallocated buffer
  (`lm_connector.py:57-81,140-166`). EXIST = header → `SUCCESS`/`FAIL`
  (`:83-100`). `ServerReturnCode` = `SUCCESS=200`/`FAIL=400` (`:36-39`).
- Key format: `CacheEngineKey.to_string()` (`utils.py:449-453`) =
  `f"{model_name}@{world_size}@{worker_id}@{chunk_hash_hex}@{dtype}"`
  (`+@tags` optional), ≤ `MAX_KEY_LENGTH=150` (`protocol.py:23`).
  `chunk_hash` is LMCache's own token hash (same `TokenHasher` family), **not**
  a vLLM block hash. `dtype` is one of a small fixed set (`DTYPE_TO_INT`,
  `protocol.py:41-53`).
- KV byte layout: a chunk in one of LMCache's `MemoryFormat`s
  (`memory_management.py:79-127`): `KV_2LTD = [2, num_layers, num_tokens,
  hidden_dim]` (`:84`), or MLA `KV_MLA_FMT = [1, num_layers, num_tokens,
  aligned_head_size]` (`:99`). The `shape` (padded to 4-D, `protocol.py:103-128`)
  and `dtype` are on the header, so the format is self-describing per object. The
  client does its OWN device→host copy into that layout and ships CPU bytes.

### Our baseline

Established by reading our source (worktree base `f6be46e`) and the landed
`ROAD-V1-D4` W1–W5.

**What we already have that an LMCache client reuses directly.**

- **The connector seam is landed.** `KV-OFFLOAD` W4 shipped a working
  `KVConnector` and its scheduler wiring
  (`include/vllm/v1/kv_offload/kv_connector.{h}` + `src/.../kv_connector.cpp`;
  `Scheduler::set_kv_connector`, null = zero change,
  `src/vllm/v1/core/sched/scheduler.{h,cpp}`), and `KV-CONNECTORS` W5 generalizes
  it into the 7-pure-virtual abstract ABI + compile-time registration. **An
  LMCache client is simply another `KVConnector` subclass**, exactly as
  `LMCacheConnectorV1` and `LMCacheMPConnector` are `KVConnectorBase_V1`
  subclasses upstream. It reuses, unchanged: `get_num_new_matched_tokens` (with
  the load-bearing nullopt third state), `update_state_after_alloc`,
  `build_connector_meta`, the worker `start_load_kv`/`wait_for_save`/
  `get_finished` hooks, deferred block-free ownership, and `KVTransferConfig`.
- **The slot formula already matches.** LMCache computes
  `slot = block_id*block_size + offset` itself and demands the engine agree
  (`vllm_v1_adapter.py:368-376`); ours is exactly that
  (`include/vllm/v1/worker/gpu/block_table.h:28` "block_id * block_size +
  within-block offset").
- **The KV cache can be host-resident** for a GPU-free dev/test path
  (`VT_DEVICE_KV_CACHE`, `src/vllm/v1/worker/gpu/runner.cpp:513-517`), and
  transfer primitives exist (`vt::Backend::Copy`/`Alloc`/events,
  `include/vt/backend.h:27-90`) for the device→host copy MODE (1) needs.
- **`Request::block_hashes`** is the incremental list MODE (2)'s hash mode would
  stride, and token ids are available for MODE (1)/token-mode.

**What we do NOT have (net-new, verified).**

- **No TCP socket client anywhere** (`grep -rniE "socket\(|AF_INET|SOCK_STREAM|
  ::connect\(" src include` → 0 hits). MODE (1) needs a ~150-line blocking TCP
  client.
- **No msgpack / msgspec / ZMQ / blake3** in the tree (grep → 0 hits). We DO have
  a hand-rolled **CBOR** encoder for `sha256_cbor` block hashes
  (`src/vllm/v1/core/kv_cache_utils.cpp:169-203`) — evidence that a small,
  self-contained binary codec is within our house style, but msgpack is a
  DIFFERENT format and would be new (MODE 2 only). blake3 is a single-file
  portable C reference impl (MODE 1 and MODE 2 token-mode key hashing).
- **No CUDA-IPC export** of our KV pool (`cudaIpcGetMemHandle` is unused). MODE
  (2) needs it; MODE (1) does not.
- **Quantized KV cannot be persisted today** (`real_page_size_bytes()` throws for
  `kv_quant_mode != kNone`, `src/vllm/v1/kv_cache_interface.cpp:18-20`), so an
  LMCache client covers unquantized KV first, same as the disk tier.

**The hash-compatibility blocker (prior R1) does NOT bind here.** The prior spike
gated LMCache on "our `sha256_cbor` hashes are not byte-compatible with vLLM's
default." That blocker is about sharing vLLM's *block-hash* namespace. **LMCache
does not use vLLM block hashes for its keys** — MODE (1) and MODE (2)/token-mode
key on LMCache's OWN blake3 rolling token hash, computed by the client (MODE 1)
or the server (MODE 2/token). We match *LMCache's* hashing, which is small and
fully specified, not vLLM's. This is the single most important correction this
spec makes to the LMCache disposition.

### Port map

An LMCache client is NOT a 1:1 port of the vLLM-vendored `lmcache_integration/`
glue (that glue imports the external package and would drag it in). It is a
NEW `KVConnector` subclass that speaks the LMCache SERVER's wire directly. The
map is therefore "what our client must implement to speak each server," with the
upstream *protocol definition* as the spec to conform to.

| Upstream protocol (LMCache `8570aad`) | Our target | Disposition |
|---|---|---|
| `lmcache/v1/protocol.py:214-321` `ClientMetaMessage`/`ServerMetaMessage` fixed `struct` | new `src/vllm/v1/kv_offload/lmcache/remote_protocol.cpp` | **direct, trivial** — 186-byte / 36-byte fixed layouts; the whole codec is ~120 lines. NOTE the endianness caveat (native `struct`, §Risks R3) |
| `lmcache/v1/server/__main__.py` + `lm_connector.py:28-177` TCP PUT/GET/EXIST | new `src/vllm/v1/kv_offload/lmcache/remote_client.cpp` | **direct** — a blocking TCP client (`connect`, `sendall`, `recv_into`); MODE (1). No new library |
| `lmcache/utils.py:449-453` `CacheEngineKey.to_string()` | key builder in `remote_protocol.cpp` | **direct** — string format `model@world@worker@hash_hex@dtype`; needs the blake3 chunk hash |
| `lmcache/v1/multiprocess/token_hasher.py:54-79` `TokenHasher` (blake3, chunk 256, rolling) | new `src/vllm/v1/kv_offload/lmcache/token_hasher.cpp` + vendored blake3 | **direct port + one vendored single-file dep**; shared by MODE (1) key and MODE (2) token-mode |
| `lmcache/v1/memory_management.py:79-127` `MemoryFormat` `KV_2LTD`/`KV_MLA_FMT` | KV↔chunk repack in `remote_client.cpp` | **layout adaptation** — transpose our page (`[K|V]` interleaved, or MLA rank-3) into `[2,L,T,D]` / `[1,L,T,H]`; pure memory reshaping, no external code |
| `lmcache/v1/multiprocess/mq.py:82-105,270-353` ZMQ DEALER + `msgspec.msgpack` framing | new `src/vllm/v1/kv_offload/lmcache/mp_client.cpp` over libzmq + a msgpack codec | **MODE (2) only** — new libzmq dep + a msgpack map encoder (field-name keyed). Larger than MODE (1) |
| `lmcache/v1/multiprocess/protocols/{base,engine}.py` `RequestType` + payload classes | request builders in `mp_client.cpp` | **MODE (2) only** — msgpack-encode `IPCCacheServerKey` maps + the fixed payload tuples |
| `platform/cuda/ipc_wrapper.py:92-187` `RawCudaIPCWrapper` + `platform/base/ipc_wrapper.py` pickle | CUDA-IPC export + a fixed pickle template in `mp_client.cpp` | **MODE (2) only, the hard part** — `cudaIpcGetMemHandle` on our pool (portable), then emit a byte-exact pickle stream reconstructing `RawCudaIPCWrapper(...)` (§Risks R2) |
| `lmcache_integration/*`, `lmcache_connector.py`, `lmcache_mp_connector.py` (vLLM glue) | **none** | **NOT PORTED** — importing the package is the thing we are avoiding; these are read as the CALLING CONVENTION only |
| `LMCacheConnectorV1Impl` in-process engine | **none** | **NOT A TARGET** — in-process = Python-only (mode 3) |
| Redis/S3/Mooncake/Infinistore/GDS backends | **none** | **NOT SCHEDULED** (§Risks D2) |

**What mirrors faithfully vs. what needs a native equivalent.**
*Mirror exactly (correctness):* the `struct` field order and widths, the
`ClientCommand`/`ServerReturnCode`/`DTYPE_TO_INT` integer mappings
(`protocol.py:28-65`), the `CacheEngineKey` string format, the blake3 rolling
hash (bit-exact or the server misses), and the `MemoryFormat` chunk layout.
*Native equivalent:* the blocking TCP client (our own, no asyncio), pickle for
the one-time IPC-wrapper (a fixed emitted template, not a pickle library), and
compile-time connector registration (not `importlib`, per the W5 deviation).

### Tests to port

Per [test-porting.md](../test-porting.md), the executable spec travels with the
port. The decisive property is **cross-implementation wire conformance against a
REAL LMCache server**, which no unit test of ours can fake.

| Upstream / target | What it pins | Tier | Disposition |
|---|---|---|---|
| `lmcache-src/tests/**/test_lm_connector*.py`, `test_protocol*.py` (remote-store) | the `struct` header + PUT/GET/EXIST round trip | T-unit | **the conformance oracle for MODE (1)** — re-express as a C++ encode/decode test asserting our 186/36-byte frames byte-match `struct.pack` fixtures captured from the Python side |
| `lmcache-src/tests/**/test_token_hasher*.py` | blake3 rolling chunk hash vectors | T-unit | **bit-exactness gate** — our `TokenHasher` must reproduce the exact `chunk_hash_hex` for known token sequences; a single-bit miss = 0% hits |
| a NEW interop e2e (no vLLM upstream analogue) | our C++ client ↔ a real `lmcache` server, store-then-retrieve | T-e2e | **the go/no-go gate itself** — stand `python -m lmcache.v1.server` (MODE 1) / the MP server (MODE 2), PUT from C++, GET back, assert byte-identical; needs `lmcache` INSTALLED on a test box (it is not today) |
| `vllm .../unit/test_lmcache_connector.py` (23), `test_lmcache_integration.py` (8) | the vLLM-side calling convention | — | **NOT PORTED** — both `import lmcache`; retained ONLY as the enumeration of the `KVConnector` hooks our subclass must honour (already covered by W5's suite) |
| `vllm .../unit/test_kv_connector_lifecycle.py`, `test_config.py` | connector metadata reset + `KVTransferConfig` | T-unit | **REUSED from W5** — an LMCache client is exercised by the same seam suite |

Every case that cannot run until an `lmcache` server exists on a test box is
checked in SKIPPED with the tracked reason, never dropped.

### Gates

1. **Wire-conformance (CPU, no GPU, no server):** our `struct`/msgpack encoders
   byte-match fixtures captured from the Python `serialize()` for every message
   type and dtype; blake3 vectors bit-exact. This is gateable on the dev box with
   only captured fixtures.
2. **Interop round-trip (the go/no-go, needs a real server):** a C++ PUT then GET
   against a running `lmcache.v1.server` returns byte-identical KV; an EXIST after
   PUT returns SUCCESS; a differing key returns FAIL/absent. MODE (1) first
   (no GPU), MODE (2) after.
3. **Key-agreement (needs a real server + a Python vLLM+LMCache peer):** a chunk
   stored by a Python vLLM+LMCache instance is FOUND and correctly decoded by our
   client for the same tokens/model/dtype, proving our blake3 key and `MemoryFormat`
   repack match theirs. This is the true "connect to LMCache" gate.
4. **Output invariance (DGX GB10):** token-exact on the gate models with the
   LMCache client ON, OFF, and vs the pinned vLLM oracle; a cache hit may change
   timing, never a token — the standard precondition, never traded for speed.
5. **Identity safety:** a chunk whose `model_name`/`world_size`/`dtype`/layout
   disagree must MISS, never be decoded as bytes for the wrong model. LMCache's
   key already carries model/world/dtype; our repack must additionally refuse a
   `MemoryFormat`/shape it did not expect.
6. **Every-axis performance (DGX GB10):** match or beat vLLM (launched with the
   equivalent `--kv-transfer-config` LMCache connector) on total/output
   throughput, req/s, TTFT, TPOT/ITL and peak memory at the large-concurrency
   operating point, per [benchmark-protocol.md](../benchmark-protocol.md); a
   hit-rate gain that regresses any axis is a failed change.
7. **Inert when off (every row):** the LMCache connector is opt-in; a same-binary
   A/B with it disabled reproduces the prior binding numbers within run noise.

### Dependencies

- **Rows.** Reuses `KV-EXTERNAL-CACHE` / `KV-CONNECTORS` W5 (the abstract
  `KVConnector` ABI + `KVTransferConfig` + registration) as the seam; that seam
  must land first. Gate 6 inherits the caching spike's `KV-PREFIX-CACHE` W1
  hit-rate counters (already landed) to prove hits. `KV-FP8`/`KV-NVFP4-TURBO`
  gate quantized-KV interop (deferred).
- **New third-party deps, by mode.** MODE (1): **none** for transport (raw TCP)
  + one vendored single-file **blake3**. MODE (2): additionally **libzmq**
  (or its C++ binding) and a **msgpack** map codec — a materially larger
  dependency and attack surface. This asymmetry is the main reason MODE (1) is
  the recommended first target.
- **LMCache must be INSTALLED on a test box** for gates 2/3 — it is not on any
  project box today (confirmed: `~/venvs` has no `lmcache`, and it is an opt-in
  extras entry `requirements/kv_connectors.txt:1`). W-plan W2 budgets standing a
  server up. The LMCache package is also **unpinned/moving** (§Risks R4).
- **Hardware.** W0–W2 (protocol codec, TCP client, blake3, MODE-1 interop) are
  **CPU-only**. MODE (2) CUDA-IPC and every throughput gate need **DGX GB10**
  under one `flock /tmp/gpu` per series.

### Work breakdown

Ordered highest-value / lowest-risk first. Nothing is started; every row is
`open`. This is a PLAN inside an analysis spec — no code is claimed.

| W | Deliverable | Gate | HW | State |
|---:|---|---|---|---|
| W0 | **This analysis** — the two wire protocols mapped `file:line`, the feasibility verdict, showstoppers, and the recommendation of MODE (1) first | this document | CPU | **DONE (analysis)** |
| W1 | **MODE (1) protocol codec + blake3** — `ClientMetaMessage`/`ServerMetaMessage` `struct` encode/decode, `CacheEngineKey.to_string`, the `TokenHasher`, all byte/bit-exact against captured Python fixtures | gate 1 | CPU | open |
| W2 | **MODE (1) TCP client + `MemoryFormat` repack + interop round-trip** — a blocking TCP `LMCacheRemoteClient`, device→host repack into `KV_2LTD`, PUT/GET/EXIST against a REAL `lmcache.v1.server`; stand the server up | gates 2, 5 | CPU | open |
| W3 | **MODE (1) as a `KVConnector` subclass** wired over the W5 seam (opt-in, default-off), + the key-agreement gate against a Python vLLM+LMCache peer | gate 3, then 4/6/7 | CPU, then DGX | open |
| W4 | **MODE (2) MP client** — libzmq DEALER + msgpack codec + `RawCudaIPCWrapper` CUDA-IPC export + the fixed pickle template; only if MODE (1) proves insufficient (e.g. a deployment that only runs the MP server) | gates 2–7 for MP | DGX | open — **contingent** |
| — | Redis/S3/Mooncake/Infinistore/GDS backends, CB/blend, P2P | **NOT SCHEDULED** (§Risks D2) | — | dispositioned |

### Risks/decisions

**R1 — the wire IS specified; the prior "no protocol" verdict is REFUTED, but the
FORMAT VERSIONING is the real risk.** Both protocols are fully specified and
portable. What is NOT stable is the SCHEMA: the MP protocol explicitly warns of
"a version mismatch between the lmcache client and lmcache server"
(`mq.py:304-313`), the payload classes are validated at runtime, and LMCache is
an unpinned, fast-moving package (§R4). The remote-store `struct` is far more
stable (it is a tiny fixed layout that has changed rarely) but is still not a
committed public contract. **Decision:** target MODE (1) (the stabler, simpler
wire), pin a specific `lmcache` version for the interop gate, and treat the
codec as version-scoped — exactly the maintenance the mechanical-port property
was meant to avoid, so this is an INTEROP feature with a standing sync cost, not
a core-parity feature. This is the honest residue of the prior spike's concern.

**R2 — pickle appears in exactly one place, and it is confined to MODE (2)'s
one-time registration.** The KV-cache IPC wrapper crosses the MP wire as a
pickled blob (`ipc_wrapper.py` `Serialize` = `pickle.dumps`). There is **zero
pickle in MODE (1)** and **zero pickle in the MODE (2) hot path** (STORE/RETRIEVE
carry msgpack keys + a raw CUDA event handle). To speak MODE (2) from C++ we
would emit a byte-exact pickle stream reconstructing a `RawCudaIPCWrapper` with a
fixed field set — annoying but bounded (a hand-written opcode template for one
known class), done ONCE per worker at registration. **Decision:** pickle is NOT
a showstopper; it is a reason to prefer MODE (1), and if MODE (2) is needed the
pickle template is a small, testable, fixed artifact.

**R3 — the remote-store `struct` uses NATIVE byte order and alignment, not
network order.** `struct.pack("iiiiiiiii150s", …)` (`protocol.py:238`) has no
`<`/`>`/`!` prefix, so it is native-endian and native-aligned. On our
little-endian x86/ARM targets talking to a same-arch co-located or LAN server
this is fine (all fields are 4-byte `int` + a 150-byte string, no padding
surprises), but it is **not portable across endianness** and we must document
that the client and server must share byte order. **Decision:** encode
little-endian explicitly in our codec and assert the server is same-endian
(true for every realistic deployment); note it in user docs.

**R4 — CUDA-IPC makes MODE (2) co-located and layout-coupled; it is speakable but
not a network cache.** In MODE (2) the LMCache server opens OUR GPU KV pool via
an IPC handle and runs its own copy kernels on it (`RawCudaIPCWrapper.to_tensor`,
`ipc_wrapper.py:154-187`). That requires same-node + same-visible-GPU, and it
couples LMCache to our exact page layout/stride/dtype. `cudaIpcGetMemHandle`/
`cudaIpcOpenMemHandle` are plain CUDA runtime C APIs we CAN call, and
`RawCudaIPCWrapper` is the exact non-PyTorch recipe — but the coupling is deep.
**Decision:** MODE (1) (which ships CPU bytes over TCP, no GPU coupling, works
across nodes) is strictly the better first target; MODE (2) is contingent (W4)
on a deployment that only offers the MP server.

**R5 — torch types on the wire are benign.** `torch.dtype`/`torch.Size` cross the
MP msgpack wire as strings/lists via enc/dec hooks (`custom_types.py:221-255`),
and dtypes map to fixed ints in the remote-store header (`protocol.py:41-53`).
No `torch.save`/tensor-pickling is on either wire. **Not a showstopper.**

**R6 — the seam reuse is high, so incremental cost is low.** An LMCache client is
one more `KVConnector` subclass; it reuses the entire landed W4/W5 scheduler and
worker seam (get_num_new_matched_tokens/update_state_after_alloc/
build_connector_meta/start_load_kv/wait_for_save/get_finished, deferred free,
`KVTransferConfig`) and the matching `slot = block_id*block_size+offset`. The
net-new surface is ONLY the transport + serialization + key/repack — the ~4 files
in the port map. **Decision:** sequence LMCache immediately after W5 lands the
abstract ABI; do not build a bespoke connector path for it.

**D2 — the other storage backends are NOT scheduled.** LMCache's in-process
engine can also offload to Redis, S3, Mooncake, Infinistore, GDS/HF3FS
(`storage_backend/connector/*_connector.py`). Each is a distinct store protocol
with its own external dependency; speaking Redis directly (RESP) would be a
different feasibility question (and arguably a way to SHARE a cache with a
Python LMCache without any LMCache code at all). Enumerated for completeness,
**dispositioned NOT SCHEDULED** — out of scope for "connect to LMCache."

**Decision, restated.** A from-scratch C++ LMCache client is **FEASIBLE and the
user's ZMQ recollection is correct**. Recommend MODE (1) (`lm://` remote-store:
plain TCP, fixed `struct` header, raw bytes, blake3 key, no pickle, no CUDA-IPC,
no ZMQ) as the first and lowest-risk target, reached over the already-landed
`KVConnector` seam; MODE (2) (ZMQ + msgpack + CUDA-IPC + one pickle template) is
the more powerful but more-coupled fallback. The single standing risk is LMCache
being an unpinned moving target (R1/R4), which makes this an INTEROP feature with
a version-sync cost rather than a mechanical core port — but it is buildable, and
no part of it requires the `lmcache` PyPI package inside our process.
