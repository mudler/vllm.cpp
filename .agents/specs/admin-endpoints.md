# Admin endpoints (`SERVE-ADMIN`, ROAD-V1-C8)

Live spec for the `SERVE-ADMIN` engine-matrix row: the OpenAI-server admin /
dev-RL control endpoints, mirroring vLLM 0.26.0.dev0 (`555967922`). First landed
surface: `POST /abort_requests`. The rest (`/sleep`, `/wake_up`, `/is_sleeping`,
`/pause`, `/resume`, `/start_profile`, `/stop_profile`, weight-update / EP
endpoints) remain INVENTORIED.

## Scope

Add admin control endpoints as ADDITIVE, opt-in routes on the `ApiServer`. Each is
registered only when its backing dependency is attached, so a server without the
backing is byte-identical to before (the route simply 404s). First endpoint:
`POST /abort_requests` — abort in-flight requests by id without pausing the
scheduler.

## Upstream chain

- `vllm/entrypoints/serve/dev/rlhf/api_router.py:94-138` (`abort_requests`):
  parses the JSON body, reads `request_ids`; a non-empty list aborts exactly those
  (external) ids via `engine.abort(request_ids)`; an empty/missing list aborts ALL
  in-flight requests (upstream enumerates `output_processor.request_states.keys()`
  + `parent_requests.keys()` and calls `engine.abort(..., internal=True)`).
  Response `{"status":"aborted","aborted":<count>}`. Invalid JSON →
  `HTTPException(400, detail="Invalid JSON format")` → `{"detail":...}`. An abort
  failure → 500 `{"error":"Failed to abort requests: ..."}`.
- `vllm/entrypoints/serve/dev/sleep/api_router.py:21`,
  `vllm/entrypoints/serve/dev/rlhf/api_router.py:29,74,136`,
  `vllm/entrypoints/serve/profile/api_router.py:21` — the still-INVENTORIED
  `/sleep`/`/wake_up`, `/pause`/`/resume`, `/start_profile`/`/stop_profile` and
  weight-update endpoints.

## Our baseline

Before this row: `ApiServer` exposed the base routes plus the C8 utility surface
(`/tokenize`, `/detokenize`, `/tokenizer_info`, `/ping`, `/server_info`,
`/reset_prefix_cache`, `/metrics`) as opt-in additive routes. The engine abort
path already exists: `AsyncLLM::abort(const std::vector<std::string>&)`
(`include/vllm/v1/engine/async_llm.h:115`) forwards to
`OutputProcessor::abort_requests` then `EngineCore` (mirrors the upstream order).
No admin/control route existed yet.

## Port map

- `handle_abort_requests(request_body)`
  (`src/vllm/entrypoints/openai/api_server.cpp:1611`) parses `{request_ids:[...]}`
  and forwards the (possibly empty) id list to an injected abort callback
  (`set_abort_requests`, `include/vllm/entrypoints/openai/api_server.h:350`)
  returning the number aborted. The callback is wired to
  `AsyncLLM::abort(const std::vector<std::string>&)`
  (`include/vllm/v1/engine/async_llm.h:115`).
- Response `{"status":"aborted","aborted":<count>}`; malformed JSON → 400
  `{"detail":"Invalid JSON format"}`; abort failure → 500 `{"error":...}` — all
  three shapes reproduce the upstream router verbatim.
- The route registers only when the abort callback is attached (opt-in additive),
  in `register_routes` alongside the other opt-in C8 routes.

## Tests to port

Upstream `tests/entrypoints/serve/dev/rlhf` abort behavior + the engine abort
coverage of `tests/v1/engine/test_async_llm.py`, re-expressed in
`tests/vllm/entrypoints/openai/test_api_server.cpp`:

- `:1104` — shape + callback wiring (explicit ids passthrough, empty→abort-all
  branch, malformed→400 `{"detail":...}`).
- `:1143` — aborts an in-flight AsyncLLM request added under a known id
  (`has_unfinished_requests()` → false).
- `:1250` — opt-in route gate over a real socket (404 no-callback → 200 attached),
  RED-first.

## Gates

- Parity: `/abort_requests` returns the exact vLLM-0.26 success/`detail`/`error`
  shapes on the fixed CPU case; RED-first via the opt-in route gate (404 when no
  callback → 200 when attached).
- Behavior: aborting an in-flight `AsyncLLM` request by id tears it down
  (`has_unfinished_requests()` → false).
- Inertness: opt-in — a server without the callback does not register
  `/abort_requests`; existing serving byte-identical (the pre-existing api_server /
  conformance / serving suites unchanged). Clean CPU `-Werror`.

## Dependencies

The engine abort path (`SERVE-ASYNC-LLM`, already present — consumed only via an
injected callback, no engine internals touched).

## Work breakdown

- W1: `POST /abort_requests` (explicit-ids + empty→abort-all callback contract) +
  tests — DONE (`CLAIM-C8-SERVE-ENDPOINTS`, 2026-07-28).
- W2: production `main.cpp` wiring of `/abort_requests` — DONE
  (`CLAIM-C8-SERVE-PROD-WIRING`, 2026-07-28). See "Production wiring" below.
- W3 (residual): the empty-list "abort ALL" enumeration needs an `AsyncLLM`
  active-request-id accessor (not exposed today — explicit-id abort IS wired to
  the live engine; empty ids report 0 aborted rather than fabricate an all-abort
  we cannot reach) — OPEN.
- W4+ (residual): `/sleep`/`/wake_up`/`/is_sleeping`, `/pause`/`/resume`,
  `/start_profile`/`/stop_profile`, weight-update/EP endpoints — INVENTORIED, OPEN.

## Production wiring (`CLAIM-C8-SERVE-PROD-WIRING`, 2026-07-28)

`examples/server/main.cpp` now wires `POST /abort_requests` to the LIVE `AsyncLLM`
through the shared `ConfigureUtilityEndpoints` seam
(`src/vllm/entrypoints/openai/api_server.cpp`), mirroring vLLM 0.26 @ `555967922`.

Default gating: `/abort_requests` is a DEV-mode endpoint in vLLM — `build_app`
registers the dev/rlhf router ONLY under `if envs.VLLM_SERVER_DEV_MODE`
(`vllm/entrypoints/openai/api_server.py:238-240` →
`vllm/entrypoints/serve/__init__.py:35` `register_vllm_dev_api_routers` →
`serve/dev/rlhf/api_router.py:94`); `VLLM_SERVER_DEV_MODE` defaults `0`
(`vllm/envs.py:157,1350`). We mirror that gate with a new CLI flag
`--enable-server-dev-mode` (default off) — a default production server 404s
`/abort_requests`, byte-identical to before.

When `--enable-server-dev-mode` is set, the abort callback routes explicit
`request_ids` through `AsyncLLM::abort` (`include/vllm/v1/engine/async_llm.h:115`)
and reports the exact drop in unfinished requests (`get_num_unfinished_requests()`
before − after; `abort()` synchronously removes the request states under the
output-processor lock, so the delta is exact). The empty-`request_ids` "abort ALL"
contract is a NAMED RESIDUAL: `AsyncLLM` exposes no active-request-id enumeration
(missing accessor: `AsyncLLM::active_request_ids()` / an internal abort-all path),
so empty ids abort nothing and report `0` rather than fabricate an all-abort the
frontend cannot reach.

## Risks/decisions

- Decision: `/abort_requests` takes an injected abort callback so the serving layer
  does not reach into engine internals; the "abort all" enumeration is the
  callback's responsibility (deferred to production wiring).
- Risk: none to generation — the endpoint is control-only and opt-in additive.
