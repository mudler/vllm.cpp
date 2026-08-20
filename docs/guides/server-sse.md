# Configure server-sent event keepalives

Use keepalives only when a proxy closes inactive response streams.

Async chat and completion streams can emit SSE comment frames (`:\n\n`) while
waiting on the engine during a long prefill or time to first token (TTFT). A
proxy with an inactivity timeout sees body bytes before the first token. The
interval is `VT_SERVER_SSE_PING_S`. The default is `0`, which disables
keepalives. A positive value enables them and is clamped to 600.


Leave keepalives off unless a proxy closes inactive streams. Some benchmark
clients do not accept SSE comment frames. Keep `VT_SERVER_SSE_PING_S=0` during
comparative measurements. See
[the server-concurrency spec](../../.agents/specs/server-concurrency-failures.md)
for the compatibility evidence.

Comment frames are not `data:` events and carry no tokens, and neither setting
turns token streaming into a poll loop. At the `0` default both streams take the
blocking `get_output()` on that request's own collector
(`serving_completion.cpp:39-43`, `serving_chat.cpp:333-337`), which returns the
instant the engine has something for that request. A positive interval swaps in
`get_output_for()`, the same wait with a timeout attached, and the timeout only
expires when the collector produced nothing at all. Deltas are therefore never
collapsed or delayed either way.

An invalid value disables the keepalive without returning an error.
`VT_SERVER_SSE_PING_S=fifteen`, an empty value and an unset variable all resolve
to `0`, so if you enable this and no comment frames appear, check the spelling
before looking anywhere else. The fallback points at OFF deliberately: under the
previous default a typo silently switched the keepalive ON, and that is the
direction that costs you requests.

The interval bounds silence on one request's stream. It does not bound TTFT.
Each wait restarts whenever anything reaches that request, so a long
prefill that keeps producing intermediate results never pings however long its
first token takes, while a request whose stream goes quiet for the whole
interval does.
