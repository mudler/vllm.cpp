# Configure server-sent event keepalives

Use keepalives only when a proxy closes inactive response streams.

Async chat/completion streams can emit SSE **comment** frames (`:\n\n`) while
waiting on the engine (long prefill / TTFT), so a proxy with an inactivity
timeout sees body bytes before the first token. Interval is
`VT_SERVER_SSE_PING_S`, **default `0`, off**; a positive value enables it and
is clamped to 600.

**It is off by default, and it should stay off unless a proxy forces your
hand.** vLLM's streaming endpoints emit no comment frame at any point, so a
server that sends one is putting a byte on the wire that OpenAI-compatible
clients written against vLLM have never had to parse. vLLM's own benchmark
client is one of them: `vllm bench serve` strips each network chunk before
parsing, which destroys the `\n\n` separator at chunk boundaries, and its only
resynchronisation path looks for a `data: ` prefix, so one comment frame
arriving before a request's first token makes it report
`Never received a valid chunk to calculate TTFT` and count that request
**failed**, while this server completes it normally and logs nothing. The
requests that reach a keepalive are by construction the slowest ones, so the
effect is to delete your own worst latencies from a measurement
([#931](https://github.com/mudler/vllm.cpp/issues/931),
[#577](https://github.com/mudler/vllm.cpp/issues/577)).

Comment frames are not `data:` events and carry no tokens, and neither setting
turns token streaming into a poll loop. At the `0` default both streams take the
blocking `get_output()` on that request's own collector
(`serving_completion.cpp:39-43`, `serving_chat.cpp:333-337`), which returns the
instant the engine has something for that request. A positive interval swaps in
`get_output_for()`, the same wait with a timeout attached, and the timeout only
expires when the collector produced nothing at all. Deltas are therefore never
collapsed or delayed either way.

**A value the server cannot parse disables the keepalive; it is not an error.**
`VT_SERVER_SSE_PING_S=fifteen`, an empty value and an unset variable all resolve
to `0`, so if you enable this and no comment frames appear, check the spelling
before looking anywhere else. The fallback points at OFF deliberately: under the
previous default a typo silently switched the keepalive ON, and that is the
direction that costs you requests.

**The interval bounds silence on one request's stream, not its time to first
token.** Each wait restarts whenever anything reaches that request, so a long
prefill that keeps producing intermediate results never pings however long its
first token takes, while a request whose stream goes quiet for the whole
interval does.
