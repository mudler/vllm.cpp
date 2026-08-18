#!/usr/bin/env python3
"""Does an SSE COMMENT frame make a client drop the request? (#931)

A socket server that emits OUR server's exact SSE bytes, so the REAL pinned
`vllm bench serve` can be pointed at them without a GPU, a checkpoint or the
box's GPU lock. It isolates ONE variable: whether a keep-alive comment frame
precedes a request's first data frame.

Run both arms (the oracle venv supplies the client; nothing here needs a GPU):

    python3 tools/bench/sse_frame_client_probe.py --port 8101 --comment-on -1 &
    vllm bench serve --backend openai --base-url http://127.0.0.1:8101 \
      --endpoint /v1/completions --model gate --tokenizer <ckpt> \
      --dataset-name random --random-input-len 1024 --random-output-len 8 \
      --random-range-ratio 0 --num-prompts 6 --max-concurrency 1 \
      --request-rate inf --ignore-eos --temperature 0 --seed 0 \
      --save-result --save-detailed --result-dir <out> \
      --result-filename noping.json --disable-tqdm

    # ... and again with --comment-on 1 (see --comment-on below for the ordinals).

Measured on 2026-08-15 against `0.23.1rc1.dev1511+g555967922`:

    --comment-on -1  ->  completed 6, failed 0
    --comment-on  1  ->  completed 5, failed 1,
                         errors[1] = "Never received a valid chunk to calculate
                         TTFT.This response will be marked as failed!",
                         output_lens[1] = 0, ttfts[1] = 0.0

Nothing but the comment frame differs between the two arms.


It reproduces one thing only: the framing our OpenAI server puts on the wire for
a streaming /v1/completions request. The frames are taken from the source --
`data: <json>\\n\\n` per delta, a `data: <json>\\n\\n` usage frame, `data: [DONE]\\n\\n`
(serving_completion.cpp:61-67,117) and the keep-alive comment `:\\n\\n`
(serving_utils.h:42 kSsePingFrame, emitted by AssignSseWaitResult when the
collector wait expires, serving_utils.cpp:242-251). Each frame is written as its
own HTTP chunk because cpp-httplib writes one chunk per content-provider call
(api_server.cpp:976-1006) and TCP_NODELAY is on (api_server.cpp:80).

--comment-on N: emit ONE comment frame immediately before the first data frame
of request ordinal N. Anything else is byte-identical between the two arms.

Ordinal N counts the connections this server accepts, and under the recipe
above ordinal 0 is the client's FIRST TIMED REQUEST -- not an untimed test run.
At the pin `--ready-check-timeout-sec` defaults to 0, so the initial test
request is skipped ("Skipping endpoint ready check.", serve.py:857-872), and
`--num-warmups` defaults to 0, so no warmup request is issued (:875-899); the
recipe passes neither flag. Pass either one and every ordinal shifts. The
measurement above is unaffected: it was taken with `--comment-on 1`, which is
the second timed request, and the failure it recorded is at index 1.
"""
import argparse
import json
import socket
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--output-len", type=int, default=8)
parser.add_argument("--itl-ms", type=float, default=20.0)
parser.add_argument("--comment-on", type=int, default=-1)
parser.add_argument("--prompt-tokens", type=int, default=1005)
args = parser.parse_args()

ordinal_lock = threading.Lock()
ordinal_counter = [0]


def frames(n_tokens, emit_comment):
    out = []
    if emit_comment:
        out.append(":\n\n")
    for index in range(n_tokens):
        out.append(
            "data: "
            + json.dumps(
                {
                    "id": "cmpl-0",
                    "object": "text_completion",
                    "created": 1,
                    "model": "gate",
                    "choices": [
                        {"index": 0, "text": f" t{index}", "finish_reason": None}
                    ],
                }
            )
            + "\n\n"
        )
    out.append(
        "data: "
        + json.dumps(
            {
                "id": "cmpl-0",
                "object": "text_completion",
                "created": 1,
                "model": "gate",
                "choices": [],
                "usage": {
                    "prompt_tokens": args.prompt_tokens,
                    "completion_tokens": n_tokens,
                    "total_tokens": args.prompt_tokens + n_tokens,
                },
            }
        )
        + "\n\n"
    )
    out.append("data: [DONE]\n\n")
    return out


def read_request(conn):
    buf = b""
    while b"\r\n\r\n" not in buf:
        piece = conn.recv(65536)
        if not piece:
            return None, None
        buf += piece
    head, rest = buf.split(b"\r\n\r\n", 1)
    length = 0
    for line in head.decode("latin-1").split("\r\n")[1:]:
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1])
    while len(rest) < length:
        rest += conn.recv(65536)
    return head.decode("latin-1"), rest[:length]


def handle(conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            head, body = read_request(conn)
            if head is None:
                return
            path = head.split(" ")[1] if " " in head else ""
            if not path.startswith("/v1/completions"):
                payload = b'{"object":"list","data":[{"id":"gate","object":"model"}]}'
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
                )
                continue
            with ordinal_lock:
                ordinal = ordinal_counter[0]
                ordinal_counter[0] += 1
            emit_comment = ordinal == args.comment_on
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            for frame in frames(args.output_len, emit_comment):
                blob = frame.encode()
                conn.sendall(b"%x\r\n" % len(blob) + blob + b"\r\n")
                time.sleep(args.itl_ms / 1000.0)
            conn.sendall(b"0\r\n\r\n")
    except (BrokenPipeError, ConnectionResetError, OSError):
        return
    finally:
        try:
            conn.close()
        except OSError:
            pass


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", args.port))
server.listen(64)
print(f"frame-mimic listening on {args.port} comment_on={args.comment_on}", flush=True)
while True:
    client, _ = server.accept()
    threading.Thread(target=handle, args=(client,), daemon=True).start()
