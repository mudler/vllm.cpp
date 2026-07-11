"""Native output-ID precondition tests.

Source: SGLang ``tokenizer_manager.py:2640-2660`` @ 28b095c.  Token identity is
checked directly; detokenize/re-tokenize is intentionally absent.
"""

from __future__ import annotations

import http.server
import json
import threading
import unittest

from tools.bench.run_serve_low import capture_native_output_ids
from tools.bench.serve_low_common import HarnessError


class _NativeHandler(http.server.BaseHTTPRequestHandler):
    output_ids = [10, 11, 12, 13]
    payload: dict | None = None

    def log_message(self, *_args) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        type(self).payload = json.loads(self.rfile.read(length))
        body = json.dumps(
            {
                "meta_info": {"completion_tokens": len(type(self).output_ids)},
                "output_ids": type(self).output_ids,
                "text": "diagnostic-only",
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TokenIdTests(unittest.TestCase):
    def setUp(self) -> None:
        _NativeHandler.output_ids = [10, 11, 12, 13]
        _NativeHandler.payload = None
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NativeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/generate"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_exact_native_ids_and_request_shape(self) -> None:
        ids = capture_native_output_ids(
            self.url,
            [1, 2, 3],
            expected_output_len=4,
            oracle_output_ids=[10, 11, 12, 13],
        )
        self.assertEqual(ids, [10, 11, 12, 13])
        self.assertEqual(_NativeHandler.payload["input_ids"], [1, 2, 3])
        self.assertEqual(_NativeHandler.payload["sampling_params"]["max_new_tokens"], 4)
        self.assertFalse(_NativeHandler.payload["stream"])

    def test_length_or_token_mismatch_prevents_binding(self) -> None:
        _NativeHandler.output_ids = [10, 11, 12]
        with self.assertRaises(HarnessError):
            capture_native_output_ids(self.url, [1], expected_output_len=4)
        _NativeHandler.output_ids = [10, 11, 12, 13]
        with self.assertRaises(HarnessError):
            capture_native_output_ids(
                self.url,
                [1],
                expected_output_len=4,
                oracle_output_ids=[10, 99, 12, 13],
            )


if __name__ == "__main__":
    unittest.main()
