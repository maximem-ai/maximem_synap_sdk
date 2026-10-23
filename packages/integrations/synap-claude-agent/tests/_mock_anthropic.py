"""A minimal Anthropic Messages API, to drive the real Claude Code CLI offline.

Point the CLI at it with ``ANTHROPIC_BASE_URL``. It answers every ``/v1/messages``
call (streamed or not, the main turn and Claude Code's own side calls alike) with
one fixed text, and keeps every request body so a test can read what the model
was sent.
"""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockAnthropic:
    def __init__(self, reply="Booked."):
        self.reply = reply
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n)
                try:
                    body = json.loads(raw or b"{}")
                except Exception:
                    body = {}
                outer.requests.append({"path": self.path, "body": body})
                if not self.path.startswith("/v1/messages") or self.path.endswith("count_tokens"):
                    out = json.dumps({"input_tokens": 10}).encode()
                    self.send_response(200); self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(out))); self.end_headers(); self.wfile.write(out); return
                text = outer.reply(body) if callable(outer.reply) else outer.reply
                msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": body.get("model", "m"),
                       "content": [], "stop_reason": None, "stop_sequence": None,
                       "usage": {"input_tokens": 10, "output_tokens": 1}}
                if body.get("stream"):
                    self.send_response(200); self.send_header("content-type", "text/event-stream"); self.end_headers()
                    def ev(name, data):
                        self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()); self.wfile.flush()
                    ev("message_start", {"type": "message_start", "message": msg})
                    ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
                    ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})
                    ev("content_block_stop", {"type": "content_block_stop", "index": 0})
                    ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}})
                    ev("message_stop", {"type": "message_stop"})
                else:
                    msg["content"] = [{"type": "text", "text": text}]; msg["stop_reason"] = "end_turn"
                    out = json.dumps(msg).encode()
                    self.send_response(200); self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(out))); self.end_headers(); self.wfile.write(out)

            def do_GET(self):
                out = b"{}"
                self.send_response(200); self.send_header("content-type", "application/json")
                self.send_header("content-length", "2"); self.end_headers(); self.wfile.write(out)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
