"""
سرور آزمایشی (Mock) برای توسعه و تست بدون GPU
=============================================
یک پیاده‌سازی سبک و سازگار با OpenAI که هر دو سرویس llama.cpp را
شبیه‌سازی می‌کند — فقط با کتابخانهٔ استاندارد پایتون:

    - GET  /health
    - POST /v1/chat/completions   (عادی و جریانی SSE)
    - POST /v1/embeddings         (امبدینگ bag-of-words هش‌محور)

رفتار مدل برای تست حلقهٔ ابزار: اگر در پیام‌ها پاسخِ ابزار
(بلوک <tool_response>) نباشد، مدل یک فراخوانی ابزار جستجو «شبیه‌سازی‌شده»
برمی‌گرداند؛ دور بعدی که نتیجهٔ ابزار را دید، پاسخ نهایی می‌سازد.

اجرا:

    python scripts/mock_server.py --port 8090

سپس در ‎.env‎ این مقادیر را بگذارید:

    LLM_BASE_URL=http://localhost:8090/v1
    EMBEDDING_BASE_URL=http://localhost:8090/v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

EMBEDDING_DIM = 1024


# ═══════════════════════════ منطق امبدینگ ═══════════════════════════
def mock_embedding(text: str) -> list[float]:
    """امبدینگ ساختگیِ bag-of-words با هشِ پایدار (بدون هوش مصنوعی)."""
    vector = [0.0] * EMBEDDING_DIM
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % EMBEDDING_DIM
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


# ═══════════════════════════ منطق چت ═══════════════════════════
def mock_chat_reply(messages: list[dict]) -> str:
    """پاسخ ساختگی مدل برای تست حلقهٔ ابزار RAG."""
    has_tool_response = any("<tool_response>" in (m.get("content") or "") for m in messages)
    if has_tool_response:
        return (
            "بر اساس جست‌وجو در دانش‌نامه، این پاسخ نهاییِ آزمایشی است و "
            "محتوا از فایل test.pdf بازیابی شده است. (پاسخ تولیدشده توسط سرور آزمایشی)"
        )
    # هنوز ابزار فراخوانی نشده → مدل یک فراخوانی ابزار شبیه‌سازی می‌کند
    return (
        "برای پاسخ بهتر اجازه بده در دانش‌نامه جست‌وجو کنم.\n"
        "<tool_call>\n"
        '{"name": "search_knowledge_base", "arguments": {"query": "متن آزمایشی", "top_k": 3}}'
        "\n</tool_call>"
    )


# ═══════════════════════════ هندلر HTTP ═══════════════════════════
class MockHandler(BaseHTTPRequestHandler):
    """هندلر سازگار با OpenAI برای /health و /v1/*."""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # خروجی کنسول تمیز بماند

    # ── GET ─────────────────────────────────────────────────
    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, status=404)

    # ── POST ────────────────────────────────────────────────
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "invalid json"}, status=400)
            return

        if path == "/v1/embeddings":
            self._handle_embeddings(payload)
        elif path == "/v1/chat/completions":
            self._handle_chat(payload)
        else:
            self._send_json({"error": "not found"}, status=404)

    # ── endpoint ها ─────────────────────────────────────────
    def _handle_embeddings(self, payload: dict) -> None:
        inputs = payload.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [
            {"object": "embedding", "index": i, "embedding": mock_embedding(text)}
            for i, text in enumerate(inputs)
        ]
        self._send_json({"object": "list", "data": data, "model": "mock-embedding"})

    def _handle_chat(self, payload: dict) -> None:
        messages = payload.get("messages", [])
        reply = mock_chat_reply(messages)
        if payload.get("stream"):
            self._handle_chat_stream(reply)
        else:
            self._send_json({
                "id": "mock-chat",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", "mock"),
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": reply},
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

    def _handle_chat_stream(self, reply: str) -> None:
        """ارسال پاسخ به‌صورت SSE — کلمه به کلمه با تاخیر کوتاه."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for word in reply.split(" "):
            chunk = {
                "id": "mock-chat", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": "mock",
                "choices": [{"index": 0, "finish_reason": None,
                             "delta": {"content": word + " "}}],
            }
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ── ابزار ───────────────────────────────────────────────
    def _send_json(self, body: dict, status: int = 200) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="سرور آزمایشی سازگار با OpenAI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), MockHandler)
    print(f"سرور آزمایشی روی http://{args.host}:{args.port} بالا آمد …")
    print("endpoints: GET /health | POST /v1/chat/completions | POST /v1/embeddings")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nسرور آزمایشی خاموش شد.")


if __name__ == "__main__":
    main()
