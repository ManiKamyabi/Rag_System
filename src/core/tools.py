"""
سیستم ابزار (Function Calling)
==============================
سیستم امبدینگ/جست‌وجو به‌صورت «ابزار» در اختیار مدل قرار می‌گیرد تا
هر جا نیاز بود خودش آن را فراخوانی کند:

  - search_knowledge_base : جست‌وجوی معنایی در ایندکس FAISS
                            (پرس‌وجو embed می‌شود و مرتبط‌ترین چانک‌ها
                            با نام فایل و امتیاز شباهت برمی‌گردند)
  - get_knowledge_base_stats : آمار دانش‌نامه (تعداد فایل/چانک و ابعاد)

ابزارها در پرامپت سیستمی (قالب hermes سازگار با Qwen3) معرفی می‌شوند
و فراخوانی‌ها در ماژول tool_parsing از خروجی مدل تشخیص داده می‌شوند؛
بنابراین بدون نیاز به ‎--jinja‎ روی llama.cpp کار می‌کند.
"""

from __future__ import annotations

import logging

from .models import RetrievalResult, ToolCall, ToolExecutionResult
from .retriever import Retriever
from .vector_store import VectorStore

logger = logging.getLogger("rag.tools")


TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "جست‌وجوی معنایی در دانش‌نامهٔ محلی (ایندکس FAISS). هر زمان که "
                "پاسخ سوال کاربر در بافتِ ارائه‌شده نیست یا به اطلاعات بیشتری "
                "نیاز داشتی، یک عبارت جست‌وجوی مناسب بساز و این ابزار را صدا بزن."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "عبارت جست‌وجو (ترجیحاً به زبان محتوای اسناد)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "تعداد نتایج؛ عددی بین ۱ تا ۱۵ (پیش‌فرض ۵)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_knowledge_base_stats",
            "description": "گرفتن آمار دانش‌نامه: تعداد فایل‌ها، تعداد چانک‌ها و ابعاد بردارها.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class ToolRegistry:
    """ثبت، اعتبارسنجی و اجرای فراخوانی ابزارها توسط مدل.

    خطای هر ابزار به‌صورت متنِ خطا به مدل برمی‌گردد تا مدل بتواند آن را
    به کاربر توضیح دهد یا مسیر دیگری را امتحان کند.
    """

    def __init__(self, retriever: Retriever, vector_store: VectorStore,
                 tool_result_max_chars: int = 3000, chunk_preview_chars: int = 1200):
        self.retriever = retriever
        self.vector_store = vector_store
        self.tool_result_max_chars = tool_result_max_chars
        self.chunk_preview_chars = chunk_preview_chars

    # ── تعاریف ──────────────────────────────────────────────
    @staticmethod
    def definitions() -> list[dict]:
        """تعریف ابزارها (اسکیمای OpenAI) برای ساخت پرامپت."""
        return TOOL_DEFINITIONS

    @staticmethod
    def tool_names() -> list[str]:
        return [tool["function"]["name"] for tool in TOOL_DEFINITIONS]

    # ── اجرا ────────────────────────────────────────────────
    def execute(self, call: ToolCall) -> ToolExecutionResult:
        """اجرای یک فراخوانی؛ همیشه نتیجه (یا متن خطا) برمی‌گردد."""
        handler = getattr(self, f"_run_{call.name}", None)
        if handler is None:
            logger.warning("مدل ابزار ناشناخته «%s» را فراخوانی کرد.", call.name)
            return ToolExecutionResult(text=f"خطا: ابزار «{call.name}» وجود ندارد.")
        try:
            return handler(**call.arguments) if call.arguments else handler()
        except TypeError as exc:
            return ToolExecutionResult(
                text=f"خطا در آرگومان‌های ابزار «{call.name}»: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 — خطای ابزار نباید خط لوله را بکشد
            logger.exception("خطا در اجرای ابزار «%s»", call.name)
            return ToolExecutionResult(text=f"خطا در اجرای ابزار «{call.name}»: {exc}")

    # ── ابزار ۱: جست‌وجوی معنایی ─────────────────────────────
    def _run_search_knowledge_base(self, query: str = "", top_k: int = 5
                                   ) -> ToolExecutionResult:
        query = str(query).strip()
        if not query:
            return ToolExecutionResult(text="خطا: پارامتر query خالی است.")
        top_k = self._clamp_int(top_k, 1, 15, default=5)
        if not self.vector_store.is_ready():
            return ToolExecutionResult(
                text="دانش‌نامه هنوز ساخته نشده است؛ ایندکس FAISS خالی است."
            )
        results: list[RetrievalResult] = self.retriever.search(query, top_k)
        if not results:
            return ToolExecutionResult(text="هیچ نتیجه‌ای برای این جست‌وجو پیدا نشد.")
        blocks: list[str] = []
        for item in results:
            preview = item.text[: self.chunk_preview_chars]
            blocks.append(
                f"[{item.rank}] منبع: {item.source} | شباهت: {item.score:.2f}\n{preview}"
            )
        text = "\n\n".join(blocks)
        if len(text) > self.tool_result_max_chars:
            text = text[: self.tool_result_max_chars] + "\n...(متن بریده شد)"
        return ToolExecutionResult(text=text, sources=results)

    # ── ابزار ۲: آمار دانش‌نامه ──────────────────────────────
    def _run_get_knowledge_base_stats(self) -> ToolExecutionResult:
        stats = self.vector_store.stats()
        if not stats["ready"]:
            return ToolExecutionResult(text="دانش‌نامه هنوز ساخته نشده است.")
        text = (
            f"تعداد فایل‌ها: {len(stats.get('sources', []))}\n"
            f"تعداد چانک‌ها: {stats['count']}\n"
            f"ابعاد بردار: {stats['dimension']}\n"
            f"زمان ساخت ایندکس: {stats.get('created_at') or '-'}"
        )
        return ToolExecutionResult(text=text)

    # ── ابزار داخلی ─────────────────────────────────────────
    @staticmethod
    def _clamp_int(value, low: int, high: int, default: int) -> int:
        """محدودسازی مقدار عددی آرگومان به بازهٔ مجاز."""
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
        return max(low, min(high, number))
