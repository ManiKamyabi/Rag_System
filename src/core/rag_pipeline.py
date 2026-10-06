"""
خط لولهٔ اصلی RAG — نسخهٔ «ابزار-محور» (Tool-Call-First)
========================================================
جریان جدید پاسخ‌دهی، مطابق نیاز پروژه:

    پیام کاربر
        → ارسال مستقیم به مدل (بدون هیچ امبدینگ یا جست‌وجوی خودکار!)
        → خودِ مدل تصمیم می‌گیرد که آیا به دانش‌نامه نیاز دارد یا نه
        → در صورت نیاز: فراخوانی ابزار search_knowledge_base (قالب hermes)
           ← در همین مرحله پرس‌وجو امبد می‌شود و FAISS جست‌وجو می‌شود
        → نتیجهٔ ابزار به مدل برمی‌گردد و تولید ادامه می‌یابد
        → (در صورت نیاز) دوباره فراخوانی ابزار تا سقف max_tool_calls
        → پاسخ نهایی + منابع استفاده‌شده

نتیجهٔ مهم: اگر به مدل «سلام» بگویید، هیچ متنی به سرویس امبدینگ
فرستاده نمی‌شود؛ چون مدل ابزار را فراخوانی نمی‌کند. امبدینگ فقط و
فقط داخل اجرای ابزار و با تصمیم خودِ مدل اتفاق می‌افتد.

کلاس ‎RAGPipeline‎ اجزا را هماهنگ می‌کند و برای هر پرس‌وجو یک
‎PipelineRun‎ می‌سازد که هم قابل استریم است و هم نتیجهٔ کامل
(پاسخ، منابع، فراخوانی‌های ابزار، هشدارها) را در ‎self.result‎
نگه می‌دارد.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator

from ..config import Settings
from .api_client import LLMClient
from .models import PipelineResult
from .prompts import build_system_prompt, build_tool_response_message
from .tool_parsing import StreamMaskFilter, parse_tool_calls, strip_blocks
from .tools import ToolRegistry

logger = logging.getLogger("rag.pipeline")

# ── رویدادهایی که UI می‌تواند روی آن‌ها واکنش نشان دهد ──────────────
EVENT_TOOL_CALL = "tool_call"       # مدل ابزاری را صدا زد (name, arguments)
EVENT_TOOL_RESULT = "tool_result"   # نتیجهٔ ابزار آماده شد (name, summary)
EVENT_LLM_ROUND = "llm_round"       # دور جدید تولید (round, forced_final)
EVENT_THINKING = "thinking"         # مدل وارد/خارج بلوک تفکر شد (active)
EVENT_WARNING = "warning"           # هشدار غیرمهلک (message)


class PipelineRun:
    """یک اجرای خاص از خط لوله برای یک سوال.

    استفاده:
        run = pipeline.create_run(question, history)
        for piece in run.stream():   # نمایش تدریجی
            ...
        print(run.result.answer)     # نتیجهٔ کامل
    """

    def __init__(self, pipeline: "RAGPipeline", question: str, history: list[dict],
                 temperature: float | None = None):
        self.pipeline = pipeline
        self.question = question
        self.history = history
        self.temperature = (pipeline.settings.temperature
                            if temperature is None else temperature)
        self.result = PipelineResult()
        self.on_event: Callable[..., None] | None = None

    def emit(self, event: str, **data) -> None:
        """ارسال رویداد به UI (در صورت ثبت callback)؛ خطای UI نادیده گرفته می‌شود."""
        if self.on_event is None:
            return
        try:
            self.on_event(event, **data)
        except Exception:  # noqa: BLE001
            logger.debug("خطای callback رویداد نادیده گرفته شد.", exc_info=True)

    # ── اجرای اصلی ──────────────────────────────────────────
    def stream(self) -> Iterator[str]:
        """مولد متن پاسخ؛ هم‌زمان result را کامل می‌کند.

        نکته: برخلاف نسخهٔ قبلی، هیچ بازیابیِ خودکاری در ابتدا انجام
        نمی‌شود. پیام کاربر خام به مدل می‌رود و فقط اگر مدل ابزار را
        فراخوانی کند، جست‌وجوی معنایی (امبد + FAISS) اجرا می‌شود.
        """
        settings = self.pipeline.settings

        messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": self.question})

        # حلقهٔ مدل + ابزار (حداکثر max_tool_calls بار فراخوانی)
        tool_names = tuple(self.pipeline.tools.tool_names())
        display_parts: list[str] = []
        total_rounds = settings.max_tool_calls + 1
        try:
            for round_number in range(1, total_rounds + 1):
                forced_final = round_number == total_rounds
                self.emit(EVENT_LLM_ROUND, round=round_number, forced_final=forced_final)
                if forced_final:
                    messages.append({
                        "role": "system",
                        "content": "دیگر فرصتی برای فراخوانی ابزار نیست؛ همین حالا "
                                   "پاسخ نهایی را بر اساس اطلاعات موجود بنویس.",
                    })

                # دور تولید مدل — جریانی برای کاربر، خام برای تحلیل ابزار
                mask = StreamMaskFilter(tool_names=tool_names)
                collected: list[str] = []
                visible_text = ""
                was_inside_block = False
                for token in self.pipeline.llm.chat_stream(
                    messages, temperature=self.temperature,
                    max_tokens=settings.max_tokens,
                ):
                    collected.append(token)
                    piece = mask.feed(token)
                    # گزارش ورود/خروج به بلوک تفکر برای نمایش وضعیت در UI
                    # (بلافاصله بعد از feed تا گذارها همان لحظه منتشر شوند)
                    inside_block = mask.inside_block
                    if inside_block != was_inside_block:
                        self.emit(EVENT_THINKING, active=inside_block)
                        was_inside_block = inside_block
                    if piece:
                        visible_text += piece
                        yield piece
                if was_inside_block:
                    # اگر جریان داخل بلوک تمام شد، وضعیت را ببند
                    self.emit(EVENT_THINKING, active=False)
                tail = mask.flush()
                if tail:
                    visible_text += tail
                    yield tail
                raw_text = "".join(collected)

                # آیا مدل ابزاری فراخوانی کرده است؟
                # (هم بلوک تگ‌دار و هم JSON خالیِ بدون تگ پشتیبانی می‌شود)
                calls = ([] if forced_final
                         else parse_tool_calls(raw_text, tool_names=tool_names))
                if not calls:
                    self.result.answer = self._compose_answer(
                        display_parts, visible_text, raw_text, tool_names
                    )
                    self.result.finished = True
                    return

                # اجرای ابزارها و برگرداندن نتیجه به مدل
                messages.append({"role": "assistant", "content": raw_text})
                for call in calls:
                    self.result.tool_calls.append(call)
                    self.emit(EVENT_TOOL_CALL, name=call.name, arguments=call.arguments)
                    execution = self.pipeline.tools.execute(call)
                    self.result.sources.extend(execution.sources)
                    summary = execution.text[:160].replace("\n", " ")
                    self.emit(EVENT_TOOL_RESULT, name=call.name, summary=summary)
                    messages.append({
                        "role": "user",
                        "content": build_tool_response_message(call.name, execution.text),
                    })
                if visible_text.strip():
                    display_parts.append(visible_text.strip())

            # عملاً به اینجا نمی‌رسیم؛ برای اطمینان:
            self.result.answer = self._compose_answer(display_parts, "", "", tool_names)
            self.result.finished = True

        except Exception as exc:  # noqa: BLE001
            logger.exception("خطا در تولید پاسخ.")
            self.result.error = f"خطا در تولید پاسخ: {exc}"
            self.result.finished = False
            yield ("\n\n(اتصال به مدل برقرار نشد یا در تولید پاسخ خطایی رخ داد؛ "
                   "سلامت سرویس‌ها را با اسکریپت scripts/check_services.py بررسی کنید.)")

    # ── کمکی‌ها ─────────────────────────────────────────────
    @staticmethod
    def _compose_answer(display_parts: list[str], visible_text: str,
                        raw_text: str, tool_names=()) -> str:
        """ترکیب متن‌های قابل‌نمایش دورهای مختلف در یک پاسخ نهایی."""
        parts = [part for part in display_parts if part and part.strip()]
        if visible_text and visible_text.strip():
            parts.append(visible_text.strip())
        answer = "\n\n".join(parts).strip()
        if not answer:
            answer = strip_blocks(raw_text, tool_names).strip()
        return answer


class RAGPipeline:
    """هماهنگ‌کنندهٔ اجزای RAG (مدل، ابزارها، تنظیمات).

    نکتهٔ معماری: بازیاب (Retriever) دیگر مستقیماً در خط لوله استفاده
    نمی‌شود؛ جست‌وجوی معنایی فقط از طریق ‎ToolRegistry‎ و با فراخوانی
    خودِ مدل اجرا می‌شود.
    """

    def __init__(self, llm: LLMClient, tools: ToolRegistry, settings: Settings):
        self.llm = llm
        self.tools = tools
        self.settings = settings

    def create_run(self, question: str, history: list[dict] | None = None,
                   temperature: float | None = None) -> PipelineRun:
        """ساخت یک اجرای خط لوله برای سوال داده‌شده."""
        return PipelineRun(self, question, history or [], temperature)
