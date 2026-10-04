"""
تشخیص و پاک‌سازی فراخوانی ابزار در خروجی مدل
============================================
مدل Qwen3 ابزارها را با قالب hermes (بلوک‌های <tool_call> ... </tool_call>)
فراخوانی می‌کند. این ماژول:
  1) فراخوانی‌ها را از متن خام مدل استخراج می‌کند (parse_tool_calls)
  2) بلوک‌های ابزار/استدلال را از متن نهایی حذف می‌کند (strip_blocks)
  3) در حالت جریانی (stream) همین کار را token به token انجام می‌دهد
     (StreamMaskFilter) حتی وقتی تگ‌ها بین چند تکه شکسته شده‌اند.

چون خودمان خروجی را parse می‌کنیم، نیازی به فعال‌کردن ‎--jinja‎ روی
llama.cpp نیست و روی هر endpoint سازگار با OpenAI کار می‌کند.
"""

from __future__ import annotations

import json
import re

from .models import ToolCall

# بلوک کامل فراخوانی ابزار: <tool_call> {json} </tool_call>
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# بلوک استدلال داخلی مدل (در صورت وجود)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# تگ‌های یتیم (بسته‌شده بدون باز شدن)
ORPHAN_TAG_RE = re.compile(r"</?(?:think|tool_call|tool_response)>")


def _safe_json_loads(text: str) -> dict | None:
    """خواندن JSON با تحمل خطاهای رایج مدل (کوتیشن تکی، کامای اضافی)."""
    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    # تلاش برای اصلاح: کاماهای انتهایی و کوتیشن تکی
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    repaired = repaired.replace("'", '"')
    try:
        payload = json.loads(repaired)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def parse_tool_calls(text: str) -> list[ToolCall]:
    """استخراج همهٔ فراخوانی‌های ابزار از متن خام مدل."""
    calls: list[ToolCall] = []
    for match in TOOL_CALL_RE.finditer(text):
        payload = _safe_json_loads(match.group(1))
        if not payload:
            continue
        name = str(payload.get("name", "")).strip()
        if not name:
            continue
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            arguments = _safe_json_loads(arguments) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(ToolCall(name=name, arguments=arguments, raw=match.group(0)))
    return calls


def strip_blocks(text: str) -> str:
    """حذف بلوک‌های استدلال و فراخوانی ابزار از متن (برای پاسخ نهایی)."""
    cleaned = THINK_BLOCK_RE.sub("", text)
    cleaned = TOOL_CALL_RE.sub("", cleaned)
    cleaned = ORPHAN_TAG_RE.sub("", cleaned)
    return cleaned.strip()


class StreamMaskFilter:
    """فیلتر جریانی برای نمایش زندهٔ پاسخ مدل به کاربر.

    تکه‌های ورودی (token ها) را می‌گیرد و فقط متنی را عبور می‌دهد که
    بیرون از بلوک‌های استدلال/فراخوانی ابزار است — حتی وقتی خودِ
    تگ‌ها میان چند تکه شکسته شده باشند. ‎flush()‎ در پایان فراخوانی شود.
    """

    OPENINGS = (
        ("<think>", "</think>"),
        ("<tool_call>", "</tool_call>"),
    )

    def __init__(self):
        self._buffer = ""
        self._closing_tag: str | None = None
        self._done = False

    # ── رابط عمومی ──────────────────────────────────────────
    def feed(self, chunk: str) -> str:
        """دریافت یک تکه از خروجی مدل؛ بخش قابل‌نمایش آن را برمی‌گرداند."""
        if not chunk or self._done:
            return ""
        self._buffer += chunk
        output: list[str] = []
        while True:
            if self._closing_tag is not None:
                # داخل بلوک هستیم؛ دنبال تگ بسته می‌گردیم
                end = self._buffer.find(self._closing_tag)
                if end != -1:
                    self._buffer = self._buffer[end + len(self._closing_tag):]
                    self._closing_tag = None
                    continue
                # شاید ابتدای ناقصِ تگِ بسته در انتهای بافر مانده باشد
                hold = self._partial_suffix_len(self._buffer, self._closing_tag)
                self._buffer = self._buffer[len(self._buffer) - hold:] if hold else ""
                break
            position, _opening, closing = self._find_opening(self._buffer)
            if position != -1:
                output.append(self._buffer[:position])
                self._buffer = self._buffer[position + len(_opening):]
                self._closing_tag = closing
                continue
            # بلوکی باز نشد؛ اما انتهای بافر ممکن است ابتدای ناقص یک تگ باشد
            hold = self._partial_opening_len(self._buffer)
            if hold:
                output.append(self._buffer[:len(self._buffer) - hold])
                self._buffer = self._buffer[len(self._buffer) - hold:]
            else:
                output.append(self._buffer)
                self._buffer = ""
            break
        return "".join(output)

    def flush(self) -> str:
        """پایان جریان؛ متن سالمِ باقیمانده را برمی‌گرداند (پیشوند ناقص تگ دور ریخته می‌شود)."""
        tail = ""
        if self._closing_tag is None and not self._is_opening_prefix(self._buffer):
            tail = self._buffer
        self._buffer = ""
        self._done = True
        return tail

    # ── ابزارهای داخلی ──────────────────────────────────────
    def _find_opening(self, text: str):
        """نزدیک‌ترین تگِ شروعِ بلوک را پیدا می‌کند؛ (موقعیت، تگ شروع، تگ بسته)."""
        best = (-1, "", "")
        for opening, closing in self.OPENINGS:
            position = text.find(opening)
            if position != -1 and (best[0] == -1 or position < best[0]):
                best = (position, opening, closing)
        return best

    def _partial_opening_len(self, text: str) -> int:
        """طول پیشوندِ ناقصِ هر یک از تگ‌های شروع در انتهای متن (بیشترین)."""
        return max(
            (self._partial_suffix_len(text, opening) for opening, _ in self.OPENINGS),
            default=0,
        )

    def _is_opening_prefix(self, text: str) -> bool:
        """آیا متن دقیقاً پیشوندِ ناقصِ یکی از تگ‌های شروع است؟"""
        return any(
            opening.startswith(text) and 0 < len(text) < len(opening)
            for opening, _ in self.OPENINGS
        )

    @staticmethod
    def _partial_suffix_len(text: str, tag: str) -> int:
        """اگر انتهای متن، ابتدای ناقصِ تگ باشد، طول آن را برمی‌گرداند."""
        for length in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0