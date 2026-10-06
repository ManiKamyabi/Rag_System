"""
تشخیص و پاک‌سازی فراخوانی ابزار در خروجی مدل — نسخهٔ مقاوم (Robust)
====================================================================
مدل Qwen3 فراخوانی ابزار را معمولاً با قالب hermes (بلوک‌های <tool_call>…</tool_call>)
انجام می‌دهد؛ اما در عمل گاهی همین قالب را «بدون تگ» و به‌صورت یک JSON خالی
در میان متن فارسی خودش می‌نویسد (مثلاً بعد از یک جملهٔ مقدمه):

    برای توضیح بیشتر لطفاً صبر کنید...
    {"name": "search_knowledge_base", "arguments": {"query": "MQTT", "top_k": 3}}

این ماژول هر دو حالت (و حالت‌های مرزی دیگر) را پوشش می‌دهد:
  1) parse_tool_calls — استخراج فراخوانی‌ها از متن خام مدل:
       • بلوک‌های تگ‌دار <tool_call>{...}</tool_call>
       • JSONهای خالیِ شامل «name» یک ابزار شناخته‌شده (+ arguments)
         که با پیمایش متوازن آکولاد پیدا می‌شوند (حتی وسط متن فارسی)
       • تگ بسته‌نشده در انتهای خروجی (وقتی مدل وسط فراخوانی قطع می‌شود)
  2) strip_blocks — حذف بلوک‌های استدلال/فراخوانی از متن نهایی
  3) StreamMaskFilter — همان کار در حالت جریانی (stream) token به token،
     به‌طوری که JSON فراخوانی هرگز به صفحهٔ کاربر «نچسبد»؛ اگر متنی
     شبیه JSON بود ولی فراخوانی ابزار معتبر نبود، بدون آسیب نمایش داده می‌شود.

چون خودمان خروجی را parse می‌کنیم، نیازی به فعال‌کردن --jinja روی
llama.cpp نیست و روی هر endpoint سازگار با OpenAI کار می‌کند.
"""

from __future__ import annotations

import json
import re

from .models import ToolCall

# ── تگ‌ها ────────────────────────────────────────────────────────────
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

# بلوک کامل فراخوانی ابزار (تگ‌دار):  <tool_call> {json} </tool_call>
TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
# بلوک استدلال داخلی مدل (در صورت وجود)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# تگ‌های یتیم (بسته‌شده بدون باز شدن، یا باز مانده در انتها)
ORPHAN_TAG_RE = re.compile(r"</?(?:think|tool_call|tool_response)>")
# شروع یک JSON احتمالی در جریان:  {  و سپس اختیاری فاصله و کوتیشن
JSON_START_RE = re.compile(r'\{\s*"')
# سقف بافرِ حالت JSON (ضد گیرکردن روی خروجی ناقص/خراب)
MAX_JSON_BUFFER = 8000


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


def _extract_arguments(payload: dict) -> dict:
    """استخراج آرگومان‌های فراخوانی از payload (arguments یا parameters)."""
    arguments = payload.get("arguments", payload.get("parameters"))
    if isinstance(arguments, str):
        arguments = _safe_json_loads(arguments)
    return arguments if isinstance(arguments, dict) else {}


def _is_tool_call_payload(payload: dict | None, tool_names) -> bool:
    """آیا این dict یک فراخوانی ابزار معتبر است؟
    اگر tool_names داده شود، نام ابزار باید یکی از آن‌ها باشد؛ این شرط
    مانع می‌شود JSONهای عادیِ پاسخ مدل (مثلاً یک مثال JSON) قورت داده شوند.
    """
    if not payload:
        return False
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return False
    if tool_names and name.strip() not in tool_names:
        return False
    if "arguments" not in payload and "parameters" not in payload:
        # فراخوانی بدون آرگومان (مثل get_knowledge_base_stats) هم مجاز است
        return True
    arguments = payload.get("arguments", payload.get("parameters"))
    return isinstance(arguments, (dict, str))


def _iter_balanced_json_objects(text: str):
    """پیمایش متن و بازدهٔ (start, end) هر شیء JSON متوازن.
    آکولادهای داخل رشته‌ها نادیده گرفته می‌شوند؛ برای متن‌های چت کاملاً سریع است.
    """
    depth = 0
    in_string = False
    escaped = False
    start = -1
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    yield start, index + 1
                    start = -1


def parse_tool_calls(text: str, tool_names=None) -> list[ToolCall]:
    """استخراج همهٔ فراخوانی‌های ابزار از متن خام مدل (تگ‌دار یا بدون تگ)."""
    calls: list[ToolCall] = []

    # ۱) بلوک‌های تگ‌دار <tool_call>{...}</tool_call>
    for match in TOOL_CALL_BLOCK_RE.finditer(text):
        payload = _safe_json_loads(match.group(1))
        if payload is None or not _is_tool_call_payload(payload, tool_names):
            continue
        calls.append(ToolCall(name=str(payload["name"]).strip(),
                              arguments=_extract_arguments(payload),
                              raw=match.group(0)))

    # ۲) JSONهای بدون تگ — روی متنی که داخل بلوک تگ‌دار/تفکر نیست
    remainder = TOOL_CALL_BLOCK_RE.sub("", text)
    remainder = THINK_BLOCK_RE.sub("", remainder)
    remainder = remainder.replace(TOOL_CALL_OPEN, "").replace(TOOL_CALL_CLOSE, "")
    remainder = remainder.replace("<think>", "").replace("</think>", "")
    remainder = ORPHAN_TAG_RE.sub("", remainder)
    for match in _iter_balanced_json_objects(remainder):
        obj_text = remainder[match[0]:match[1]]
        payload = _safe_json_loads(obj_text)
        if payload is None or not _is_tool_call_payload(payload, tool_names):
            continue
        calls.append(ToolCall(name=str(payload["name"]).strip(),
                              arguments=_extract_arguments(payload),
                              raw=obj_text))
    return calls


def strip_blocks(text: str, tool_names=None) -> str:
    """حذف بلوک‌های استدلال و فراخوانی ابزار از متن (برای پاسخ نهایی)."""
    cleaned = THINK_BLOCK_RE.sub("", text)
    cleaned = TOOL_CALL_BLOCK_RE.sub("", cleaned)

    # حذف JSONهای فراخوانی بدون تگ
    keep, cursor = [], 0
    for start, end in _iter_balanced_json_objects(cleaned):
        obj_text = cleaned[start:end]
        payload = _safe_json_loads(obj_text)
        if payload is not None and _is_tool_call_payload(payload, tool_names):
            keep.append(cleaned[cursor:start])
            cursor = end
    keep.append(cleaned[cursor:])
    cleaned = "".join(keep)

    cleaned = ORPHAN_TAG_RE.sub("", cleaned)
    return cleaned.strip()


class StreamMaskFilter:
    """فیلتر جریانی برای نمایش زندهٔ پاسخ مدل به کاربر.

    تکه‌های ورودی (token ها) را می‌گیرد و فقط متنی را عبور می‌دهد که
    بیرون از بلوک‌های استدلال/فراخوانی ابزار است:
      • بلوک‌های <think>…</think> و <tool_call>…</tool_call> حتی وقتی
        خودِ تگ‌ها میان چند تکه شکسته شده‌اند
      • JSONهای خالیِ فراخوانی ابزار (بدون تگ) با رهگیری متوازن آکولاد؛
        اگر معتبر نبودند عیناً به خروجی برمی‌گردند
    ‎flush()‎ در پایان فراخوانی شود.
    """

    OPENINGS = (
        ("<think>", "</think>"),
        (TOOL_CALL_OPEN, TOOL_CALL_CLOSE),
    )

    def __init__(self, tool_names=None):
        self._buffer = ""
        self._closing_tag: str | None = None
        self._done = False
        self._tool_names = {name.strip() for name in (tool_names or ())}

        # وضعیت رهگیری JSON (حالت JSON)
        self._json_depth = 0
        self._json_in_string = False
        self._json_escaped = False
        self._json_active = False

    # ── رابط عمومی ──────────────────────────────────────────
    @property
    def inside_block(self) -> bool:
        """آیا جریان فعلاً داخل یک ناحیهٔ مخفی (تفکر/فراخوانی ابزار) است؟

        UI از این وضعیت برای نمایش «مدل در حال فکر کردن است…» استفاده
        می‌کند؛ مقایسهٔ مقدار قبل و بعد از هر ‎feed()‎ ورود/خروج را نشان
        می‌دهد.
        """
        return self._closing_tag is not None or self._json_active

    def feed(self, chunk: str) -> str:
        """دریافت یک تکه از خروجی مدل؛ بخش قابل‌نمایش آن را برمی‌گرداند."""
        if not chunk or self._done:
            return ""
        self._buffer += chunk
        output: list[str] = []
        while True:
            if self._closing_tag is not None:
                # داخل بلوک تگ‌دار هستیم؛ دنبال تگ بسته می‌گردیم
                end = self._buffer.find(self._closing_tag)
                if end != -1:
                    self._buffer = self._buffer[end + len(self._closing_tag):]
                    self._closing_tag = None
                    continue
                # شاید ابتدای ناقصِ تگِ بسته در انتهای بافر مانده باشد
                hold = self._partial_suffix_len(self._buffer, self._closing_tag)
                self._buffer = self._buffer[len(self._buffer) - hold:] if hold else ""
                break
            if self._json_active:
                output.extend(self._drain_json_mode())
                break
            position, _opening, closing = self._find_opening(self._buffer)
            json_start = self._find_json_start(self._buffer)
            if position != -1 and (json_start == -1 or position < json_start):
                output.append(self._buffer[:position])
                self._buffer = self._buffer[position + len(_opening):]
                self._closing_tag = closing
                continue
            if json_start != -1:
                # ورود به حالت JSON: متن قبل از آن منتشر، خودش نگه داشته می‌شود
                output.append(self._buffer[:json_start])
                self._buffer = self._buffer[json_start:]
                self._enter_json_mode()
                continue
            # بلوکی باز نشد؛ اما انتهای بافر ممکن است ابتدای ناقص یک تگ یا
            # شروعِ یک JSON (آکولاد تنهای انتهای بافر) باشد
            hold = self._partial_opening_len(self._buffer)
            tail_hold = self._trailing_brace_len(self._buffer)
            hold = max(hold, tail_hold)
            if hold:
                output.append(self._buffer[:len(self._buffer) - hold])
                self._buffer = self._buffer[len(self._buffer) - hold:]
            else:
                output.append(self._buffer)
                self._buffer = ""
            break
        return "".join(output)

    def flush(self) -> str:
        """پایان جریان؛ متن سالمِ باقیمانده را برمی‌گرداند.

        اگر جریان داخل یک بلوک مخفی تمام شده باشد، محتوای آن دور ریخته
        می‌شود (مدل وسط فراخوانی/تفکر قطع شده است).
        """
        tail = ""
        if self._json_active:
            # JSON ناتمام: اگر شبیه فراخوانی ابزار بود دور ریخته می‌شود؛
            # در غیر این صورت به‌عنوان متن نمایش داده می‌شود
            candidate = self._buffer
            payload = _safe_json_loads(candidate)
            if not _is_tool_call_payload(payload, self._tool_names or None):
                tail = candidate
            self._reset_json_state()
        elif self._closing_tag is None and not self._is_opening_prefix(self._buffer):
            tail = self._buffer
        self._buffer = ""
        self._done = True
        return tail

    # ── حالت JSON ───────────────────────────────────────────
    def _enter_json_mode(self) -> None:
        self._json_active = True
        self._json_depth = 0
        self._json_in_string = False
        self._json_escaped = False
        self._json_scanned = 0

    def _reset_json_state(self) -> None:
        self._json_active = False
        self._json_depth = 0
        self._json_in_string = False
        self._json_escaped = False
        self._json_scanned = 0

    def _drain_json_mode(self) -> list[str]:
        """پردازش بافر در حالت JSON؛ خروجی قابل‌نمایش (احتمالاً خالی) را می‌دهد."""
        output: list[str] = []
        index = self._json_scanned
        completed_end = -1
        while index < len(self._buffer):
            char = self._buffer[index]
            index += 1
            if self._json_escaped:
                self._json_escaped = False
                continue
            if self._json_in_string:
                if char == "\\":
                    self._json_escaped = True
                elif char == '"':
                    self._json_in_string = False
                continue
            if char == '"':
                self._json_in_string = True
            elif char == "{":
                self._json_depth += 1
            elif char == "}":
                if self._json_depth > 0:
                    self._json_depth -= 1
                    if self._json_depth == 0:
                        completed_end = index
                        break
        self._json_scanned = index
        if completed_end != -1:
            candidate = self._buffer[:completed_end]
            rest = self._buffer[completed_end:]
            self._reset_json_state()
            payload = _safe_json_loads(candidate)
            if _is_tool_call_payload(payload, self._tool_names or None):
                # فراخوانی ابزار — هرگز نمایش داده نمی‌شود
                self._buffer = rest.lstrip(" \t\r\n")
            else:
                # JSON عادیِ پاسخ مدل — عیناً نمایش داده می‌شود
                output.append(candidate)
                self._buffer = rest
        elif len(self._buffer) > MAX_JSON_BUFFER:
            # گیر روی خروجی ناقص/خراب — همه‌چیز به‌عنوان متن آزاد می‌شود
            output.append(self._buffer)
            self._buffer = ""
            self._reset_json_state()
        # در غیر این صورت بافنگه می‌دارد تا تکه‌های بعدی JSON را کامل کنند
        return output

    @staticmethod
    def _find_json_start(text: str) -> int:
        """اولین جایی که یک JSON احتمالی شروع می‌شود ({ به‌دنبال ")؛ -1 اگر نباشد."""
        match = JSON_START_RE.search(text)
        return match.start() if match else -1

    @staticmethod
    def _trailing_brace_len(text: str) -> int:
        """اگر انتهای متن فقط یک { (با فاصله‌های اختیاری) باشد، طول آن را برمی‌گرداند.
        این آکولاد ممکن است ابتدای یک JSON فراخوانی ابزار باشد و نباید فلش شود.
        """
        match = re.search(r"\{\s*$", text)
        return (len(text) - match.start()) if match else 0

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
