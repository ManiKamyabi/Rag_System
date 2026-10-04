"""
ساخت پرامپت‌ها (فارسی)
======================
- پرامپت سیستمی: معرفی نقش، قواعد پاسخ‌دهی و آموزش قالب فراخوانی ابزار
- قالب پیام کاربر: بافتِ بازیابی‌شده + سوال
- قالب پاسخ ابزار به مدل (همان چیزی که مدل پس از فراخوانی می‌بیند)
"""

from __future__ import annotations

import json

from .models import RetrievalResult
from .tools import TOOL_DEFINITIONS


def render_tool_instructions() -> str:
    """معرفی ابزارها به مدل به سبک hermes (سازگار با Qwen3)."""
    lines: list[str] = [
        "# ابزارها (Tools)",
        "",
        "می‌توانی برای کسب اطلاعات بیشتر از ابزارهای زیر استفاده کنی. برای فراخوانی",
        "دقیقاً از این قالب استفاده کن و بلافاصله پس از آن نوشتن را متوقف کن تا",
        "نتیجهٔ ابزار را دریافت کنی:",
        "",
        "<tool_call>",
        '{"name": "نام_ابزار", "arguments": {"پارامتر": "مقدار"}}',
        "</tool_call>",
        "",
        "ابزارهای موجود:",
        "",
    ]
    for tool in TOOL_DEFINITIONS:
        function = tool["function"]
        params = function.get("parameters", {}).get("properties", {})
        param_names = ", ".join(params) if params else "بدون پارامتر"
        lines.append(f"- {function['name']}({param_names}): {function['description']}")
    lines += [
        "",
        "قواعد استفاده از ابزار:",
        "- فقط وقتی ابزار را صدا بزن که اطلاعات موجود کافی نیست.",
        "- در هر نوبت حداکثر یک فراخوانی؛ سپس منتظر نتیجه بمان.",
        "- هرگز خروجی ساختگی برای ابزار نساز؛ نتیجه در پیام بعدی به تو داده می‌شود.",
    ]
    return "\n".join(lines)


def build_system_prompt() -> str:
    """پرامپت سیستمی کامل مدل."""
    rules = [
        "شما «دستیار دانش‌نامه» هستید؛ یک دستیار پرسش‌وپاسخ هوشمند که به مجموعهٔ",
        "اسناد محلی (دانش‌نامه) دسترسی دارد.",
        "",
        "## قواعد پاسخ‌دهی",
        "1. فقط بر اساس اطلاعات موجود در بخش بافت (context) پیام کاربر یا نتایج",
        "   ابزارها پاسخ دهید؛ از دانش عمومی فقط برای تکمیل جمله‌بندی استفاده کنید.",
        "2. اگر پاسخ در منابع موجود نیست، صادقانه اعلام کنید؛ اطلاعات از خودتان",
        "   نسازید.",
        "3. در صورت استفاده از منابع، به نام فایل مرجع (مثلاً report.pdf) اشاره کنید.",
        "4. به زبان کاربر پاسخ دهید (پیش‌فرض: فارسی).",
        "5. پاسخ‌ها را مرتب، با پاراگراف‌بندی مناسب و در صورت نیاز فهرست‌وار بنویسید.",
        "6. در پاسخ نهایی از تگ‌های XML یا بلوک‌های فراخوانی ابزار استفاده نکنید؛",
        "   فقط متن خوانا برای کاربر بنویسید.",
    ]
    return "\n".join(rules) + "\n\n" + render_tool_instructions()


def build_context_block(results: list[RetrievalResult], max_chars: int) -> str:
    """ساخت بخش بافت از نتایج بازیابی با رعایت سقف کاراکتر."""
    if not results:
        return ""
    blocks: list[str] = []
    used = 0
    for item in results:
        header = f"[منبع {item.rank} | فایل: {item.source} | شباهت: {item.score:.2f}]"
        block = f"{header}\n{item.text}"
        if used + len(block) > max_chars and blocks:
            break   # سقف پر شد؛ نتایج ضعیف‌تر (رتبه پایین‌تر) حذف می‌شوند
        blocks.append(block)
        used += len(block)
    if not blocks:
        # حتی یک چانک کامل جا نشد → برش اولین نتیجه
        first = results[0]
        blocks.append(f"[منبع 1 | فایل: {first.source}]\n{first.text[:max_chars]}")
    return "<context>\n" + "\n\n".join(blocks) + "\n</context>"


def build_user_prompt(question: str, context_block: str) -> str:
    """قالب پیام کاربر: بافت بازیابی‌شده + سوال."""
    parts: list[str] = []
    if context_block:
        parts.append(context_block)
        parts.append(
            "با توجه به متن‌های بالا (در صورت مرتبط بودن) به سوال کاربر پاسخ دهید. "
            "اگر پاسخ در آن‌ها نیست و نیاز به اطلاعات بیشتری دارید، از ابزار "
            "search_knowledge_base استفاده کنید."
        )
    else:
        parts.append(
            "بافت مرتبطی برای این سوال پیدا نشد. در صورت نیاز از ابزار "
            "search_knowledge_base استفاده کنید."
        )
    parts.append(f"\nسوال کاربر:\n{question}")
    return "\n\n".join(parts)


def build_tool_response_message(tool_name: str, content: str) -> str:
    """قالب برگرداندن نتیجهٔ ابزار به مدل (نقش user با پوشش تگ پاسخ ابزار)."""
    payload = json.dumps({"name": tool_name, "content": content}, ensure_ascii=False)
    return f"<tool_response>\n{payload}\n</tool_response>"
