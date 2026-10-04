"""ساختارهای دادهٔ مشترک سیستم (dataclass ها)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """یک سند ورودی؛ معادل یک ردیف از دیتاست CSV (یک فایل PDF)."""

    source: str        # نام فایل/منبع (مقدار ستون filename)
    content: str       # متن کامل فایل (مقدار ستون content)


@dataclass
class Chunk:
    """یک قطعهٔ چانک‌شدهٔ آمادهٔ امبدینگ."""

    text: str                      # متن چانک
    source: str                    # منبع (نام فایل)
    chunk_index: int               # شمارهٔ چانک در همان سند
    paragraph_indices: tuple = ()  # شمارهٔ پاراگراف‌های پوشش‌داده‌شده
    char_count: int = 0            # طول متن چانک


@dataclass
class RetrievalResult:
    """نتیجهٔ یک جست‌وجوی معنایی (یک چانک بازیابی‌شده)."""

    text: str
    score: float        # شباهت کسینوسی (بالاتر = مرتبط‌تر)
    source: str
    chunk_index: int
    rank: int = 0       # رتبه در نتایج (از 1)


@dataclass
class ToolCall:
    """یک فراخوانی ابزار که مدل در خروجی خود اعلام کرده است."""

    name: str
    arguments: dict
    raw: str = ""       # بلوک خام متن فراخوانی


@dataclass
class ToolExecutionResult:
    """خروجی اجرای یک ابزار توسط ToolRegistry."""

    text: str                                   # متن نتیجه برای مدل
    sources: list = field(default_factory=list)  # منابع بازیابی‌شده (برای UI)


@dataclass
class PipelineResult:
    """نتیجهٔ نهایی یک اجرای کامل خط لولهٔ RAG."""

    answer: str = ""
    sources: list = field(default_factory=list)    # list[RetrievalResult]
    tool_calls: list = field(default_factory=list) # list[ToolCall]
    warnings: list = field(default_factory=list)   # list[str]
    error: str | None = None
    finished: bool = False

    def add_warning(self, message: str) -> None:
        """افزودن هشدار بدون تکرار."""
        if message not in self.warnings:
            self.warnings.append(message)
