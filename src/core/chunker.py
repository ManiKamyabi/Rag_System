"""
چانکینگ پاراگراف‌محورِ بدونِ افتِ متن
====================================
ورودی هر سند (یک ردیف CSV = یک فایل PDF) به پاراگراف‌ها شکسته
می‌شود؛ سپس پاراگراف‌ها با روش «پنجرهٔ لغزان» در چانک‌هایی از چند
پاراگراف (+ هم‌پوشانی اختیاری) گروه می‌شوند.

تضمین‌ها:
  1) هیچ متنی از بین نمی‌رود — پس از چانک، پوشش کامل پاراگراف‌ها
     و برابری محتوایی (بدون فاصله‌ها) با متن اصلی راستی‌آزمایی می‌شود.
  2) پاراگراف‌های خیلی بلند (بلندتر از PARAGRAPH_MAX_CHARS) به‌صورت
     جمله‌به‌جمله و بدون افت به قطعات کوچک‌تر شکسته می‌شوند.
  3) طول هر چانک از CHUNK_MAX_CHARS عبور نمی‌کند (مگر یک پاراگراف
     تکیِ طولانی که خودش زیر سقف پاراگراف است).
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from .exceptions import DatasetError
from .models import Chunk, Document

logger = logging.getLogger("rag.chunker")

# جداسازی پاراگراف: یک خط خالی (فقط فاصله/تب)
_BLANK_LINE_RE = re.compile(r"\n\s*\n")
# جداسازی جمله: بعد از . ! ? … ؛ : یا هر خط جدیدی
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…؛:])\s+|\n+")
# حذف همهٔ فاصله‌ها برای مقایسهٔ محتوایی
_WHITESPACE_RE = re.compile(r"\s+")


class CSVLoader:
    """خواندن دیتاست CSV با ستون‌های «نام فایل» و «محتوا».

    نام ستون‌های قابل قبول (به‌صورت case-insensitive):
      نام فایل:   filename | file_name | file | name | source | title
      محتوا:      content  | text | body | document | contents
    """

    NAME_COLUMNS = ("filename", "file_name", "file", "name", "source", "title")
    CONTENT_COLUMNS = ("content", "text", "body", "document", "contents")

    def __init__(self, name_column: str | None = None, content_column: str | None = None):
        self.name_column = name_column
        self.content_column = content_column

    # ── بارگذاری ─────────────────────────────────────────────
    def load(self, path: str | Path) -> list[Document]:
        """بارگذاری یک فایل CSV یا همهٔ فایل‌های CSV یک پوشه."""
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("*.csv"))
        else:
            files = [path]
        if not files or not files[0].exists():
            raise DatasetError(f"مسیر دیتاست پیدا نشد: {path}")
        documents: list[Document] = []
        for file in files:
            documents.extend(self._load_single(file))
        if not documents:
            raise DatasetError("هیچ سند معتبری در دیتاست پیدا نشد (همه ردیف‌ها خالی‌اند؟).")
        return documents

    def _load_single(self, file: Path) -> list[Document]:
        documents: list[Document] = []
        with file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise DatasetError(f"فایل «{file.name}» سطر هدر ندارد.")
            name_col = self._resolve_column(reader.fieldnames, self.NAME_COLUMNS,
                                            self.name_column, "نام فایل")
            content_col = self._resolve_column(reader.fieldnames, self.CONTENT_COLUMNS,
                                               self.content_column, "محتوا")
            for row_number, row in enumerate(reader, start=2):
                source = (row.get(name_col) or "").strip() or f"{file.stem}#row{row_number}"
                content = (row.get(content_col) or "").strip()
                if not content:
                    logger.warning("ردیف %d از «%s» محتوای خالی دارد و نادیده گرفته شد.",
                                   row_number, file.name)
                    continue
                documents.append(Document(source=source, content=content))
        logger.info("فایل «%s» خوانده شد: %d سند.", file.name, len(documents))
        return documents

    @staticmethod
    def _resolve_column(fieldnames, candidates, override, label) -> str:
        """پیدا کردن نام واقعی ستون بین نام‌های مجاز (یا ستون صریح کاربر)."""
        normalized = {name.strip().lower(): name for name in fieldnames if name}
        if override:
            if override.lower() in normalized:
                return normalized[override.lower()]
            raise DatasetError(f"ستون {label} با نام «{override}» در فایل پیدا نشد.")
        for candidate in candidates:
            if candidate in normalized:
                return normalized[candidate]
        raise DatasetError(
            f"ستون {label} پیدا نشد. نام‌های قابل قبول: {list(candidates)} — "
            f"ستون‌های موجود فایل: {list(fieldnames)}"
        )


class ParagraphChunker:
    """چانکر پنجره‌ای چندپاراگرافی با هم‌پوشانی؛ تضمین پوشش کامل متن."""

    def __init__(self, paragraphs_per_chunk: int = 4, overlap_paragraphs: int = 1,
                 max_chunk_chars: int = 3000, paragraph_max_chars: int = 1500):
        self.paragraphs_per_chunk = max(1, paragraphs_per_chunk)
        self.overlap_paragraphs = max(
            0, min(overlap_paragraphs, self.paragraphs_per_chunk - 1)
        )
        self.max_chunk_chars = max(200, max_chunk_chars)
        self.paragraph_max_chars = max(100, paragraph_max_chars)

    # ── مرحلهٔ ۱: شکستن به پاراگراف ──────────────────────────
    def split_paragraphs(self, text: str) -> list[str]:
        """شکستن متن به پاراگراف‌ها (با خط خالی) و ریزکردن پاراگراف‌های بلند."""
        paragraphs = [part.strip() for part in _BLANK_LINE_RE.split(text) if part.strip()]
        pieces: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.paragraph_max_chars:
                pieces.append(paragraph)
            else:
                pieces.extend(self._split_long_paragraph(paragraph))
        return pieces

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        """ریزکردن پاراگراف بلند به قطعات حداکثر paragraph_max_chars — بدون افت."""
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s]
        pieces: list[str] = []
        buffer = ""
        for sentence in sentences:
            # جملهٔ خودش از سقف بلندتر است → برش ناچارِ کاراکتری
            while len(sentence) > self.paragraph_max_chars:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                pieces.append(sentence[: self.paragraph_max_chars])
                sentence = sentence[self.paragraph_max_chars:]
            if not sentence:
                continue
            if buffer and len(buffer) + len(sentence) + 1 > self.paragraph_max_chars:
                pieces.append(buffer)
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            pieces.append(buffer)
        return pieces

    # ── مرحلهٔ ۲: پنجرهٔ لغزان روی پاراگراف‌ها ────────────────
    def chunk(self, document: Document) -> list[Chunk]:
        """تقسیم یک سند به چانک‌های چندپاراگرافی + راستی‌آزمایی پوشش."""
        pieces = self.split_paragraphs(document.content)
        if not pieces:
            return []
        chunks: list[Chunk] = []
        index = 0
        total = len(pieces)
        while index < total:
            # انتخاب پنجره: تا سقف تعداد پاراگراف و سقف کاراکتر
            window: list[int] = []
            char_count = 0
            for cursor in range(index, min(index + self.paragraphs_per_chunk, total)):
                piece = pieces[cursor]
                if window and char_count + len(piece) > self.max_chunk_chars:
                    break
                window.append(cursor)
                char_count += len(piece) + 2   # +2 برای جداساز \n\n
            text = "\n\n".join(pieces[i] for i in window)
            chunks.append(Chunk(
                text=text,
                source=document.source,
                chunk_index=len(chunks),
                paragraph_indices=tuple(window),
                char_count=len(text),
            ))
            # گام حرکت پنجره با احتساب هم‌پوشانی
            index += max(1, len(window) - self.overlap_paragraphs)
        self._verify(document, pieces, chunks)
        return chunks

    def chunk_all(self, documents: list[Document]) -> list[Chunk]:
        """چانک کردن همهٔ اسناد (شمارهٔ هر چانک نسبت به سند خودش است)."""
        all_chunks: list[Chunk] = []
        for document in documents:
            all_chunks.extend(self.chunk(document))
        return all_chunks

    # ── راستی‌آزمایی «بدون افت» ──────────────────────────────
    @staticmethod
    def _verify(document: Document, pieces: list[str], chunks: list[Chunk]) -> None:
        """پوشش همهٔ پاراگراف‌ها توسط چانک‌ها + برابری محتوایی با متن اصلی."""
        covered: set[int] = set()
        for chunk in chunks:
            covered.update(chunk.paragraph_indices)
        if covered != set(range(len(pieces))):
            raise DatasetError(
                f"خطای پوشش چانک در سند «{document.source}»: "
                f"پاراگراف‌هایی از چانک‌ها جا مانده‌اند."
            )
        original = _WHITESPACE_RE.sub("", document.content)
        rebuilt = _WHITESPACE_RE.sub("", "".join(pieces))
        if original != rebuilt:
            raise DatasetError(
                f"افت متن در شکستن پاراگراف‌های سند «{document.source}» تشخیص داده شد."
            )

    # ── آمار ─────────────────────────────────────────────────
    @staticmethod
    def build_report(documents: list[Document], chunks: list[Chunk]) -> dict:
        """گزارش آماری چانکینگ برای نمایش در اسکریپت embed."""
        original_chars = sum(len(doc.content) for doc in documents)
        chunk_chars = [chunk.char_count for chunk in chunks]
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "original_chars": original_chars,
            "chunked_chars": sum(chunk_chars),
            "avg_chunk_chars": int(sum(chunk_chars) / len(chunk_chars)) if chunk_chars else 0,
            "max_chunk_chars": max(chunk_chars) if chunk_chars else 0,
            "sources": sorted({doc.source for doc in documents}),
        }
