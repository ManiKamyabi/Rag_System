"""
امبد کردن دیتاست و ساخت ایندکس FAISS
=====================================
دیتاست (فایل CSV با ستون‌های filename و content) را می‌خواند،
پاراگراف‌به‌پاراگراف و «بدون افت متن» چانک می‌کند، با سرویس
امبدینگ (bge-m3) بردارها را می‌سازد و ایندکس FAISS را ذخیره می‌کند.

استفاده از ریشهٔ پروژه:

    python scripts/embed_dataset.py --data /مسیر/به/data.csv
    python scripts/embed_dataset.py --data ./data              # پوشهٔ CSV ها
    python scripts/embed_dataset.py --data data.csv --preview  # فقط گزارش چانکینگ

گزینه‌ها:
    --data            مسیر فایل CSV یا پوشه‌ای از فایل‌های CSV (الزامی)
    --preview         فقط چانک کن و گزارش بده؛ embed و ذخیره انجام نشود
    --index-dir       پوشهٔ خروجی ایندکس (پیش‌فرض از ‎.env‎ یا storage/index)
    --paragraphs      تعداد پاراگراف در هر چانک (پیش‌فرض 4)
    --overlap         هم‌پوشانی پاراگرافی بین چانک‌های مجاور (پیش‌فرض 1)
    --max-chars       سقف کاراکترِ هر چانک (پیش‌فرض 3000)
    --batch-size      تعداد متن در هر درخواست امبدینگ (پیش‌فرض 16)
    --name-column     نام صریح ستون «نام فایل» (اختیاری)
    --content-column  نام صریح ستون «محتوا» (اختیاری)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.core.api_client import EmbeddingClient
from src.core.chunker import CSVLoader, ParagraphChunker
from src.core.embedder import EmbeddingService
from src.core.exceptions import RAGSystemError
from src.core.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("embed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="امبد کردن دیتاست CSV و ساخت ایندکس FAISS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True,
                        help="مسیر فایل CSV یا پوشه‌ای از فایل‌های CSV")
    parser.add_argument("--preview", action="store_true",
                        help="فقط چانکینگ و گزارش؛ بدون embed و ذخیره")
    parser.add_argument("--index-dir", default=None,
                        help="پوشهٔ خروجی ایندکس (پیش‌فرض از تنظیمات)")
    parser.add_argument("--paragraphs", type=int, default=None,
                        help="تعداد پاراگراف در هر چانک")
    parser.add_argument("--overlap", type=int, default=None,
                        help="هم‌پوشانی پاراگرافی بین چانک‌ها")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="سقف کاراکتر هر چانک")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="اندازهٔ دستهٔ امبدینگ")
    parser.add_argument("--name-column", default=None,
                        help="نام صریح ستون نام فایل")
    parser.add_argument("--content-column", default=None,
                        help="نام صریح ستون محتوا")
    return parser.parse_args()


def print_report(report: dict) -> None:
    """نمایش گزارش چانکینگ در کنسول."""
    line = "─" * 52
    print(f"\n{line}")
    print("گزارش چانکینگ (پاراگراف‌محور، بدون افت متن)")
    print(line)
    print(f"تعداد اسناد (فایل‌ها)      : {report['documents']}")
    print(f"تعداد چانک‌ها              : {report['chunks']}")
    print(f"مجموع متن اصلی             : {report['original_chars']:,} کاراکتر")
    print(f"مجموع چانک‌ها (با هم‌پوشانی): {report['chunked_chars']:,} کاراکتر")
    print(f"میانگین طول چانک           : {report['avg_chunk_chars']:,} کاراکتر")
    print(f"بلندترین چانک              : {report['max_chunk_chars']:,} کاراکتر")
    print(f"پوشش کامل متن              : تایید شد (بدون از دست رفتن هیچ متنی)")
    print(f"فایل‌ها                     : {', '.join(report['sources'][:5])}"
          + (" و…" if len(report["sources"]) > 5 else ""))
    print(line + "\n")


def main() -> int:
    args = parse_args()
    settings = Settings.load()

    chunker = ParagraphChunker(
        paragraphs_per_chunk=(args.paragraphs if args.paragraphs is not None
                              else settings.chunk_paragraphs),
        overlap_paragraphs=(args.overlap if args.overlap is not None
                            else settings.chunk_overlap_paragraphs),
        max_chunk_chars=(args.max_chars if args.max_chars is not None
                         else settings.chunk_max_chars),
        paragraph_max_chars=settings.paragraph_max_chars,
    )
    loader = CSVLoader(name_column=args.name_column,
                       content_column=args.content_column)

    # ۱) خواندن دیتاست
    logger.info("[1/4] خواندن دیتاست از «%s» …", args.data)
    try:
        documents = loader.load(args.data)
    except RAGSystemError as exc:
        logger.error("خطا در دیتاست: %s", exc)
        return 1

    # ۲) چانکینگ پاراگراف‌محور + راستی‌آزمایی پوشش کامل
    logger.info("[2/4] چانکینگ پاراگراف‌محور …")
    try:
        chunks = chunker.chunk_all(documents)
    except RAGSystemError as exc:
        logger.error("خطا در چانکینگ: %s", exc)
        return 1
    report = ParagraphChunker.build_report(documents, chunks)
    print_report(report)

    if args.preview:
        logger.info("حالت پیش‌نمایش: امبدینگ انجام نشد.")
        return 0

    # ۳) امبدینگ دسته‌ای با سرویس bge-m3
    batch_size = args.batch_size if args.batch_size is not None else settings.embed_batch_size
    embedding_client = EmbeddingClient(settings)
    embedding_service = EmbeddingService(embedding_client, batch_size)
    logger.info("[3/4] امبدینگ %d چانک با «%s» (دسته‌های %dتایی) …",
                len(chunks), settings.embedding_model, batch_size)
    try:
        embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])
    except RAGSystemError as exc:
        logger.error("خطا در سرویس امبدینگ: %s", exc)
        logger.error("سلامت سرویس‌ها را بررسی کنید: python scripts/check_services.py")
        return 1

    # ۴) ساخت و ذخیرهٔ ایندکس FAISS
    index_dir = Path(args.index_dir) if args.index_dir else settings.index_dir
    logger.info("[4/4] ساخت و ذخیرهٔ ایندکس FAISS در «%s» …", index_dir)
    vector_store = VectorStore(index_dir)
    try:
        vector_store.build(chunks, embeddings)
        vector_store.save()
    except (RAGSystemError, ValueError) as exc:
        logger.error("خطا در ساخت ایندکس: %s", exc)
        return 1

    info = vector_store.info
    print(f"\nایندکس با موفقیت ساخته و ذخیره شد.")
    print(f"  مسیر       : {index_dir}")
    print(f"  چانک‌ها     : {info['count']}")
    print(f"  ابعاد بردار : {info['dimension']}")
    print(f"  فایل‌ها     : {len(info['sources'])}")
    print("\nحالا رابط کاربری را اجرا کنید:  python run_app.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
