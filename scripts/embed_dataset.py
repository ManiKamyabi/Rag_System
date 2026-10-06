"""
امبد کردن دیتاست و ساخت ایندکس FAISS
=====================================
دیتاست (فایل CSV با ستون‌های filename و content) را می‌خواند،
پاراگراف‌به‌پاراگراف و «بدون افت متن» چانک می‌کند، با سرویس
امبدینگ (bge-m3) بردارها را می‌سازد و ایندکس FAISS را ذخیره می‌کند.

ویژگی‌های ایمنی (برای جلوگیری از هدر رفتن هزینه):
    * اعتبارسنجی ورودی‌ها و دسترسی نوشتن قبل از شروع امبد
    * تست سلامت سرویس امبدینگ قبل از شروع کار اصلی
    * حذف چانک‌های خالی و متن‌های تکراری (امبد فقط یک‌بار)
    * Retry با backoff و تقسیم دسته در صورت خطا
    * کش دیسکی هر دسته؛ در صورت خطا/Ctrl+C اجرای بعدی از همان‌جا ادامه می‌دهد
    * نوار پیشرفت با درصد، سرعت، زمان سپری‌شده و زمان باقی‌مانده (ETA)

استفاده از ریشهٔ پروژه:

    python scripts/embed_dataset.py --data /مسیر/به/data.csv
    python scripts/embed_dataset.py --data ./data              # پوشهٔ CSV ها
    python scripts/embed_dataset.py --data data.csv --preview  # فقط گزارش چانکینگ
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

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

MAX_RETRIES = 4          # تعداد تلاش مجدد برای هر دسته
BACKOFF_BASE = 2         # ثانیه؛ 2، 4، 8، 16 …


# ───────────────────────── نوار پیشرفت ─────────────────────────
class ProgressBar:
    """نوار پیشرفت؛ اگر tqdm نصب باشد از آن استفاده می‌شود، وگرنه نسخهٔ ساده."""

    def __init__(self, total: int, desc: str = "امبدینگ") -> None:
        self.total = total
        self.n = 0
        self.start = time.time()
        self._tqdm = None
        try:
            from tqdm import tqdm  # type: ignore
            self._tqdm = tqdm(total=total, desc=desc, unit="chunk",
                              dynamic_ncols=True,
                              bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                                         "{n_fmt}/{total_fmt} "
                                         "[سپری‌شده {elapsed} | باقی‌مانده {remaining} | {rate_fmt}]")
        except ImportError:
            self._render()

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = int(max(sec, 0))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _render(self) -> None:
        elapsed = time.time() - self.start
        rate = self.n / elapsed if elapsed > 0 and self.n else 0.0
        eta = (self.total - self.n) / rate if rate else 0.0
        pct = self.n / self.total if self.total else 1.0
        filled = int(30 * pct)
        bar = "█" * filled + "░" * (30 - filled)
        sys.stderr.write(
            f"\rامبدینگ: {pct*100:5.1f}% |{bar}| {self.n}/{self.total} "
            f"[سپری‌شده {self._fmt(elapsed)} | باقی‌مانده {self._fmt(eta)} | "
            f"{rate:.1f} chunk/s]   ")
        sys.stderr.flush()

    def update(self, k: int) -> None:
        self.n += k
        if self._tqdm is not None:
            self._tqdm.update(k)
        else:
            self._render()

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
        else:
            sys.stderr.write("\n")


# ───────────────────────── ابزارهای کمکی ─────────────────────────
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
    parser.add_argument("--cache-dir", default=None,
                        help="پوشهٔ کش بردارها برای ادامهٔ کار پس از خطا")
    parser.add_argument("--no-cache", action="store_true",
                        help="غیرفعال کردن کش (ادامهٔ کار پس از خطا ممکن نیست)")
    parser.add_argument("--keep-cache", action="store_true",
                        help="پاک نکردن کش پس از ساخت موفق ایندکس")
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


def validate_args(args: argparse.Namespace, paragraphs: int, overlap: int,
                  max_chars: int, batch_size: int) -> str | None:
    """قبل از هر هزینه‌ای ورودی‌ها را بررسی می‌کند؛ پیام خطا یا None."""
    data_path = Path(args.data)
    if not data_path.exists():
        return f"مسیر دیتا وجود ندارد: {data_path}"
    if data_path.is_dir() and not any(data_path.glob("*.csv")):
        return f"در پوشهٔ {data_path} هیچ فایل CSV ای پیدا نشد."
    if paragraphs < 1:
        return "--paragraphs باید حداقل 1 باشد."
    if overlap < 0 or overlap >= paragraphs:
        return "--overlap باید بین 0 و (paragraphs-1) باشد."
    if max_chars < 100:
        return "--max-chars خیلی کوچک است (حداقل 100)."
    if batch_size < 1:
        return "--batch-size باید حداقل 1 باشد."
    return None


def check_writable(directory: Path) -> str | None:
    """مطمئن می‌شود قبل از امبد، امکان نوشتن در مسیر خروجی هست."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, delete=True):
            pass
    except OSError as exc:
        return f"امکان نوشتن در «{directory}» نیست: {exc}"
    usage = shutil.disk_usage(directory)
    if usage.free < 200 * 1024 * 1024:
        return f"فضای دیسک کم است ({usage.free // (1024*1024)} MB آزاد)."
    return None


def to_array(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"خروجی امبدینگ دوبعدی نیست (shape={arr.shape})")
    return arr


def check_vectors(arr: np.ndarray, n_expected: int, dim: int | None) -> None:
    if arr.shape[0] != n_expected:
        raise ValueError(f"تعداد بردار ({arr.shape[0]}) با تعداد متن ({n_expected}) برابر نیست")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f"بُعد بردار {arr.shape[1]} است ولی {dim} انتظار می‌رفت")
    if not np.isfinite(arr).all():
        raise ValueError("بردار نامعتبر (NaN/Inf) از سرویس برگشت")


def embed_batch(service: EmbeddingService, texts: list[str], dim: int | None,
                retries: int) -> np.ndarray:
    """یک دسته را با retry و backoff امبد می‌کند و خروجی را اعتبارسنجی می‌کند."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            arr = to_array(service.embed_texts(texts))
            check_vectors(arr, len(texts), dim)
            return arr
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - خطاهای شبکه/سرویس متنوع‌اند
            last_exc = exc
            if attempt < retries:
                wait = BACKOFF_BASE * (2 ** attempt)
                logger.warning("\n⚠ خطا در دسته (%d متن): %s — تلاش مجدد %d/%d بعد از %d ثانیه",
                               len(texts), exc, attempt + 1, retries, wait)
                time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def embed_robust(service: EmbeddingService, texts: list[str], dim: int | None,
                 retries: int = MAX_RETRIES) -> np.ndarray:
    """اگر دسته حتی بعد از retry خطا داد، نصف می‌کند تا متن مشکل‌دار جدا شود."""
    try:
        return embed_batch(service, texts, dim, retries)
    except KeyboardInterrupt:
        raise
    except Exception:  # noqa: BLE001
        if len(texts) == 1:
            raise
        mid = len(texts) // 2
        logger.warning("\n⚠ دسته به دو نیم تقسیم شد (%d + %d) تا متن مشکل‌دار پیدا شود",
                       mid, len(texts) - mid)
        left = embed_robust(service, texts[:mid], dim, retries=1)
        right = embed_robust(service, texts[mid:], dim, retries=1)
        return np.vstack([left, right])


def batch_cache_path(cache_dir: Path, model: str, texts: list[str]) -> Path:
    h = hashlib.sha1(model.encode("utf-8"))
    for t in texts:
        h.update(hashlib.sha1(t.encode("utf-8")).digest())
    return cache_dir / f"{h.hexdigest()}.npy"


def embed_all(service: EmbeddingService, texts: list[str], batch_size: int,
              model: str, cache_dir: Path | None, dim: int) -> np.ndarray:
    """همهٔ متن‌ها را دسته‌ای امبد می‌کند؛ با کش و نوار پیشرفت."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    parts: list[np.ndarray] = []
    bar = ProgressBar(len(texts))
    cached_batches = 0
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            cpath = batch_cache_path(cache_dir, model, batch) if cache_dir else None
            arr = None
            if cpath is not None and cpath.exists():
                try:
                    arr = np.load(cpath)
                    check_vectors(arr, len(batch), dim)
                    cached_batches += 1
                except Exception:  # noqa: BLE001 - کش خراب را نادیده بگیر
                    arr = None
            if arr is None:
                arr = embed_robust(service, batch, dim)
                if cpath is not None:
                    tmp = cpath.with_suffix(".tmp.npy")
                    np.save(tmp, arr)
                    tmp.replace(cpath)  # نوشتن اتمیک
            parts.append(arr)
            bar.update(len(batch))
    finally:
        bar.close()
    if cached_batches:
        logger.info("↻ %d دسته از کش خوانده شد (بدون هزینه).", cached_batches)
    return np.vstack(parts)


# ───────────────────────── برنامهٔ اصلی ─────────────────────────
def main() -> int:
    args = parse_args()
    try:
        settings = Settings.load()
    except Exception as exc:  # noqa: BLE001
        logger.error("خطا در بارگذاری تنظیمات (.env): %s", exc)
        return 1

    paragraphs = args.paragraphs if args.paragraphs is not None else settings.chunk_paragraphs
    overlap = args.overlap if args.overlap is not None else settings.chunk_overlap_paragraphs
    max_chars = args.max_chars if args.max_chars is not None else settings.chunk_max_chars
    batch_size = args.batch_size if args.batch_size is not None else settings.embed_batch_size

    # ۰) اعتبارسنجی پیش از هر هزینه‌ای
    problem = validate_args(args, paragraphs, overlap, max_chars, batch_size)
    if problem:
        logger.error("خطای ورودی: %s", problem)
        return 1
    index_dir = Path(args.index_dir) if args.index_dir else Path(settings.index_dir)
    if not args.preview:
        problem = check_writable(index_dir)
        if problem:
            logger.error("%s", problem)
            return 1

    chunker = ParagraphChunker(
        paragraphs_per_chunk=paragraphs,
        overlap_paragraphs=overlap,
        max_chunk_chars=max_chars,
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
    except Exception as exc:  # noqa: BLE001 - انکودینگ خراب، فایل قفل‌شده و …
        logger.error("خطای پیش‌بینی‌نشده در خواندن دیتاست: %s", exc)
        return 1
    if not documents:
        logger.error("دیتاست خالی است.")
        return 1

    # ۲) چانکینگ پاراگراف‌محور + راستی‌آزمایی پوشش کامل
    logger.info("[2/4] چانکینگ پاراگراف‌محور …")
    try:
        chunks = chunker.chunk_all(documents)
    except RAGSystemError as exc:
        logger.error("خطا در چانکینگ: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("خطای پیش‌بینی‌نشده در چانکینگ: %s", exc)
        return 1
    report = ParagraphChunker.build_report(documents, chunks)
    print_report(report)

    if args.preview:
        logger.info("حالت پیش‌نمایش: امبدینگ انجام نشد.")
        return 0

    # حذف چانک‌های خالی (سرویس امبدینگ معمولاً روی متن خالی خطا می‌دهد)
    non_empty = [c for c in chunks if c.text and c.text.strip()]
    if len(non_empty) != len(chunks):
        logger.warning("⚠ %d چانک خالی حذف شد.", len(chunks) - len(non_empty))
    chunks = non_empty
    if not chunks:
        logger.error("هیچ چانک غیرخالی‌ای برای امبد وجود ندارد.")
        return 1

    # حذف متن‌های تکراری: هر متن یک‌بار امبد می‌شود
    unique_texts: list[str] = []
    seen: dict[str, int] = {}
    inverse: list[int] = []
    for c in chunks:
        idx = seen.get(c.text)
        if idx is None:
            idx = len(unique_texts)
            seen[c.text] = idx
            unique_texts.append(c.text)
        inverse.append(idx)
    n_requests = -(-len(unique_texts) // batch_size)
    logger.info("چانک‌ها: %d | یکتا: %d (%d تکراری امبد نمی‌شود) | درخواست‌ها: ~%d | کاراکتر: %s",
                len(chunks), len(unique_texts), len(chunks) - len(unique_texts),
                n_requests, f"{sum(map(len, unique_texts)):,}")

    # ۳) تست سلامت سرویس قبل از شروع کار اصلی
    embedding_client = EmbeddingClient(settings)
    embedding_service = EmbeddingService(embedding_client, batch_size)
    logger.info("[3/4] تست سرویس امبدینگ «%s» …", settings.embedding_model)
    try:
        probe = embed_batch(embedding_service, ["تست سلامت سرویس"], None, retries=2)
        dim = int(probe.shape[1])
        logger.info("سرویس سالم است (بُعد بردار: %d).", dim)
    except Exception as exc:  # noqa: BLE001
        logger.error("سرویس امبدینگ در دسترس نیست: %s", exc)
        logger.error("سلامت سرویس‌ها را بررسی کنید: python scripts/check_services.py")
        return 1

    cache_dir = None if args.no_cache else (
        Path(args.cache_dir) if args.cache_dir else index_dir.parent / ".embed_cache")
    logger.info("امبدینگ %d متن (دسته‌های %dتایی)%s …", len(unique_texts), batch_size,
                f" | کش: {cache_dir}" if cache_dir else "")
    try:
        unique_emb = embed_all(embedding_service, unique_texts, batch_size,
                               settings.embedding_model, cache_dir, dim)
    except KeyboardInterrupt:
        logger.error("\nمتوقف شد. بردارهای تا اینجا در کش ذخیره‌اند؛ "
                     "همان دستور را دوباره اجرا کنید تا ادامه یابد.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("\nخطا در سرویس امبدینگ: %s", exc)
        logger.error("پیشرفت تا اینجا در کش ذخیره شد؛ بعد از رفع مشکل همان دستور را "
                     "دوباره اجرا کنید تا از ادامه شروع شود.")
        logger.error("بررسی سلامت: python scripts/check_services.py")
        return 1

    embeddings = unique_emb[np.asarray(inverse)]
    try:
        check_vectors(embeddings, len(chunks), dim)
    except ValueError as exc:
        logger.error("بردارهای نهایی نامعتبرند: %s", exc)
        return 1

    # ۴) ساخت و ذخیرهٔ ایندکس FAISS
    logger.info("[4/4] ساخت و ذخیرهٔ ایندکس FAISS در «%s» …", index_dir)
    vector_store = VectorStore(index_dir)
    try:
        vector_store.build(chunks, embeddings)
        vector_store.save()
    except (RAGSystemError, ValueError, OSError) as exc:
        logger.error("خطا در ساخت/ذخیرهٔ ایندکس: %s", exc)
        logger.error("بردارها در کش هستند؛ اجرای دوباره هزینهٔ امبد ندارد.")
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("خطای پیش‌بینی‌نشده در ساخت ایندکس: %s", exc)
        logger.error("بردارها در کش هستند؛ اجرای دوباره هزینهٔ امبد ندارد.")
        return 1

    # ایندکس با موفقیت ذخیره شد؛ کش دیگر لازم نیست
    if cache_dir is not None and not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

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
