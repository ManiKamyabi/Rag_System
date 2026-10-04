"""
تنظیمات سراسری سیستم RAG
=========================
تمام تنظیمات از فایل ‎.env‎ در ریشهٔ پروژه (در صورت وجود) و در غیر
این صورت از مقادیر پیش‌فرض خوانده می‌شود. تمام اجزای سیستم فقط از
این ماژول تنظیمات خود را دریافت می‌کنند (یک منبع حقیقت واحد).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ── ریشهٔ پروژه ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """بارگذاری سبکِ فایل ‎.env‎ بدون وابستگی خارجی (کلیدهای موجود در
    محیط بازنویسی نمی‌شوند تا متغیرهای shell اولویت داشته باشند)."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # حذف کامنت انتهای خط و کوتیشن‌های دور مقدار
        value = value.split("#")[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """کانتینر یکپارچهٔ تنظیمات؛ با ‎Settings.load()‎ ساخته می‌شود."""

    # ── سرویس مدل زبانی (Qwen3 روی llama.cpp) ──
    llm_base_url: str
    llm_model: str
    llm_timeout: int               # ثانیه
    llm_max_retries: int

    # ── سرویس امبدینگ (bge-m3 روی llama.cpp) ──
    embedding_base_url: str
    embedding_model: str
    embedding_timeout: int
    embedding_max_retries: int
    embed_batch_size: int

    # ── مسیرها ──
    data_dir: Path
    storage_dir: Path
    index_dir: Path
    history_db_path: Path

    # ── چانکینگ ──
    chunk_paragraphs: int          # تعداد پاراگراف در هر چانک
    chunk_overlap_paragraphs: int  # هم‌پوشانی پاراگرافی بین چانک‌ها
    chunk_max_chars: int           # سقف کاراکترِ هر چانک
    paragraph_max_chars: int       # سقف کاراکترِ یک پاراگراف پیش از ریزشگی

    # ── بازیابی ──
    default_top_k: int
    max_context_chars: int         # سقف کاراکترِ بخش <context>
    tool_result_max_chars: int     # سقف کاراکترِ نتیجهٔ هر ابزار

    # ── تولید پاسخ ──
    temperature: float
    max_tokens: int
    max_tool_calls: int            # حداکثر دفعات فراخوانی ابزار در یک پاسخ

    @classmethod
    def load(cls, **overrides) -> "Settings":
        """ساخت تنظیمات از ‎.env‎ و متغیرهای محیطی؛ با overrides قابل بازنویسی."""
        values: dict = dict(
            llm_base_url=_env_str("LLM_BASE_URL", "http://localhost:8080/v1"),
            llm_model=_env_str("LLM_MODEL", "qwen3"),
            llm_timeout=_env_int("LLM_TIMEOUT", 300),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
            embedding_base_url=_env_str("EMBEDDING_BASE_URL", "http://localhost:8081/v1"),
            embedding_model=_env_str("EMBEDDING_MODEL", "bge-m3"),
            embedding_timeout=_env_int("EMBEDDING_TIMEOUT", 180),
            embedding_max_retries=_env_int("EMBEDDING_MAX_RETRIES", 3),
            embed_batch_size=_env_int("EMBED_BATCH_SIZE", 16),
            data_dir=Path(_env_str("DATA_DIR", str(ROOT_DIR / "data"))),
            storage_dir=Path(_env_str("STORAGE_DIR", str(ROOT_DIR / "storage"))),
            chunk_paragraphs=_env_int("CHUNK_PARAGRAPHS", 4),
            chunk_overlap_paragraphs=_env_int("CHUNK_OVERLAP_PARAGRAPHS", 1),
            chunk_max_chars=_env_int("CHUNK_MAX_CHARS", 3000),
            paragraph_max_chars=_env_int("PARAGRAPH_MAX_CHARS", 1500),
            default_top_k=_env_int("DEFAULT_TOP_K", 5),
            max_context_chars=_env_int("MAX_CONTEXT_CHARS", 6000),
            tool_result_max_chars=_env_int("TOOL_RESULT_MAX_CHARS", 3000),
            temperature=_env_float("TEMPERATURE", 0.3),
            max_tokens=_env_int("MAX_TOKENS", 1500),
            max_tool_calls=_env_int("MAX_TOOL_CALLS", 3),
        )
        storage = values["storage_dir"]
        values["index_dir"] = Path(_env_str("INDEX_DIR", str(storage / "index")))
        values["history_db_path"] = Path(
            _env_str("HISTORY_DB_PATH", str(storage / "history.db"))
        )
        # بازنویسی صریح (برای تست یا فراخوانی برنامه‌ای)
        for key, value in overrides.items():
            if value is not None and key in values:
                values[key] = value
        return cls(**values)
