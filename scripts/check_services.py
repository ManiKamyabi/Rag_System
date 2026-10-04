"""
بررسی سلامت سرویس‌ها
====================
اتصال به سرویس مدل زبانی (Qwen3) و سرویس امبدینگ (bge-m3) را
به همراه یک درخواست واقعی کوچک تست می‌کند و وضعیت ایندکس FAISS
را نشان می‌دهد. قبل از اجرای اپ یا embed، این اسکریپت را اجرا کنید:

    python scripts/check_services.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.core.api_client import EmbeddingClient, LLMClient
from src.core.vector_store import VectorStore

logging.basicConfig(level=logging.WARNING)


def mark(ok: bool) -> str:
    return "[ OK ]" if ok else "[FAIL]"


def main() -> int:
    settings = Settings.load()
    all_ok = True
    print("=" * 56)
    print("بررسی سلامت سرویس‌های سیستم RAG")
    print("=" * 56)

    # ── سرویس مدل زبانی ─────────────────────────────────────
    llm = LLMClient(settings)
    print(f"\n1) سرویس مدل زبانی ({settings.llm_base_url})")
    alive = llm.health_check()
    print(f"   {mark(alive)} اتصال به /health")
    all_ok &= alive
    if alive:
        try:
            reply = llm.chat(
                [{"role": "user", "content": "فقط بنویس: سلام"}],
                temperature=0.1, max_tokens=16,
            )
            ok = bool(reply.strip())
            print(f"   {mark(ok)} تولید متن آزمایشی: «{reply.strip()[:40]}»")
            all_ok &= ok
        except Exception as exc:  # noqa: BLE001
            print(f"   {mark(False)} تولید متن آزمایشی: {exc}")
            all_ok = False

    # ── سرویس امبدینگ ───────────────────────────────────────
    emb = EmbeddingClient(settings)
    print(f"\n2) سرویس امبدینگ ({settings.embedding_base_url})")
    alive = emb.health_check()
    print(f"   {mark(alive)} اتصال به /health")
    all_ok &= alive
    if alive:
        try:
            vector = emb.embed(["این یک متن آزمایشی است."])[0]
            ok = len(vector) > 0
            print(f"   {mark(ok)} امبدینگ متن آزمایشی (ابعاد: {len(vector)})")
            all_ok &= ok
        except Exception as exc:  # noqa: BLE001
            print(f"   {mark(False)} امبدینگ متن آزمایشی: {exc}")
            all_ok = False

    # ── ایندکس FAISS ────────────────────────────────────────
    print("\n3) ایندکس FAISS")
    store = VectorStore(settings.index_dir)
    loaded = store.try_load()
    if loaded:
        stats = store.stats()
        print(f"   {mark(True)} ایندکس بارگذاری شد — {stats['count']} چانک، "
              f"{len(stats['sources'])} فایل، ابعاد {stats['dimension']}، "
              f"ساخته‌شده: {stats['created_at']}")
    else:
        print("   [INFO] ایندکسی یافت نشد — با اسکریپت embed_dataset.py بسازید:")
        print("          python scripts/embed_dataset.py --data data/data.csv")

    # ── نتیجه ───────────────────────────────────────────────
    print("\n" + "=" * 56)
    if all_ok:
        print("نتیجه: سرویس‌ها سالم‌اند. برای اجرای اپ: python run_app.py")
    else:
        print("نتیجه: برخی سرویس‌ها در دسترس نیستند!")
        print("چک‌لیست:")
        print("  - docker compose up -d  (هر دو سرویس qwen3 و bge-m3)")
        print("  - پورت‌ها: 8080 (مدل) و 8081 (امبدینگ)")
        print("  - مقادیر LLM_BASE_URL و EMBEDDING_BASE_URL در فایل .env")
    print("=" * 56)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
