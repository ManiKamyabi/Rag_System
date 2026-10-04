"""
ذخیره‌گاه برداری FAISS
======================
مدیریت ایندکس ‎IndexFlatIP‎ (جست‌وجوی دقیق) روی بردارهای نرمال‌شده؛
بنابراین امتیازها معادل شباهت کسینوسی هستند. متن و متادیتای هر
چانک در فایل ‎metadata.json‎ کنار ‎index.faiss‎ نگهداری می‌شود.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np

from .exceptions import IndexNotFoundError
from .models import Chunk, RetrievalResult

logger = logging.getLogger("rag.vector_store")


class VectorStore:
    """ساخت، ذخیره، بارگذاری و جست‌وجوی ایندکس FAISS."""

    INDEX_FILE = "index.faiss"
    META_FILE = "metadata.json"

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.index = None            # faiss.IndexFlatIP
        self.metadatas: list[dict] = []
        self.info: dict = {}

    # ── وضعیت ────────────────────────────────────────────────
    def is_ready(self) -> bool:
        """آیا ایندکس بارگذاری شده و حداقل یک بردار دارد؟"""
        return self.index is not None and self.index.ntotal > 0

    def try_load(self) -> bool:
        """بارگذاری ایندکس از دیسک در صورت وجود؛ در غیر این صورت False."""
        index_path = self.index_dir / self.INDEX_FILE
        meta_path = self.index_dir / self.META_FILE
        if not index_path.is_file() or not meta_path.is_file():
            return False
        try:
            self.index = faiss.read_index(str(index_path))
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            self.metadatas = payload.get("metadatas", [])
            self.info = payload.get("info", {})
            if len(self.metadatas) != self.index.ntotal:
                raise IndexNotFoundError(
                    "تعداد متادیتاها با تعداد بردارهای ایندکس هم‌خوانی ندارد؛ "
                    "ایندکس را دوباره بسازید."
                )
            logger.info("ایندکس بارگذاری شد: %d چانک از «%s».",
                        self.index.ntotal, self.index_dir)
            return True
        except (OSError, RuntimeError, json.JSONDecodeError, ValueError,
                IndexNotFoundError) as exc:
            self.index = None
            self.metadatas = []
            self.info = {}
            logger.error("بارگذاری ایندکس ناموفق بود: %s", exc)
            return False

    # ── ساخت و ذخیره ─────────────────────────────────────────
    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """ساخت ایندکس از چانک‌ها و بردارهای آن‌ها."""
        if not chunks:
            raise IndexNotFoundError("چانکی برای ساخت ایندکس وجود ندارد.")
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("تعداد چانک‌ها با ابعاد ماتریس بردارها هم‌خوانی ندارد.")
        vectors = np.ascontiguousarray(embeddings)
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        self.index = index
        self.metadatas = [
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "char_count": chunk.char_count,
            }
            for chunk in chunks
        ]
        self.info = {
            "dimension": dimension,
            "count": len(chunks),
            "sources": sorted({chunk.source for chunk in chunks}),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def save(self) -> Path:
        """ذخیرهٔ ایندکس و متادیتا روی دیسک."""
        if self.index is None:
            raise IndexNotFoundError("ایندکسی برای ذخیره ساخته نشده است.")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / self.INDEX_FILE))
        payload = {"info": self.info, "metadatas": self.metadatas}
        (self.index_dir / self.META_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        logger.info("ایندکس در «%s» ذخیره شد (%d چانک).", self.index_dir, self.index.ntotal)
        return self.index_dir

    # ── جست‌وجو ───────────────────────────────────────────────
    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievalResult]:
        """جست‌وجوی بردار پرس‌وجو و برگرداندن مرتبط‌ترین چانک‌ها."""
        if not self.is_ready():
            return []
        vector = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype="float32"
        )
        if vector.shape[1] != self.index.d:
            raise IndexNotFoundError(
                f"ابعاد بردار پرس‌وجو ({vector.shape[1]}) با ابعاد ایندکس "
                f"({self.index.d}) فرق دارد؛ ایندکس را دوباره بسازید."
            )
        k = min(max(1, top_k), self.index.ntotal)
        scores, ids = self.index.search(vector, k)
        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0].tolist(), ids[0].tolist())):
            if idx < 0:
                continue   # FAISS برای k بیش از ntotal مقدار -1 برمی‌گرداند
            meta = self.metadatas[idx]
            results.append(RetrievalResult(
                text=meta["text"],
                score=float(score),
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                rank=rank + 1,
            ))
        return results

    def stats(self) -> dict:
        """آمار ایندکس برای نمایش در UI و ابزار get_knowledge_base_stats."""
        ready = self.is_ready()
        return {
            "ready": ready,
            "count": int(self.index.ntotal) if self.index is not None else 0,
            "dimension": int(self.index.d) if self.index is not None else 0,
            "sources": self.info.get("sources", []),
            "created_at": self.info.get("created_at", ""),
        }
