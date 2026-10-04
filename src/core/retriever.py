"""
بازیاب معنایی (Retriever)
=========================
نقطهٔ ورودِ «امبد شدن پرامپت کاربر»: پرس‌وجو با سرویس امبدینگ به
بردار تبدیل و در ایندکس FAISS جست‌وجو می‌شود؛ نتیجه، لیستی از
مرتبط‌ترین چانک‌ها همراه با نام فایل و امتیاز شباهت است.
"""

from __future__ import annotations

import logging

from .embedder import EmbeddingService
from .models import RetrievalResult
from .vector_store import VectorStore

logger = logging.getLogger("rag.retriever")


class Retriever:
    """هماهنگی امبدینگ پرس‌وجو و جست‌وجوی برداری."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """امبد کردن پرس‌وجو و برگرداندن ‎top_k‎ چانکِ مرتبط."""
        query = (query or "").strip()
        if not query:
            return []
        if not self.vector_store.is_ready():
            logger.warning("ایندکس آماده نیست؛ جست‌وجو نتیجه‌ای برنمی‌گرداند.")
            return []
        query_vector = self.embedding_service.embed_query(query)
        results = self.vector_store.search(query_vector, top_k)
        logger.info("جست‌وجو «%s…» → %d نتیجه.", query[:40], len(results))
        return results
