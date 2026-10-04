"""
سرویس امبدینگ — لایهٔ بالادستی EmbeddingClient
===============================================
دسته‌بندی (batch)، نرمال‌سازی L2 و اعتبارسنجی خروجی. تمام اجزای
سیستم (خط لولهٔ RAG و ابزارهای مدل) فقط از طریق همین سرویس
امبدینگ انجام می‌دهند.
"""

from __future__ import annotations

import logging

import numpy as np

from .api_client import EmbeddingClient
from .exceptions import APIResponseError

logger = logging.getLogger("rag.embedding")


class EmbeddingService:
    """سرویس امبدینگ با دسته‌بندی خودکار و نرمال‌سازی بردارها.

    نرمال‌سازی L2 باعث می‌شود ضرب داخلی (IndexFlatIP در FAISS)
    معادل شباهت کسینوسی باشد.
    """

    def __init__(self, client: EmbeddingClient, batch_size: int = 16):
        self.client = client
        self.batch_size = max(1, batch_size)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """امبدینگ چند متن به‌صورت دسته‌ای؛ خروجی آرایه‌ای ‎(N, D)‎ نرمال‌شده."""
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self.client.embed(batch))
            done = min(start + self.batch_size, total)
            logger.info("امبدینگ %d از %d متن انجام شد.", done, total)
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] != total:
            raise APIResponseError(
                "[Embedding] تعداد بردارهای دریافتی با تعداد ورودی برابر نیست."
            )
        return self._normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        """امبدینگ یک متن (پرس‌وجوی کاربر یا جست‌وجوی ابزار)؛ خروجی ‎(D,)‎."""
        matrix = self.embed_texts([text])
        return matrix[0]

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """نرمال‌سازی L2 سطر به سطر (نرم صفر به‌جای تقسیم بر صفر → 1)."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype("float32")
