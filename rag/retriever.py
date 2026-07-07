"""
rag/retriever.py

ماژول Retrieval Logic برای پروژه «امدادیار رضوی».

این ماژول مسئول موارد زیر است:
    1. تبدیل پرسش کاربر به بردار Embedding (با استفاده از همان مدل rag/vectorstore.py)
    2. جستجوی Top-K نزدیک‌ترین Chunkها در ایندکس FAISS
    3. فیلتر نتایج بر اساس آستانه امتیاز شباهت (اختیاری)
    4. بازگرداندن نتایج به شکل استاندارد RetrievedChunk

خروجی این ماژول مستقیماً ورودی ماژول generation/prompt_builder.py
(طبق معماری کلی) خواهد بود و مبنای پاسخ‌گویی، تولید سناریوی آموزشی
و تولید سؤالات ارزیابی قرار می‌گیرد.

نکته معماری مهم: این ماژول تنها از داده‌های موجود در ایندکس FAISS
(که صرفاً از PDFهای پوشه knowledge_base ساخته شده) بازیابی انجام
می‌دهد و هیچ منبع داده دیگری را مصرف نمی‌کند.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from rag.vectorstore import EmbeddingModel, FAISSVectorStore, IndexNotBuiltError

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------- #
# مدل داده خروجی
# --------------------------------------------------------------------------- #
@dataclass
class RetrievedChunk:
    """
    نتیجه یک بازیابی واحد از ایندکس FAISS.

    Attributes:
        text: متن Chunk بازیابی‌شده.
        source_file: نام فایل PDF منبع این Chunk.
        page_number: شماره صفحه منبع در فایل PDF.
        chunk_index: شماره ترتیبی Chunk در همان صفحه.
        score: امتیاز شباهت (Inner Product روی بردارهای نرمال‌شده؛ بازه تقریبی [-1, 1]).
    """

    text: str
    source_file: str
    page_number: int
    chunk_index: int
    score: float

    def to_dict(self) -> dict:
        """تبدیل به دیکشنری (مثلاً برای Serialize کردن در پاسخ API)."""
        return {
            "text": self.text,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "score": self.score,
        }


# --------------------------------------------------------------------------- #
# خطاهای اختصاصی
# --------------------------------------------------------------------------- #
class RetrieverNotReadyError(Exception):
    """خطای مربوط به تلاش برای بازیابی از Vector Store‌ای که آماده نیست."""


class EmptyQueryError(Exception):
    """خطای مربوط به دریافت پرسش خالی یا فقط شامل فاصله."""


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """
    لایه Retrieval که پرسش کاربر را به نتایج بازیابی‌شده از FAISS تبدیل می‌کند.

    این کلاس Wrapper سطح بالایی روی FAISSVectorStore و EmbeddingModel است و
    منطق کسب‌وکاری بازیابی (فیلتر آستانه، محدودسازی top_k، مرتب‌سازی) را
    از منطق خام ذخیره‌سازی برداری جدا نگه می‌دارد.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model: Optional[EmbeddingModel] = None,
        default_top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            vector_store: نمونه از FAISSVectorStore که ایندکس آن قبلاً build یا load شده است.
            embedding_model: مدل Embedding برای Encode کردن پرسش؛ در صورت عدم ارائه،
                از همان مدل موجود در vector_store استفاده می‌شود (توصیه‌شده، برای
                تضمین سازگاری فضای برداری).
            default_top_k: تعداد پیش‌فرض نتایج بازگشتی در صورت عدم مشخص‌شدن در فراخوانی.
            score_threshold: حداقل امتیاز شباهت قابل قبول؛ نتایج پایین‌تر از این مقدار
                فیلتر می‌شوند. مقدار None به‌معنای عدم فیلتر است.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model or vector_store.embedding_model
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        بازیابی مرتبط‌ترین Chunkها برای یک پرسش کاربر.

        Args:
            query: متن پرسش کاربر.
            top_k: تعداد نتایج مورد نیاز؛ در صورت None از default_top_k استفاده می‌شود.
            score_threshold: آستانه امتیاز برای این فراخوانی خاص؛ در صورت None از
                مقدار پیش‌فرض کلاس (self.score_threshold) استفاده می‌شود.

        Returns:
            لیستی از RetrievedChunk مرتب‌شده بر اساس بیشترین امتیاز شباهت.

        Raises:
            EmptyQueryError: اگر پرسش خالی یا فقط شامل فاصله باشد.
            RetrieverNotReadyError: اگر ایندکس Vector Store هنوز آماده نباشد.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise EmptyQueryError("پرسش ورودی نمی‌تواند خالی باشد.")

        if not self.vector_store.is_ready:
            raise RetrieverNotReadyError(
                "ایندکس Vector Store آماده نیست. ابتدا باید Pipeline ساخت ایندکس "
                "(indexing_pipeline) اجرا شده و ایندکس load شده باشد."
            )

        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )

        logger.info("در حال بازیابی برای پرسش: '%s' (top_k=%d)", cleaned_query, effective_top_k)

        try:
            query_vector = self.embedding_model.encode_single(cleaned_query)
            raw_results = self.vector_store.search(query_vector, top_k=effective_top_k)
        except IndexNotBuiltError as exc:
            raise RetrieverNotReadyError(str(exc)) from exc

        retrieved_chunks: List[RetrievedChunk] = []
        for metadata, score in raw_results:
            if effective_threshold is not None and score < effective_threshold:
                continue
            retrieved_chunks.append(
                RetrievedChunk(
                    text=metadata["text"],
                    source_file=metadata["source_file"],
                    page_number=metadata["page_number"],
                    chunk_index=metadata["chunk_index"],
                    score=score,
                )
            )

        logger.info(
            "بازیابی کامل شد: %d نتیجه (پس از اعمال آستانه امتیاز).",
            len(retrieved_chunks),
        )
        return retrieved_chunks

    def retrieve_as_context_string(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        include_citations: bool = True,
    ) -> str:
        """
        بازیابی و تبدیل مستقیم نتایج به یک رشته متنی آماده تزریق در Prompt.

        این متد برای سادگی استفاده در generation/prompt_builder.py طراحی شده
        تا آن ماژول مجبور به تکرار منطق قالب‌بندی نباشد.

        Args:
            query: پرسش کاربر.
            top_k: تعداد نتایج.
            score_threshold: آستانه امتیاز.
            include_citations: در صورت True، هر بخش با ذکر نام فایل و شماره صفحه
                برچسب‌گذاری می‌شود تا در پاسخ نهایی امکان استناد وجود داشته باشد.

        Returns:
            رشته متنی حاوی محتوای Chunkهای بازیابی‌شده؛ در صورت نبود نتیجه، رشته خالی.
        """
        chunks = self.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
        if not chunks:
            return ""

        parts: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            if include_citations:
                header = f"[منبع {i}: {chunk.source_file} - صفحه {chunk.page_number}]"
                parts.append(f"{header}\n{chunk.text}")
            else:
                parts.append(chunk.text)

        return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------- #
# اجرای مستقل جهت تست سریع (python -m rag.retriever)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/vector_store"
    test_query = sys.argv[2] if len(sys.argv) > 2 else "اقدامات اولیه در زلزله چیست؟"

    embedder = EmbeddingModel()
    store = FAISSVectorStore(embedding_model=embedder, index_dir=index_dir)
    store.load()

    retriever = Retriever(vector_store=store, default_top_k=5)
    results = retriever.retrieve(test_query)

    print(f"پرسش: {test_query}")
    print(f"تعداد نتایج بازیابی‌شده: {len(results)}\n")
    for rank, item in enumerate(results, start=1):
        print(f"--- نتیجه {rank} (امتیاز: {item.score:.4f}) ---")
        print(f"منبع: {item.source_file} | صفحه: {item.page_number}")
        print(item.text[:300])
        print()
