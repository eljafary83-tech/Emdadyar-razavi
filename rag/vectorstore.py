"""
rag/vectorstore.py

ماژول Embedding و FAISS Vector Store برای پروژه «امدادیار رضوی».

این ماژول مسئول موارد زیر است:
    1. تبدیل متن Chunkها (یا Query کاربر) به بردار عددی (Embedding)
    2. ساخت ایندکس FAISS از روی بردارهای Embedding
    3. ذخیره و بارگذاری ایندکس و متادیتای مرتبط روی دیسک
    4. اجرای جستجوی شباهت برداری (Similarity Search)

ورودی این ماژول خروجی rag/loader.py (لیستی از اشیاء Chunk) است و
خروجی آن (ایندکس FAISS + متادیتا) توسط rag/retriever.py مصرف می‌شود.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.loader import Chunk

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# مدل پیش‌فرض Embedding: مدل چندزبانه که از فارسی پشتیبانی می‌کند
DEFAULT_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# --------------------------------------------------------------------------- #
# خطاهای اختصاصی
# --------------------------------------------------------------------------- #
class VectorStoreError(Exception):
    """خطای عمومی مربوط به عملیات Vector Store."""


class IndexNotBuiltError(VectorStoreError):
    """خطای مربوط به تلاش برای جستجو یا ذخیره ایندکسی که هنوز ساخته نشده."""


# --------------------------------------------------------------------------- #
# EmbeddingModel
# --------------------------------------------------------------------------- #
class EmbeddingModel:
    """
    Wrapper روی مدل SentenceTransformer برای تولید Embedding متن.

    این کلاس هم برای Embedding کردن Chunkها در زمان Indexing و هم برای
    Embedding کردن پرسش کاربر در زمان Retrieval استفاده می‌شود، تا هر دو
    طرف در یک فضای برداری یکسان قرار داشته باشند.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_NAME) -> None:
        """
        Args:
            model_name: نام مدل SentenceTransformer قابل بارگذاری از HuggingFace Hub.
        """
        logger.info("در حال بارگذاری مدل Embedding: %s ...", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info("مدل Embedding بارگذاری شد. ابعاد بردار: %d", self.dimension)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        تبدیل لیستی از متن‌ها به بردارهای Embedding نرمال‌شده (برای شباهت کسینوسی).

        Args:
            texts: لیست متن‌های ورودی.
            batch_size: اندازه Batch برای پردازش دسته‌ای.

        Returns:
            آرایه Numpy با شکل (n_texts, dimension) و نوع float32، نرمال‌شده با L2.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # برای استفاده از Inner Product به‌جای Cosine مستقیم
        )
        return embeddings.astype("float32")

    def encode_single(self, text: str) -> np.ndarray:
        """تبدیل یک متن واحد (مثلاً پرسش کاربر) به بردار Embedding."""
        return self.encode([text])[0]


# --------------------------------------------------------------------------- #
# FAISSVectorStore
# --------------------------------------------------------------------------- #
class FAISSVectorStore:
    """
    مدیریت ایندکس FAISS و متادیتای Chunkهای مرتبط با آن.

    ساختار ذخیره‌سازی روی دیسک:
        {index_dir}/index.faiss     -> ایندکس باینری FAISS
        {index_dir}/metadata.pkl    -> لیست دیکشنری متادیتای هر Chunk (هم‌راستا با ترتیب ایندکس)
        {index_dir}/config.json     -> اطلاعات مدل Embedding و ابعاد بردار برای اعتبارسنجی سازگاری
    """

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.pkl"
    CONFIG_FILENAME = "config.json"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        index_dir: Union[str, Path],
    ) -> None:
        """
        Args:
            embedding_model: نمونه از EmbeddingModel برای تولید بردارها.
            index_dir: مسیر پوشه‌ای که ایندکس و متادیتا در آن ذخیره/بارگذاری می‌شود.
        """
        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._index: Optional[faiss.Index] = None
        self._metadata: List[dict] = []

    # ----------------------------------------------------------------- #
    # ساخت ایندکس
    # ----------------------------------------------------------------- #
    def build_index(self, chunks: List[Chunk]) -> None:
        """
        ساخت ایندکس FAISS از ابتدا با استفاده از لیستی از Chunkها.

        Args:
            chunks: خروجی PDFLoader.load_all_pdfs().

        Raises:
            VectorStoreError: اگر لیست chunks خالی باشد.
        """
        if not chunks:
            raise VectorStoreError("لیست Chunkها خالی است؛ امکان ساخت ایندکس وجود ندارد.")

        logger.info("در حال تولید Embedding برای %d Chunk ...", len(chunks))
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        logger.info("در حال ساخت ایندکس FAISS (IndexFlatIP) با ابعاد %d ...", self.embedding_model.dimension)
        index = faiss.IndexFlatIP(self.embedding_model.dimension)
        index.add(embeddings)

        self._index = index
        self._metadata = [chunk.to_dict() for chunk in chunks]

        logger.info("ایندکس FAISS با موفقیت ساخته شد. تعداد بردارها: %d", index.ntotal)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        افزودن تدریجی Chunkهای جدید به ایندکس موجود (بدون Re-index کامل).

        Args:
            chunks: لیست Chunkهای جدید برای افزودن.

        Raises:
            IndexNotBuiltError: اگر ایندکسی از قبل ساخته یا بارگذاری نشده باشد.
        """
        if self._index is None:
            raise IndexNotBuiltError(
                "ایندکس هنوز ساخته نشده است؛ ابتدا build_index یا load را فراخوانی کنید."
            )
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        self._index.add(embeddings)
        self._metadata.extend(chunk.to_dict() for chunk in chunks)

        logger.info("تعداد %d Chunk جدید به ایندکس اضافه شد. مجموع فعلی: %d", len(chunks), self._index.ntotal)

    # ----------------------------------------------------------------- #
    # جستجو
    # ----------------------------------------------------------------- #
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        جستجوی Top-K نزدیک‌ترین بردار به query_vector در ایندکس.

        Args:
            query_vector: بردار Embedding پرسش (باید با encode_single تولید شده باشد).
            top_k: تعداد نتایج مورد نیاز.

        Returns:
            لیستی از تاپل‌های (متادیتای Chunk، امتیاز شباهت) مرتب‌شده بر اساس بیشترین شباهت.

        Raises:
            IndexNotBuiltError: اگر ایندکس هنوز آماده نباشد.
        """
        if self._index is None:
            raise IndexNotBuiltError(
                "ایندکس هنوز ساخته یا بارگذاری نشده است؛ ابتدا build_index یا load را فراخوانی کنید."
            )
        if self._index.ntotal == 0:
            return []

        query_vector = query_vector.reshape(1, -1).astype("float32")
        top_k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(query_vector, top_k)

        results: List[Tuple[dict, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self._metadata[idx], float(score)))

        return results

    # ----------------------------------------------------------------- #
    # ذخیره‌سازی و بارگذاری
    # ----------------------------------------------------------------- #
    def save(self) -> None:
        """ذخیره ایندکس FAISS، متادیتا و تنظیمات مدل روی دیسک."""
        if self._index is None:
            raise IndexNotBuiltError("ایندکسی برای ذخیره‌سازی وجود ندارد.")

        index_path = self.index_dir / self.INDEX_FILENAME
        metadata_path = self.index_dir / self.METADATA_FILENAME
        config_path = self.index_dir / self.CONFIG_FILENAME

        faiss.write_index(self._index, str(index_path))

        with open(metadata_path, "wb") as f:
            pickle.dump(self._metadata, f)

        config = {
            "embedding_model_name": self.embedding_model.model_name,
            "dimension": self.embedding_model.dimension,
            "total_vectors": self._index.ntotal,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logger.info("ایندکس، متادیتا و تنظیمات با موفقیت در '%s' ذخیره شدند.", self.index_dir)

    def load(self) -> None:
        """
        بارگذاری ایندکس FAISS و متادیتای ذخیره‌شده از دیسک.

        Raises:
            VectorStoreError: اگر فایل‌های لازم یافت نشوند یا مدل Embedding
                با مدلی که ایندکس با آن ساخته شده هم‌خوانی نداشته باشد.
        """
        index_path = self.index_dir / self.INDEX_FILENAME
        metadata_path = self.index_dir / self.METADATA_FILENAME
        config_path = self.index_dir / self.CONFIG_FILENAME

        if not index_path.exists() or not metadata_path.exists():
            raise VectorStoreError(
                f"فایل‌های ایندکس در مسیر '{self.index_dir}' یافت نشدند. "
                "ابتدا باید Pipeline ساخت ایندکس اجرا شود."
            )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if config.get("embedding_model_name") != self.embedding_model.model_name:
                logger.warning(
                    "هشدار: مدل Embedding فعلی ('%s') با مدلی که ایندکس با آن ساخته شده "
                    "('%s') یکسان نیست. نتایج ممکن است نامعتبر باشند.",
                    self.embedding_model.model_name,
                    config.get("embedding_model_name"),
                )

        self._index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self._metadata = pickle.load(f)

        logger.info(
            "ایندکس از '%s' بارگذاری شد. تعداد بردارها: %d",
            self.index_dir,
            self._index.ntotal,
        )

    # ----------------------------------------------------------------- #
    # ویژگی‌های کمکی
    # ----------------------------------------------------------------- #
    @property
    def is_ready(self) -> bool:
        """بررسی اینکه آیا ایندکس ساخته/بارگذاری شده و آماده جستجو است یا خیر."""
        return self._index is not None and self._index.ntotal > 0

    @property
    def total_vectors(self) -> int:
        """تعداد کل بردارهای موجود در ایندکس."""
        return self._index.ntotal if self._index is not None else 0


# --------------------------------------------------------------------------- #
# اجرای مستقل جهت تست سریع (python -m rag.vectorstore)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    from rag.loader import PDFLoader

    kb_dir = sys.argv[1] if len(sys.argv) > 1 else "knowledge_base"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/vector_store"

    loader = PDFLoader(knowledge_base_dir=kb_dir)
    all_chunks = loader.load_all_pdfs()

    embedder = EmbeddingModel()
    store = FAISSVectorStore(embedding_model=embedder, index_dir=out_dir)
    store.build_index(all_chunks)
    store.save()

    print(f"ایندکس با {store.total_vectors} بردار در '{out_dir}' ذخیره شد.")
