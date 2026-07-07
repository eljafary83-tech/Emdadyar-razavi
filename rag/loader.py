"""
rag/loader.py

ماژول PDF Loader و Chunking برای پروژه «امدادیار رضوی».

این ماژول مسئول موارد زیر است:
    1. خواندن تمام فایل‌های PDF موجود در پوشه knowledge_base با استفاده از PyMuPDF (fitz)
    2. استخراج متن به تفکیک صفحه
    3. پاکسازی متن استخراج‌شده
    4. تقسیم متن به Chunkهای هم‌پوشان (Sliding Window) به‌همراه متادیتا

خروجی این ماژول (لیستی از اشیاء Chunk) مستقیماً به‌عنوان ورودی به
ماژول rag/vectorstore.py برای Embedding و ساخت ایندکس FAISS داده می‌شود.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------- #
# مدل داده
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """
    ساختار داده‌ای واحد قابل بازیابی (یک تکه از متن یک PDF).

    Attributes:
        chunk_id: شناسه یکتای Chunk (برای هم‌راستایی با ایندکس FAISS).
        text: متن نهایی و پاکسازی‌شده Chunk.
        source_file: نام فایل PDF منبع.
        page_number: شماره صفحه‌ای که این Chunk از آن استخراج شده (از ۱ شروع می‌شود).
        chunk_index: شماره ترتیبی این Chunk در همان صفحه (از ۰ شروع می‌شود).
    """

    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """تبدیل Chunk به دیکشنری قابل سریالایز (برای ذخیره در metadata_store)."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """بازسازی شیء Chunk از دیکشنری ذخیره‌شده."""
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            source_file=data["source_file"],
            page_number=data["page_number"],
            chunk_index=data["chunk_index"],
            metadata=data.get("metadata", {}),
        )


# --------------------------------------------------------------------------- #
# خطاهای اختصاصی
# --------------------------------------------------------------------------- #
class PDFLoadError(Exception):
    """خطای مربوط به شکست در خواندن یا پردازش یک فایل PDF."""


class KnowledgeBaseNotFoundError(Exception):
    """خطای مربوط به عدم وجود پوشه knowledge_base یا خالی بودن آن."""


# --------------------------------------------------------------------------- #
# PDFLoader
# --------------------------------------------------------------------------- #
class PDFLoader:
    """
    بارگذاری و Chunk کردن فایل‌های PDF موجود در پوشه knowledge_base.

    این کلاس تنها منبع مجاز استخراج داده خام برای کل سیستم است؛ طبق
    محدودیت معماری، هیچ منبع دیگری غیر از PDFهای این پوشه نباید در
    Pipeline بازیابی مورد استفاده قرار گیرد.
    """

    def __init__(
        self,
        knowledge_base_dir: Union[str, Path],
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_length: int = 40,
    ) -> None:
        """
        Args:
            knowledge_base_dir: مسیر پوشه حاوی فایل‌های PDF منبع دانش.
            chunk_size: حداکثر طول هر Chunk بر حسب تعداد کاراکتر.
            chunk_overlap: میزان هم‌پوشانی بین Chunkهای متوالی بر حسب کاراکتر.
            min_chunk_length: حداقل طول مجاز برای یک Chunk (Chunkهای کوتاه‌تر حذف می‌شوند).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap باید کوچک‌تر از chunk_size باشد.")

        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    # ----------------------------------------------------------------- #
    # API عمومی
    # ----------------------------------------------------------------- #
    def load_all_pdfs(self) -> List[Chunk]:
        """
        تمام فایل‌های PDF موجود در knowledge_base_dir را می‌خواند و Chunk می‌کند.

        Returns:
            لیستی از تمام اشیاء Chunk استخراج‌شده از تمام PDFها.

        Raises:
            KnowledgeBaseNotFoundError: اگر پوشه وجود نداشته باشد یا هیچ PDF‌ای در آن نباشد.
        """
        if not self.knowledge_base_dir.exists() or not self.knowledge_base_dir.is_dir():
            raise KnowledgeBaseNotFoundError(
                f"پوشه knowledge_base در مسیر '{self.knowledge_base_dir}' یافت نشد."
            )

        pdf_paths = sorted(self.knowledge_base_dir.glob("*.pdf"))
        if not pdf_paths:
            raise KnowledgeBaseNotFoundError(
                f"هیچ فایل PDF‌ای در پوشه '{self.knowledge_base_dir}' یافت نشد."
            )

        all_chunks: List[Chunk] = []
        for pdf_path in pdf_paths:
            try:
                chunks = self._process_single_pdf(pdf_path)
                logger.info(
                    "فایل '%s' با موفقیت پردازش شد: %d Chunk استخراج شد.",
                    pdf_path.name,
                    len(chunks),
                )
                all_chunks.extend(chunks)
            except PDFLoadError as exc:
                logger.error("خطا در پردازش فایل '%s': %s", pdf_path.name, exc)
                # طبق سیاست سیستم، خطای یک فایل نباید کل Pipeline را متوقف کند
                continue

        if not all_chunks:
            raise PDFLoadError(
                "پردازش تمام فایل‌های PDF با شکست مواجه شد؛ هیچ Chunk‌ای تولید نشد."
            )

        logger.info("مجموع Chunkهای استخراج‌شده از knowledge_base: %d", len(all_chunks))
        return all_chunks

    def load_single_pdf(self, pdf_path: Union[str, Path]) -> List[Chunk]:
        """
        بارگذاری و Chunk کردن یک فایل PDF مشخص (برای افزودن تدریجی به Index).

        Args:
            pdf_path: مسیر فایل PDF.

        Returns:
            لیست Chunkهای استخراج‌شده از همان فایل.
        """
        return self._process_single_pdf(Path(pdf_path))

    # ----------------------------------------------------------------- #
    # متدهای داخلی
    # ----------------------------------------------------------------- #
    def _process_single_pdf(self, pdf_path: Path) -> List[Chunk]:
        """استخراج متن یک PDF و تبدیل آن به لیستی از Chunkها."""
        pages_text = self._extract_text_from_pdf(pdf_path)

        chunks: List[Chunk] = []
        for page_number, raw_text in pages_text:
            cleaned_text = self._clean_text(raw_text)
            if not cleaned_text:
                continue
            page_chunks = self._split_into_chunks(
                text=cleaned_text,
                source_file=pdf_path.name,
                page_number=page_number,
            )
            chunks.extend(page_chunks)

        return chunks

    def _extract_text_from_pdf(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """
        استخراج متن خام هر صفحه از یک فایل PDF با استفاده از PyMuPDF.

        Returns:
            لیستی از تاپل‌های (شماره صفحه، متن خام صفحه). شماره صفحه از ۱ شروع می‌شود.

        Raises:
            PDFLoadError: در صورت شکست در باز کردن یا خواندن فایل.
        """
        try:
            document = fitz.open(pdf_path)
        except Exception as exc:  # noqa: BLE001
            raise PDFLoadError(f"باز کردن فایل PDF با شکست مواجه شد: {exc}") from exc

        pages_text: List[Tuple[int, str]] = []
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text = page.get_text("text")
                pages_text.append((page_index + 1, text))
        except Exception as exc:  # noqa: BLE001
            raise PDFLoadError(f"خواندن محتوای صفحات PDF با شکست مواجه شد: {exc}") from exc
        finally:
            document.close()

        return pages_text

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        پاکسازی متن استخراج‌شده:
            - حذف فاصله‌ها و خطوط اضافی
            - حذف کاراکترهای کنترلی نامرئی
            - یکسان‌سازی فاصله‌های سفید
        """
        if not text:
            return ""

        # حذف کاراکترهای کنترلی به‌جز newline
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        # یکسان‌سازی چند فاصله پشت‌سرهم
        text = re.sub(r"[ \t]+", " ", text)
        # یکسان‌سازی چند خط خالی پشت‌سرهم
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def _split_into_chunks(
        self,
        text: str,
        source_file: str,
        page_number: int,
    ) -> List[Chunk]:
        """
        تقسیم متن یک صفحه به Chunkهای هم‌پوشان با استفاده از راهبرد Sliding Window.

        راهبرد Chunking:
            - طول هر Chunk حداکثر برابر self.chunk_size کاراکتر است.
            - بین Chunkهای متوالی به‌اندازه self.chunk_overlap کاراکتر هم‌پوشانی
              وجود دارد تا اطلاعات در مرز Chunkها از دست نرود.
            - Chunkهایی که طول آن‌ها کمتر از min_chunk_length باشد حذف می‌شوند
              (معمولاً باقیمانده‌های بی‌معنی انتهای متن).
        """
        chunks: List[Chunk] = []
        text_length = len(text)

        if text_length <= self.chunk_size:
            candidate = text.strip()
            if len(candidate) >= self.min_chunk_length:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=candidate,
                        source_file=source_file,
                        page_number=page_number,
                        chunk_index=0,
                    )
                )
            return chunks

        start = 0
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            candidate = text[start:end].strip()

            if len(candidate) >= self.min_chunk_length:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=candidate,
                        source_file=source_file,
                        page_number=page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            if end == text_length:
                break
            start += step

        return chunks


# --------------------------------------------------------------------------- #
# اجرای مستقل جهت تست سریع (python -m rag.loader)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    kb_dir = sys.argv[1] if len(sys.argv) > 1 else "knowledge_base"
    loader = PDFLoader(knowledge_base_dir=kb_dir)
    result_chunks = loader.load_all_pdfs()
    print(f"تعداد کل Chunkها: {len(result_chunks)}")
    if result_chunks:
        print("نمونه اولین Chunk:")
        print(result_chunks[0].to_dict())
