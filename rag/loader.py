"""PDF ingestion with Persian-aware paragraph and sentence chunking."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable

import pymupdf

from rag.models import Chunk

logger = logging.getLogger(__name__)


class KnowledgeBaseError(Exception): pass


class PDFLoader:
    def __init__(self, knowledge_base_dir: Path | str, chunk_size: int = 900,
                 chunk_overlap: int = 140, min_chunk_length: int = 60, max_pdf_bytes: int = 25_000_000,
                 max_pages_per_pdf: int = 500, max_pdf_count: int = 100) -> None:
        if not (100 <= chunk_size <= 10_000 and 0 <= chunk_overlap < chunk_size and 1 <= min_chunk_length <= chunk_size):
            raise ValueError("تنظیمات chunking نامعتبر است.")
        if min(max_pdf_bytes, max_pages_per_pdf, max_pdf_count) < 1: raise ValueError("محدودیت‌های PDF نامعتبر است.")
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.chunk_size, self.chunk_overlap, self.min_chunk_length = chunk_size, chunk_overlap, min_chunk_length
        self.max_pdf_bytes, self.max_pages_per_pdf, self.max_pdf_count = max_pdf_bytes, max_pages_per_pdf, max_pdf_count

    def load_all_pdfs(self) -> list[Chunk]:
        root = self.knowledge_base_dir.resolve()
        paths = [path for path in sorted(root.glob("*.pdf")) if path.is_file() and path.resolve().parent == root]
        if not paths: raise KnowledgeBaseError("هیچ فایل PDF در پوشهٔ پایگاه دانش یافت نشد.")
        if len(paths) > self.max_pdf_count: raise KnowledgeBaseError("تعداد PDFهای پایگاه دانش از حد مجاز بیشتر است.")
        chunks: list[Chunk] = []
        failures: list[str] = []
        for path in paths:
            try: chunks.extend(self.load_single_pdf(path))
            except (KnowledgeBaseError, OSError, RuntimeError, pymupdf.FileDataError) as error:
                logger.warning("Skipping invalid PDF %s: %s", path.name, type(error).__name__)
                failures.append(path.name)
        if not chunks: raise KnowledgeBaseError("متن قابل استفاده‌ای از PDFها استخراج نشد: " + "; ".join(failures))
        return chunks

    def load_single_pdf(self, path: Path | str) -> list[Chunk]:
        path = Path(path)
        if path.suffix.lower() != ".pdf" or not path.is_file() or path.stat().st_size > self.max_pdf_bytes:
            raise KnowledgeBaseError("فایل PDF از سیاست ingestion عبور نکرد.")
        chunks: list[Chunk] = []
        with pymupdf.open(path) as document:
            if document.needs_pass or document.page_count > self.max_pages_per_pdf:
                raise KnowledgeBaseError("PDF رمزگذاری‌شده است یا صفحات آن بیش از حد مجاز است.")
            for page_no, page in enumerate(document, 1):
                text = self.clean_text(page.get_text("text"))
                chunks.extend(self._chunk_page(text, path.name, page_no))
        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return re.sub(r"[ \t]+", " ", text).strip()

    def _chunk_page(self, text: str, source: str, page_no: int) -> list[Chunk]:
        if len(text) < self.min_chunk_length: return []
        units = self._semantic_units(text)
        pieces: list[str] = []
        current = ""
        for unit in units:
            if current and len(current) + len(unit) + 1 > self.chunk_size:
                pieces.append(current.strip())
                current = current[-self.chunk_overlap:] + " " + unit
            else: current = f"{current} {unit}".strip()
        if current.strip(): pieces.append(current.strip())
        return [Chunk(self._id(source, page_no, index, value), value, source, page_no, index)
                for index, value in enumerate(pieces) if len(value) >= self.min_chunk_length]

    @staticmethod
    def _semantic_units(text: str) -> Iterable[str]:
        paragraphs = re.split(r"\n+", text)
        for paragraph in paragraphs:
            for sentence in re.split(r"(?<=[.!؟])\s+", paragraph):
                if sentence.strip(): yield sentence.strip()

    @staticmethod
    def _id(source: str, page: int, index: int, text: str) -> str:
        return hashlib.sha256(f"{source}:{page}:{index}:{text}".encode()).hexdigest()[:24]
