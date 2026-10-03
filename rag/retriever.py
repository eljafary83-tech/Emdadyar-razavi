"""Single retrieval path that refreshes an index safely before each request."""
from __future__ import annotations
from pathlib import Path
from rag.index import Embeddings, FaissIndex, knowledge_base_fingerprint
from rag.loader import PDFLoader
from rag.models import SearchResult

class Retriever:
    def __init__(self, knowledge_base_dir: Path, index_dir: Path, embeddings: Embeddings, top_k: int = 5,
                 threshold: float = 0.42, chunk_size: int = 900, chunk_overlap: int = 140, min_chunk_length: int = 60,
                 max_pdf_bytes: int = 25_000_000, max_pages_per_pdf: int = 500, max_pdf_count: int = 100) -> None:
        if top_k < 1 or not -1 <= threshold <= 1: raise ValueError("تنظیمات retrieval نامعتبر است.")
        self.knowledge_base_dir, self.top_k, self.threshold = knowledge_base_dir.resolve(), top_k, threshold
        self.loader_options = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "min_chunk_length": min_chunk_length,
                               "max_pdf_bytes": max_pdf_bytes, "max_pages_per_pdf": max_pages_per_pdf, "max_pdf_count": max_pdf_count}
        self.index = FaissIndex(index_dir, embeddings, {key: value for key, value in self.loader_options.items() if key.startswith("chunk_") or key == "min_chunk_length"})

    def ensure_ready(self) -> None:
        with self.index.locked():
            fingerprint = knowledge_base_fingerprint(self.knowledge_base_dir)
            if self.index.is_current(fingerprint): self.index.load(); return
            chunks = PDFLoader(self.knowledge_base_dir, **self.loader_options).load_all_pdfs()
            self.index.build(chunks, fingerprint)

    def retrieve(self, query: str) -> list[SearchResult]:
        cleaned = query.strip()
        if not cleaned: raise ValueError("متن درخواست نمی‌تواند خالی باشد.")
        self.ensure_ready()
        return self.index.search(cleaned, self.top_k, self.threshold)
