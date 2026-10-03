"""Safe FAISS persistence with atomic writes, integrity metadata and a file lock."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import faiss
import numpy as np

from rag.models import Chunk, SearchResult

class Embeddings(Protocol):
    dimension: int
    model_name: str
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(values, dtype="float32")

def knowledge_base_fingerprint(directory: Path) -> str:
    directory = directory.resolve()
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.pdf")):
        if not path.is_file() or path.resolve().parent != directory: continue
        digest.update(path.name.encode("utf-8")); digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()

def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class FaissIndex:
    INDEX_NAME, METADATA_NAME, MANIFEST_NAME, LOCK_NAME = "index.faiss", "metadata.json", "manifest.json", ".index.lock"
    def __init__(self, directory: Path, embeddings: Embeddings, index_config: dict[str, int]) -> None:
        self.directory, self.embeddings, self.index_config = directory.resolve(), embeddings, index_config
        self._index: faiss.Index | None = None; self._chunks: list[Chunk] = []

    def is_current(self, fingerprint: str) -> bool:
        manifest = self._read_manifest()
        return bool(manifest and manifest.get("fingerprint") == fingerprint and manifest.get("model") == self.embeddings.model_name
                    and manifest.get("dimension") == self.embeddings.dimension and manifest.get("index_config") == self.index_config
                    and self._files_match_manifest(manifest))

    def build(self, chunks: Sequence[Chunk], fingerprint: str) -> None:
        if not chunks: raise ValueError("هیچ متن قابل ایندکس‌گذاری در منابع یافت نشد.")
        vectors = self.embeddings.encode([chunk.text for chunk in chunks])
        if vectors.shape != (len(chunks), self.embeddings.dimension): raise ValueError("ابعاد embedding نامعتبر است.")
        index = faiss.IndexFlatIP(self.embeddings.dimension); index.add(vectors)
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.directory.parent, prefix=".index-") as temp_dir:
            temp = Path(temp_dir); index_file, metadata_file = temp / self.INDEX_NAME, temp / self.METADATA_NAME
            faiss.write_index(index, str(index_file))
            metadata_file.write_text(json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False), encoding="utf-8")
            manifest = {"version": 2, "fingerprint": fingerprint, "model": self.embeddings.model_name,
                        "dimension": self.embeddings.dimension, "chunk_count": len(chunks), "index_config": self.index_config,
                        "index_sha256": _hash_file(index_file), "metadata_sha256": _hash_file(metadata_file)}
            manifest_file = temp / self.MANIFEST_NAME
            manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            for name in (self.INDEX_NAME, self.METADATA_NAME, self.MANIFEST_NAME): os.replace(temp / name, self.directory / name)
        self._index, self._chunks = index, list(chunks)

    def load(self) -> None:
        manifest = self._read_manifest()
        if not manifest or not self._files_match_manifest(manifest): raise ValueError("صحت فایل‌های ایندکس تأیید نشد.")
        metadata = json.loads((self.directory / self.METADATA_NAME).read_text(encoding="utf-8"))
        if not isinstance(metadata, list): raise ValueError("metadata ایندکس نامعتبر است.")
        index = faiss.read_index(str(self.directory / self.INDEX_NAME)); chunks = [Chunk.from_dict(item) for item in metadata]
        if index.ntotal != len(chunks) or index.d != self.embeddings.dimension: raise ValueError("فایل‌های ایندکس سازگار نیستند.")
        self._index, self._chunks = index, chunks

    def search(self, query: str, top_k: int, threshold: float) -> list[SearchResult]:
        if self._index is None: raise RuntimeError("ایندکس آماده نیست.")
        vector = self.embeddings.encode([query]); scores, positions = self._index.search(vector, min(top_k, self._index.ntotal))
        return [SearchResult(self._chunks[int(pos)], float(score)) for score, pos in zip(scores[0], positions[0]) if pos >= 0 and float(score) >= threshold]

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / self.LOCK_NAME).open("w") as lock:
            try:
                try:
                    import fcntl
                except ImportError:  # pragma: no cover - Windows fallback; atomic writes still protect files.
                    fcntl = None
                if fcntl: fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                if 'fcntl' in locals() and fcntl: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read_manifest(self) -> dict[str, object] | None:
        try:
            value = json.loads((self.directory / self.MANIFEST_NAME).read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (FileNotFoundError, json.JSONDecodeError): return None

    def _files_match_manifest(self, manifest: dict[str, object]) -> bool:
        try:
            return _hash_file(self.directory / self.INDEX_NAME) == manifest.get("index_sha256") and _hash_file(self.directory / self.METADATA_NAME) == manifest.get("metadata_sha256")
        except FileNotFoundError: return False
