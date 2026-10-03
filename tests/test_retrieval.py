from __future__ import annotations
import pymupdf
from rag.models import Chunk
from rag.index import FaissIndex
from rag.retriever import Retriever
from tests.conftest import FakeEmbeddings
def make_pdf(path, text):
    document = pymupdf.open(); page = document.new_page(); page.insert_text((72, 72), text * 20); document.save(path); document.close()
def test_threshold_filters_unrelated_results(tmp_path):
    index = FaissIndex(tmp_path / "index", FakeEmbeddings(), {"chunk_size": 100, "chunk_overlap": 10, "min_chunk_length": 10})
    index.build([Chunk("1", "آموزش امداد", "a.pdf", 1, 0), Chunk("2", "موضوع دیگر", "b.pdf", 1, 0)], "x")
    assert [x.chunk.id for x in index.search("امداد", 2, .8)] == ["1"]
def test_pdf_add_change_delete_rebuild_index(tmp_path):
    kb, output = tmp_path / "kb", tmp_path / "index"; kb.mkdir(); make_pdf(kb / "a.pdf", "امداد")
    retriever = Retriever(kb, output, FakeEmbeddings(), threshold=-1, chunk_size=100, chunk_overlap=10, min_chunk_length=10)
    retriever.retrieve("امداد"); first = retriever.index._index.ntotal
    (kb / "a.pdf").unlink(); make_pdf(kb / "a.pdf", "امداد تغییر یافته")
    retriever.retrieve("امداد"); assert retriever.index._index.ntotal == first
    make_pdf(kb / "b.pdf", "موضوع دیگر"); retriever.retrieve("امداد"); assert retriever.index._index.ntotal > first
    (kb / "a.pdf").unlink(); retriever.retrieve("امداد"); assert all(item.chunk.source_file == "b.pdf" for item in retriever.retrieve("امداد"))
def test_model_or_dimension_marks_index_stale(tmp_path):
    index = FaissIndex(tmp_path / "index", FakeEmbeddings(), {"chunk_size": 100, "chunk_overlap": 10, "min_chunk_length": 10})
    index.build([Chunk("1", "امداد", "a.pdf", 1, 0)], "f")
    class Other(FakeEmbeddings): model_name = "other"
    assert not FaissIndex(tmp_path / "index", Other(), {"chunk_size": 100, "chunk_overlap": 10, "min_chunk_length": 10}).is_current("f")
    class OtherDimension(FakeEmbeddings): dimension = 3
    assert not FaissIndex(tmp_path / "index", OtherDimension(), {"chunk_size": 100, "chunk_overlap": 10, "min_chunk_length": 10}).is_current("f")
