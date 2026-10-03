from __future__ import annotations
import pymupdf
from rag.loader import PDFLoader

def test_loader_extracts_and_chunks_persian_pdf(tmp_path):
    pdf = tmp_path / "guide.pdf"
    document = pymupdf.open(); page = document.new_page()
    page.insert_text((72, 72), "این یک متن آموزشی امداد است. " * 30)
    document.save(pdf); document.close()
    chunks = PDFLoader(tmp_path, chunk_size=120, chunk_overlap=20).load_all_pdfs()
    assert len(chunks) > 1
    assert all(chunk.source_file == "guide.pdf" and chunk.page_number == 1 for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)

def test_bad_pdf_is_skipped_when_a_valid_source_exists(tmp_path):
    valid = tmp_path / "guide.pdf"; bad = tmp_path / "bad.pdf"; bad.write_bytes(b"not a pdf")
    document = pymupdf.open(); page = document.new_page(); page.insert_text((72, 72), "امداد " * 40); document.save(valid); document.close()
    chunks = PDFLoader(tmp_path).load_all_pdfs()
    assert chunks and all(chunk.source_file == "guide.pdf" for chunk in chunks)
