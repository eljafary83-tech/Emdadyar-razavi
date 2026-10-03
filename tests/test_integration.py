from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pymupdf

from llm.client import APIConfig, ChatCompletionsClient
from llm.provider import GenericChatProvider
from rag.retriever import Retriever
from services.assistant import AssistantService
from tests.conftest import FakeEmbeddings

class Handler(BaseHTTPRequestHandler):
    request_payload = None
    def do_POST(self):
        Handler.request_payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        answer = json.dumps({"answer": "پاسخ فقط از منبع", "citations": ["src_1"]})
        body = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_): pass

def test_pdf_to_faiss_to_mock_llm_to_validated_response(tmp_path):
    kb = tmp_path / "kb"; kb.mkdir(); pdf = kb / "guide.pdf"
    document = pymupdf.open(); page = document.new_page(); page.insert_text((72, 72), "امداد " * 60); document.save(pdf); document.close()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler); thread = threading.Thread(target=server.serve_forever); thread.start()
    try:
        retriever = Retriever(kb, tmp_path / "index", FakeEmbeddings(), threshold=-1, chunk_size=100, chunk_overlap=10, min_chunk_length=10)
        provider = GenericChatProvider(ChatCompletionsClient(APIConfig(f"http://127.0.0.1:{server.server_port}", "test-key", "test-model")))
        result = AssistantService(retriever, provider).answer("امداد")
    finally:
        server.shutdown(); thread.join()
    assert result.ok and result.citation_ids == ("src_1",)
    assert Handler.request_payload["messages"][0]["role"] == "system"
    assert "<SOURCE id='src_1'" in Handler.request_payload["messages"][1]["content"]
