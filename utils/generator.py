"""
utils/generator.py

هسته اصلی Backend پروژه «امدادیار رضوی» — مدیریت کامل RAG (بازیابی از
فایل‌های PDF) از طریق LangChain و بازگرداندن پاسخ مستقیماً از منابع.

الزامات فنی رعایت‌شده در این فایل:
    - مدیریت Vector Store و Retrieval از طریق LangChain
      (langchain_community.vectorstores.FAISS)
    - Retrieval واقعی بر پایه FAISS (LangChain به‌صورت داخلی از کتابخانه FAISS استفاده می‌کند)
    - خواندن فایل‌های PDF صرفاً با PyMuPDF (از طریق بازاستفاده از rag/loader.py
      که در مرحله قبل با PyMuPDF پیاده‌سازی شد؛ از هیچ Loader دیگری استفاده نمی‌شود)
    - محدودسازی کامل منبع داده به فایل‌های PDF داخل پوشه knowledge_base
    - بازگرداندن دقیق پیام ثابت زیر در صورت نبود اطلاعات کافی:
      "پاسخی برای این سوال در منابع موجود یافت نشد."

به‌روزرسانی مهم (حذف کامل هوش مصنوعی از مسیر پاسخ‌دهی):
    طبق الزام جدید پروژه، این دستیار دیگر از هیچ مدل زبانی (LLM) استفاده
    نمی‌کند و هیچ پرسشی به سرویس هوش مصنوعی (Anthropic یا هر سرویس دیگر)
    ارسال نمی‌شود. جریان کامل به شکل زیر است:

        ۱. دریافت پرسش کاربر
        ۲. جستجو (Retrieval) در ایندکس FAISS ساخته‌شده از فایل‌های PDF
           پوشه knowledge_base
        ۳. اگر نتیجه‌ای یافت شود -> نمایش مستقیم متن همان بخش‌های
           بازیابی‌شده (به‌همراه نام فایل و شماره صفحه) به‌عنوان پاسخ
        ۴. اگر هیچ نتیجه‌ای یافت نشود -> بازگرداندن پیام ثابت:
           "پاسخی برای این سوال در منابع موجود یافت نشد."

    به همین دلیل، در این نسخه هیچ وابستگی‌ای به ChatAnthropic، کلید
    ANTHROPIC_API_KEY، یا بسته‌های langchain-anthropic / anthropic وجود
    ندارد. کلاس RequestClassifier (تشخیص نوع QA/SCENARIO/QUIZ) همچنان
    حفظ شده و صرفاً برای اطلاع‌رسانی نوع درخواست در خروجی استفاده می‌شود؛
    اما دیگر برای انتخاب/ساخت Prompt به کار نمی‌رود، چون دیگر Promptی
    ساخته یا به جایی ارسال نمی‌شود.

نکته سازگاری معماری:
    ماژول rag/vectorstore.py (مرحله قبل) یک پیاده‌سازی FAISS خام و سبک‌وزن
    است که مستقیماً توسط rag/retriever.py و رابط کاربری Streamlit استفاده
    می‌شود. این فایل (utils/generator.py) طبق الزام صریح این مرحله، یک
    ایندکس FAISS مدیریت‌شده توسط LangChain می‌سازد و آن را در مسیر جداگانه
    data/vector_store_langchain/ ذخیره می‌کند تا با ایندکس خام قبلی
    (data/vector_store/) تداخلی ایجاد نشود. هر دو ایندکس از همان Chunkهای
    استخراج‌شده توسط rag/loader.py (PyMuPDF) ساخته می‌شوند و بنابراین از
    نظر محتوایی کاملاً سازگار و هم‌راستا هستند.

جریان داده در این فایل:
    rag.loader.PDFLoader (PyMuPDF)
        -> List[Chunk]
        -> تبدیل به langchain.schema.Document
        -> langchain_community.vectorstores.FAISS.from_documents(...)
        -> retriever.invoke(query)
        -> utils.classifier (فقط برای تشخیص و گزارش نوع درخواست)
        -> پاسخ نهایی مستقیماً از متن Chunkهای بازیابی‌شده ساخته می‌شود
           (بدون فراخوانی هیچ مدل هوش مصنوعی) یا پیام ثابت Fallback
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_core.documents import Document

from generation.intent_classifier import RequestType
from generation.prompt_templates import NOT_FOUND_MESSAGE
from rag.loader import Chunk, KnowledgeBaseNotFoundError, PDFLoader, PDFLoadError
from utils.classifier import RequestClassifier, get_default_classifier

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# --------------------------------------------------------------------------- #
# تنظیمات پیش‌فرض
# --------------------------------------------------------------------------- #
DEFAULT_KNOWLEDGE_BASE_DIR = "knowledge_base"
DEFAULT_LANGCHAIN_INDEX_DIR = "data/vector_store_langchain"
DEFAULT_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 5
DEFAULT_NUM_QUESTIONS = 5


# --------------------------------------------------------------------------- #
# خطاهای اختصاصی
# --------------------------------------------------------------------------- #
class GenerationError(Exception):
    """خطای عمومی مربوط به فرآیند تولید پاسخ."""


class IndexNotReadyError(GenerationError):
    """خطای مربوط به تلاش برای Retrieval پیش از آماده‌شدن ایندکس."""


class EmptyQueryError(GenerationError):
    """خطای مربوط به دریافت پرسش خالی یا فقط شامل فاصله."""


# --------------------------------------------------------------------------- #
# RAGGenerator
# --------------------------------------------------------------------------- #
class RAGGenerator:
    """
    هسته Backend که کل جریان بازیابی (بارگذاری PDF، Embedding، Retrieval با
    FAISS از طریق LangChain) را مدیریت می‌کند و پاسخ را مستقیماً از متن
    بازیابی‌شده فایل‌های PDF می‌سازد — بدون هیچ فراخوانی مدل هوش مصنوعی.
    """

    def __init__(
        self,
        knowledge_base_dir: Union[str, Path] = DEFAULT_KNOWLEDGE_BASE_DIR,
        index_dir: Union[str, Path] = DEFAULT_LANGCHAIN_INDEX_DIR,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        classifier: Optional[RequestClassifier] = None,
    ) -> None:
        """
        Args:
            knowledge_base_dir: مسیر پوشه حاوی فایل‌های PDF منبع دانش (تنها منبع مجاز).
            index_dir: مسیر ذخیره/بارگذاری ایندکس FAISS مدیریت‌شده توسط LangChain.
            embedding_model_name: نام مدل Embedding چندزبانه (سازگار با rag/vectorstore.py).
            classifier: نمونه RequestClassifier؛ در صورت عدم ارائه از نمونه پیش‌فرض مشترک استفاده می‌شود.
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model_name
        self.classifier: RequestClassifier = classifier or get_default_classifier()

        logger.info("در حال بارگذاری مدل Embedding (LangChain): %s ...", self.embedding_model_name)
        self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self._vectorstore: Optional[LangChainFAISS] = None

    # ----------------------------------------------------------------- #
    # ساخت / ذخیره / بارگذاری ایندکس
    # ----------------------------------------------------------------- #
    def build_index(self) -> None:
        """
        ساخت ایندکس FAISS مدیریت‌شده توسط LangChain از تمام PDFهای knowledge_base.

        این متد از rag.loader.PDFLoader (مبتنی بر PyMuPDF) برای استخراج و
        Chunk‌کردن متن استفاده می‌کند، سپس هر Chunk را به یک شیء
        langchain_core.documents.Document تبدیل کرده و ایندکس FAISS را می‌سازد.

        Raises:
            GenerationError: در صورت شکست کامل فرآیند بارگذاری PDFها.
        """
        loader = PDFLoader(knowledge_base_dir=self.knowledge_base_dir)
        try:
            chunks = loader.load_all_pdfs()
        except (KnowledgeBaseNotFoundError, PDFLoadError) as exc:
            raise GenerationError(f"ساخت ایندکس با شکست مواجه شد: {exc}") from exc

        documents = self._chunks_to_documents(chunks)
        logger.info("در حال ساخت ایندکس FAISS (LangChain) از %d سند ...", len(documents))

        self._vectorstore = LangChainFAISS.from_documents(documents, self._embeddings)
        logger.info("ایندکس LangChain FAISS با موفقیت ساخته شد.")

    def save_index(self) -> None:
        """ذخیره ایندکس LangChain FAISS روی دیسک (مسیر index_dir)."""
        if self._vectorstore is None:
            raise IndexNotReadyError("ایندکسی برای ذخیره‌سازی وجود ندارد؛ ابتدا build_index را اجرا کنید.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(self.index_dir))
        logger.info("ایندکس LangChain FAISS در مسیر '%s' ذخیره شد.", self.index_dir)

    def load_index(self) -> None:
        """
        بارگذاری ایندکس LangChain FAISS از دیسک.

        Raises:
            IndexNotReadyError: اگر فایل‌های ایندکس در مسیر index_dir یافت نشوند.
        """
        if not self.index_dir.exists():
            raise IndexNotReadyError(
                f"مسیر ایندکس '{self.index_dir}' یافت نشد. ابتدا باید build_index و save_index اجرا شوند."
            )

        # allow_dangerous_deserialization: چون ایندکس توسط خود سیستم ساخته و
        # ذخیره شده (نه منبع خارجی نامعتبر)، بارگذاری آن امن است.
        self._vectorstore = LangChainFAISS.load_local(
            str(self.index_dir),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("ایندکس LangChain FAISS از مسیر '%s' بارگذاری شد.", self.index_dir)

    def ensure_ready(self) -> None:
        """
        اطمینان از آماده‌بودن ایندکس: تلاش برای بارگذاری از دیسک و در صورت
        نبود، ساخت و ذخیره ایندکس جدید از ابتدا.
        """
        if self.is_ready:
            return
        try:
            self.load_index()
        except IndexNotReadyError:
            logger.info("ایندکس ذخیره‌شده یافت نشد؛ در حال ساخت ایندکس جدید از knowledge_base ...")
            self.build_index()
            self.save_index()

    @property
    def is_ready(self) -> bool:
        """بررسی اینکه آیا ایندکس ساخته/بارگذاری شده و آماده Retrieval است."""
        return self._vectorstore is not None

    # ----------------------------------------------------------------- #
    # API عمومی تولید پاسخ
    # ----------------------------------------------------------------- #
    def generate(
        self,
        user_input: str,
        top_k: int = DEFAULT_TOP_K,
        num_questions: int = DEFAULT_NUM_QUESTIONS,
    ) -> str:
        """
        بازگرداندن پاسخ نهایی (متن ساده) برای یک درخواست کاربر.

        Args:
            user_input: متن خام درخواست/پرسش کاربر.
            top_k: تعداد Chunkهای مورد بازیابی از FAISS.
            num_questions: در این نسخه استفاده نمی‌شود (چون Promptی ساخته
                نمی‌شود)؛ صرفاً برای حفظ سازگاری امضای متد نگه داشته شده است.

        Returns:
            رشته پاسخ نهایی (متن مستقیم بازیابی‌شده از PDFها)؛ در صورت نبود
            اطلاعات کافی، دقیقاً همان پیام ثابت
            "پاسخی برای این سوال در منابع موجود یافت نشد." بازگردانده می‌شود.
        """
        answer, _docs, _request_type = self._run(user_input, top_k, num_questions)
        return answer

    def generate_with_sources(
        self,
        user_input: str,
        top_k: int = DEFAULT_TOP_K,
        num_questions: int = DEFAULT_NUM_QUESTIONS,
    ) -> Dict[str, Any]:
        """
        بازگرداندن پاسخ نهایی به‌همراه جزئیات منابع بازیابی‌شده و نوع درخواست
        (مناسب برای مصرف در لایه API یا رابط کاربری).

        Args:
            user_input: متن خام درخواست/پرسش کاربر.
            top_k: تعداد Chunkهای مورد بازیابی از FAISS.
            num_questions: در این نسخه استفاده نمی‌شود؛ صرفاً برای حفظ سازگاری
                امضای متد نگه داشته شده است.

        Returns:
            دیکشنری شامل کلیدهای:
                answer: متن پاسخ نهایی (مستقیماً از PDFها) یا پیام Fallback
                is_answerable: bool
                request_type: یکی از "QA" / "SCENARIO" / "QUIZ" یا None (اگر پرسش پاسخ‌داده‌نشد)
                sources: لیستی از دیکشنری‌های {source_file, page_number, chunk_index}
        """
        answer, docs, request_type = self._run(user_input, top_k, num_questions)

        sources = [
            {
                "source_file": doc.metadata.get("source_file"),
                "page_number": doc.metadata.get("page_number"),
                "chunk_index": doc.metadata.get("chunk_index"),
            }
            for doc in docs
        ]

        return {
            "answer": answer,
            "is_answerable": answer != NOT_FOUND_MESSAGE,
            "request_type": request_type.value if request_type else None,
            "sources": sources,
        }

    # ----------------------------------------------------------------- #
    # منطق داخلی مشترک
    # ----------------------------------------------------------------- #
    def _run(
        self,
        user_input: str,
        top_k: int,
        num_questions: int,
    ) -> Tuple[str, List[Document], Optional[RequestType]]:
        """
        اجرای کامل جریان: اعتبارسنجی پرسش، بازیابی از FAISS، تشخیص نوع
        درخواست (فقط برای گزارش) و ساخت پاسخ مستقیم از متن بازیابی‌شده.

        نکته مهم: این متد هیچ پرسشی به مدل هوش مصنوعی ارسال نمی‌کند. پاسخ
        نهایی همان متن Chunkهای بازیابی‌شده از فایل‌های PDF است.

        Args:
            user_input: متن خام درخواست/پرسش کاربر.
            top_k: تعداد Chunkهای مورد بازیابی.
            num_questions: در این نسخه استفاده نمی‌شود (بدون کاربرد عملی)؛
                صرفاً برای حفظ سازگاری امضای generate/generate_with_sources
                نگه داشته شده است.

        Returns:
            تاپل (پاسخ نهایی، اسناد بازیابی‌شده، نوع درخواست تشخیص‌داده‌شده).
            در صورت عدم بازیابی هیچ سندی، نوع درخواست همچنان بازگردانده می‌شود
            ولی لیست اسناد خالی و پاسخ برابر NOT_FOUND_MESSAGE خواهد بود.
        """
        cleaned_input = (user_input or "").strip()
        if not cleaned_input:
            raise EmptyQueryError("متن درخواست نمی‌تواند خالی باشد.")

        if not self.is_ready:
            raise IndexNotReadyError(
                "ایندکس آماده نیست. ابتدا ensure_ready() یا load_index() را فراخوانی کنید."
            )

        request_type = self.classifier.classify(cleaned_input)
        logger.info("نوع درخواست تشخیص داده شد: %s", request_type.value)

        retrieved_docs = self._retrieve(cleaned_input, top_k=top_k)

        # --- قانون اصلی: عدم وجود منبع مرتبط -> بازگرداندن فوری پیام ثابت --- #
        if not retrieved_docs:
            logger.warning("هیچ سند مرتبطی در knowledge_base یافت نشد؛ بازگرداندن پیام Fallback.")
            return NOT_FOUND_MESSAGE, [], request_type

        # --- ساخت پاسخ مستقیم از متن بازیابی‌شده، بدون فراخوانی هیچ مدل هوش مصنوعی --- #
        final_answer = self._build_direct_answer(retrieved_docs)

        return final_answer, retrieved_docs, request_type

    def _retrieve(self, query: str, top_k: int) -> List[Document]:
        """
        بازیابی Top-K سند مرتبط از ایندکس FAISS مدیریت‌شده توسط LangChain.

        Args:
            query: متن پرسش (پس از پاکسازی).
            top_k: تعداد اسناد مورد نیاز.

        Returns:
            لیستی از اشیاء langchain_core.documents.Document.
        """
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": top_k})
        try:
            # رابط جدید LangChain (Runnable Interface)
            return retriever.invoke(query)
        except AttributeError:
            # سازگاری با نسخه‌های قدیمی‌تر LangChain
            return retriever.get_relevant_documents(query)

    @staticmethod
    def _chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
        """
        تبدیل خروجی rag.loader.PDFLoader (لیست Chunk) به اسناد سازگار با LangChain.

        متادیتای هر Chunk (نام فایل، شماره صفحه، شماره ترتیبی) در متادیتای
        Document حفظ می‌شود تا در Retrieval و نمایش منبع پاسخ قابل استفاده باشد.
        """
        return [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk in chunks
        ]

    @staticmethod
    def _build_direct_answer(docs: List[Document]) -> str:
        """
        ساخت پاسخ نهایی مستقیماً از متن Chunkهای بازیابی‌شده از فایل‌های PDF،
        بدون هیچ فراخوانی مدل هوش مصنوعی (LLM).

        هر بخش از پاسخ دقیقاً همان متن استخراج‌شده از knowledge_base است و
        پیش از آن برچسب منبع (نام فایل PDF و شماره صفحه) درج می‌شود تا
        کاربر بداند هر بخش از کدام فایل و چه صفحه‌ای آمده است.

        Args:
            docs: لیست اسناد بازیابی‌شده (خروجی self._retrieve).

        Returns:
            رشته متنی نهایی آماده نمایش به‌عنوان پاسخ.
        """
        parts: List[str] = []
        for i, doc in enumerate(docs, start=1):
            header = (
                f"[منبع {i}: {doc.metadata.get('source_file')} - "
                f"صفحه {doc.metadata.get('page_number')}]"
            )
            parts.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------- #
# اجرای مستقل جهت تست سریع (python -m utils.generator)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    test_query = sys.argv[1] if len(sys.argv) > 1 else "اقدامات اولیه در زلزله چیست؟"

    generator = RAGGenerator()
    generator.ensure_ready()

    result = generator.generate_with_sources(test_query)

    print(f"پرسش: {test_query}")
    print(f"نوع درخواست: {result['request_type']}")
    print(f"قابل پاسخ‌گویی: {result['is_answerable']}\n")
    print("--- پاسخ ---")
    print(result["answer"])

    if result["sources"]:
        print("\n--- منابع ---")
        for src in result["sources"]:
            print(f"- {src['source_file']} | صفحه {src['page_number']}")
