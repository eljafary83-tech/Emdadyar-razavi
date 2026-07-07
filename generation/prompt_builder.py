"""
generation/prompt_builder.py

ماژول منطق انتخاب Prompt Template و ساخت Prompt نهایی برای پروژه «امدادیار رضوی».

این ماژول نقطه اتصال سه بخش زیر است:
    1. rag/retriever.py           -> بازیابی Chunkهای مرتبط از FAISS
    2. generation/intent_classifier.py -> تشخیص نوع درخواست (QA/SCENARIO/QUIZ)
    3. generation/prompt_templates.py  -> قالب‌های آماده هر حالت

قانون اصلی «یافت نشدن اطلاعات» در همین ماژول و در سطح کد (نه فقط Prompt)
اجرا می‌شود: اگر Retriever هیچ Chunk مرتبطی بازنگرداند، اصلاً درخواستی
به LLM ارسال نمی‌شود و پیام ثابت Fallback مستقیماً بازگردانده می‌شود.
این رویکرد هم هزینه فراخوانی LLM را کاهش می‌دهد و هم یک لایه Guardrail
قطعی (غیرقابل‌دورزدن توسط مدل زبانی) ایجاد می‌کند.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from generation.intent_classifier import IntentClassifier, RequestType
from generation.prompt_templates import NOT_FOUND_MESSAGE, _COMMON_RULES, get_prompt_template
from rag.retriever import (
    EmptyQueryError,
    Retriever,
    RetrievedChunk,
    RetrieverNotReadyError,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

DEFAULT_TOP_K = 5
DEFAULT_QUIZ_QUESTION_COUNT = 5


# --------------------------------------------------------------------------- #
# مدل داده خروجی
# --------------------------------------------------------------------------- #
@dataclass
class PromptResult:
    """
    نتیجه فرآیند ساخت Prompt؛ خروجی نهایی این ماژول که به generation/llm_client.py
    (در مرحله بعد) یا مستقیماً به لایه API تحویل داده می‌شود.

    Attributes:
        is_answerable: آیا اطلاعات کافی برای پاسخ‌گویی در knowledge_base یافت شد یا خیر.
        request_type: نوع درخواست تشخیص داده‌شده (QA/SCENARIO/QUIZ).
        prompt: متن کامل Prompt آماده ارسال به LLM (فقط زمانی که is_answerable=True پر می‌شود).
        fallback_message: پیام ثابت «یافت نشد» (فقط زمانی که is_answerable=False پر می‌شود).
        retrieved_chunks: لیست Chunkهای بازیابی‌شده (برای Logging، Debug یا استناد در پاسخ نهایی).
    """

    is_answerable: bool
    request_type: RequestType
    prompt: Optional[str] = None
    fallback_message: Optional[str] = None
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# خطای اختصاصی
# --------------------------------------------------------------------------- #
class PromptBuildError(Exception):
    """خطای عمومی مربوط به فرآیند ساخت Prompt."""


# --------------------------------------------------------------------------- #
# PromptBuilder
# --------------------------------------------------------------------------- #
class PromptBuilder:
    """
    اتصال Retriever، IntentClassifier و Prompt Templates برای تولید Prompt نهایی.

    جریان اجرای build():
        1. تشخیص نوع درخواست از روی متن کاربر
        2. بازیابی Chunkهای مرتبط از FAISS از طریق Retriever
        3. اگر هیچ Chunk‌ای بازیابی نشد -> بازگرداندن فوری Fallback (بدون تماس با LLM)
        4. در غیر این صورت -> ساخت Context متنی از Chunkها، انتخاب قالب مناسب،
           پر کردن placeholderها و بازگرداندن Prompt نهایی
    """

    def __init__(
        self,
        retriever: Retriever,
        intent_classifier: Optional[IntentClassifier] = None,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """
        Args:
            retriever: نمونه آماده از rag.retriever.Retriever (ایندکس آن load شده باشد).
            intent_classifier: نمونه از IntentClassifier؛ در صورت عدم ارائه، یک نمونه
                پیش‌فرض ساخته می‌شود.
            default_top_k: تعداد پیش‌فرض Chunkهای بازیابی‌شده در هر درخواست.
        """
        self.retriever = retriever
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.default_top_k = default_top_k

    def build(
        self,
        user_input: str,
        top_k: Optional[int] = None,
        num_questions: int = DEFAULT_QUIZ_QUESTION_COUNT,
    ) -> PromptResult:
        """
        ساخت Prompt نهایی برای یک درخواست کاربر.

        Args:
            user_input: متن خام درخواست/پرسش کاربر.
            top_k: تعداد Chunkهای مورد نیاز برای بازیابی؛ در صورت None از default_top_k استفاده می‌شود.
            num_questions: تعداد سؤالات ارزیابی درخواستی (فقط در حالت QUIZ استفاده می‌شود).

        Returns:
            PromptResult حاوی Prompt نهایی (در صورت موفقیت) یا پیام Fallback (در صورت عدم کفایت اطلاعات).

        Raises:
            PromptBuildError: در صورت بروز خطای غیرمنتظره در فرآیند بازیابی یا ساخت Prompt.
        """
        request_type = self.intent_classifier.classify(user_input)
        effective_top_k = top_k if top_k is not None else self.default_top_k

        try:
            retrieved_chunks = self.retriever.retrieve(query=user_input, top_k=effective_top_k)
        except EmptyQueryError as exc:
            raise PromptBuildError(f"پرسش نامعتبر است: {exc}") from exc
        except RetrieverNotReadyError as exc:
            raise PromptBuildError(f"سامانه بازیابی آماده نیست: {exc}") from exc

        # --- قانون اصلی Guardrail: اگر هیچ Chunk مرتبطی یافت نشد، مستقیماً Fallback --- #
        if not retrieved_chunks:
            logger.warning(
                "هیچ Chunk مرتبطی برای درخواست یافت نشد؛ بازگرداندن پیام Fallback بدون تماس با LLM."
            )
            return PromptResult(
                is_answerable=False,
                request_type=request_type,
                fallback_message=NOT_FOUND_MESSAGE,
                retrieved_chunks=[],
            )

        context_text = self._format_context(retrieved_chunks)
        template = get_prompt_template(request_type)

        try:
            if request_type == RequestType.QUIZ:
                final_prompt = template.format(
                    common_rules=_COMMON_RULES,
                    context=context_text,
                    user_input=user_input,
                    num_questions=num_questions,
                )
            else:
                final_prompt = template.format(
                    common_rules=_COMMON_RULES,
                    context=context_text,
                    user_input=user_input,
                )
        except KeyError as exc:
            raise PromptBuildError(f"خطا در پرکردن قالب Prompt: placeholder گمشده {exc}") from exc

        logger.info(
            "Prompt نهایی برای نوع '%s' با %d Chunk بازیابی‌شده ساخته شد.",
            request_type.value,
            len(retrieved_chunks),
        )

        return PromptResult(
            is_answerable=True,
            request_type=request_type,
            prompt=final_prompt,
            retrieved_chunks=retrieved_chunks,
        )

    # ----------------------------------------------------------------- #
    # متدهای داخلی
    # ----------------------------------------------------------------- #
    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> str:
        """
        تبدیل لیست Chunkهای بازیابی‌شده به یک رشته متنی قابل تزریق در Prompt،
        به‌همراه برچسب منبع (نام فایل و شماره صفحه) برای هر بخش.
        """
        parts: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            header = f"[منبع {i}: {chunk.source_file} - صفحه {chunk.page_number}]"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------- #
# اجرای مستقل جهت تست سریع (python -m generation.prompt_builder)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    from rag.vectorstore import EmbeddingModel, FAISSVectorStore

    index_dir = sys.argv[1] if len(sys.argv) > 1 else "data/vector_store"
    test_query = sys.argv[2] if len(sys.argv) > 2 else "اقدامات اولیه در زلزله چیست؟"

    embedder = EmbeddingModel()
    store = FAISSVectorStore(embedding_model=embedder, index_dir=index_dir)
    store.load()

    retriever = Retriever(vector_store=store, default_top_k=5)
    builder = PromptBuilder(retriever=retriever)

    result = builder.build(test_query)

    if not result.is_answerable:
        print(result.fallback_message)
    else:
        print(f"نوع درخواست: {result.request_type.value}\n")
        print("--- Prompt نهایی ---")
        print(result.prompt)
