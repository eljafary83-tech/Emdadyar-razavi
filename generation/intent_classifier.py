"""
generation/intent_classifier.py

ماژول تشخیص نوع درخواست (Intent Classification) برای پروژه «امدادیار رضوی».

این ماژول مسئول تعیین این موضوع است که پرسش/درخواست کاربر به کدام‌یک از
سه حالت عملکردی سیستم تعلق دارد:

    QA        -> پاسخ‌گویی مستقیم به یک سؤال
    SCENARIO  -> تولید سناریوی آموزشی
    QUIZ      -> تولید سؤالات ارزیابی

تشخیص به‌صورت Rule-Based و مبتنی بر کلیدواژه‌های فارسی انجام می‌شود تا:
    - سریع، قطعی (Deterministic) و بدون هزینه فراخوانی LLM باشد
    - در Production قابل اعتماد و قابل تست واحد باشد

خروجی این ماژول (RequestType) توسط services/assistant.py برای انتخاب
Prompt فایل‌محور و اعتبارسنجی خروجی استفاده می‌شود.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------- #
# نوع درخواست
# --------------------------------------------------------------------------- #
class RequestType(str, Enum):
    """سه نوع مجاز درخواست در سیستم امدادیار رضوی."""

    QA = "QA"
    SCENARIO = "SCENARIO"
    QUIZ = "QUIZ"


# --------------------------------------------------------------------------- #
# IntentClassifier
# --------------------------------------------------------------------------- #
class IntentClassifier:
    """
    تشخیص نوع درخواست کاربر بر اساس کلیدواژه‌های فارسی موجود در متن.

    منطق اولویت‌بندی:
        1. ابتدا بررسی کلیدواژه‌های QUIZ (چون معمولاً دقیق‌تر و اختصاصی‌تر هستند)
        2. سپس بررسی کلیدواژه‌های SCENARIO
        3. در صورت عدم تطبیق با هیچ‌کدام، نوع پیش‌فرض QA در نظر گرفته می‌شود

    این کلاس Stateless است و می‌توان یک نمونه واحد از آن را در کل عمر
    برنامه (Singleton) استفاده کرد.
    """

    # کلیدواژه‌های مرتبط با تولید سؤالات ارزیابی
    QUIZ_KEYWORDS: List[str] = [
        "سؤال ارزیابی",
        "سوال ارزیابی",
        "سؤالات ارزیابی",
        "سوالات ارزیابی",
        "کوییز",
        "کویز",
        "quiz",
        "آزمون",
        "تست بگیر",
        "تست بساز",
        "چهارگزینه‌ای",
        "چهار گزینه‌ای",
        "چند گزینه‌ای",
        "چندگزینه‌ای",
        "طراحی سؤال",
        "طراحی سوال",
        "بسازید سؤال",
        "بساز سؤال",
        "سؤال بساز",
        "سوال بساز",
        "سنجش دانش",
        "امتحان بگیر",
    ]

    # کلیدواژه‌های مرتبط با تولید سناریوی آموزشی
    SCENARIO_KEYWORDS: List[str] = [
        "سناریو",
        "سناریوی آموزشی",
        "شبیه‌سازی",
        "شبیه سازی",
        "مانور",
        "وضعیت فرضی",
        "فرض کنید",
        "فرض کن",
        "تمرین عملیاتی",
        "موقعیت فرضی",
        "داستان آموزشی",
        "بازی نقش",
        "ایفای نقش",
        "روایت آموزشی",
    ]

    def __init__(
        self,
        extra_quiz_keywords: Optional[List[str]] = None,
        extra_scenario_keywords: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            extra_quiz_keywords: کلیدواژه‌های اضافی برای تشخیص QUIZ (قابل تنظیم از config).
            extra_scenario_keywords: کلیدواژه‌های اضافی برای تشخیص SCENARIO.
        """
        self.quiz_keywords = list(self.QUIZ_KEYWORDS) + (extra_quiz_keywords or [])
        self.scenario_keywords = list(self.SCENARIO_KEYWORDS) + (extra_scenario_keywords or [])

        self._quiz_pattern = self._build_pattern(self.quiz_keywords)
        self._scenario_pattern = self._build_pattern(self.scenario_keywords)

    # ----------------------------------------------------------------- #
    # API عمومی
    # ----------------------------------------------------------------- #
    def classify(self, user_input: str) -> RequestType:
        """
        تشخیص نوع درخواست بر اساس متن ورودی کاربر.

        Args:
            user_input: متن خام درخواست/پرسش کاربر.

        Returns:
            یکی از مقادیر RequestType (QA به‌عنوان پیش‌فرض در صورت عدم تطبیق).
        """
        normalized_text = self._normalize(user_input)

        if self._quiz_pattern.search(normalized_text):
            logger.info("نوع درخواست تشخیص داده شد: QUIZ")
            return RequestType.QUIZ

        if self._scenario_pattern.search(normalized_text):
            logger.info("نوع درخواست تشخیص داده شد: SCENARIO")
            return RequestType.SCENARIO

        logger.info("نوع درخواست تشخیص داده شد: QA (پیش‌فرض)")
        return RequestType.QA

    # ----------------------------------------------------------------- #
    # متدهای داخلی
    # ----------------------------------------------------------------- #
    @staticmethod
    def _normalize(text: str) -> str:
        """
        نرمال‌سازی سبک متن فارسی جهت افزایش دقت تطبیق کلیدواژه:
            - یکسان‌سازی نویسه «ی» و «ک» عربی/فارسی
            - حذف فاصله‌های اضافه
        """
        if not text:
            return ""
        text = text.replace("ي", "ی").replace("ك", "ک")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _build_pattern(keywords: List[str]) -> re.Pattern:
        """ساخت یک الگوی Regex واحد از لیست کلیدواژه‌ها برای جستجوی سریع."""
        escaped_keywords = [re.escape(keyword) for keyword in keywords]
        pattern = "|".join(escaped_keywords)
        return re.compile(pattern, flags=re.IGNORECASE)


# --------------------------------------------------------------------------- #
# اجرای مستقل جهت تست سریع (python -m generation.intent_classifier)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    classifier = IntentClassifier()

    test_inputs = [
        "اقدامات اولیه در زلزله چیست؟",
        "یک سناریوی آموزشی برای تیم امداد در سیل طراحی کن",
        "برای این مبحث ۵ سؤال ارزیابی چهارگزینه‌ای بساز",
        "فرض کنید یک زلزله ۶ ریشتری رخ داده، وضعیت را شبیه‌سازی کن",
    ]

    for text in test_inputs:
        result = classifier.classify(text)
        print(f"ورودی: {text}\n=> نوع تشخیص داده‌شده: {result.value}\n")
