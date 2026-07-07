"""
utils/classifier.py

لایه سازگاری (Compatibility Layer) برای تشخیص نوع درخواست، در بخش Backend.

منطق واقعی تشخیص نوع درخواست (QA / SCENARIO / QUIZ) قبلاً در
generation/intent_classifier.py پیاده‌سازی شده است. این فایل صرفاً یک
Wrapper سبک روی همان ماژول است تا:

    - لایه Backend (utils/generator.py) و لایه API آینده بتوانند از یک
      مسیر واردسازی ساده و پایدار (utils.classifier) استفاده کنند.
    - منطق تشخیص در یک‌جا (generation/intent_classifier.py) نگهداری شود
      و از تکرار کد (Code Duplication) جلوگیری شود.
    - یک نمونه پیش‌فرض (Singleton سبک) از Classifier در اختیار کل Backend
      قرار گیرد تا هزینه ساخت مجدد الگوهای Regex در هر فراخوانی حذف شود.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from generation.intent_classifier import IntentClassifier, RequestType

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------- #
# RequestClassifier
# --------------------------------------------------------------------------- #
class RequestClassifier:
    """
    Wrapper سطح Backend روی generation.intent_classifier.IntentClassifier.

    این کلاس هیچ منطق تشخیصی جدیدی اضافه نمی‌کند؛ صرفاً یک واسط پایدار
    برای استفاده در utils/generator.py و لایه‌های بالادست فراهم می‌کند.
    """

    def __init__(
        self,
        extra_quiz_keywords: Optional[List[str]] = None,
        extra_scenario_keywords: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            extra_quiz_keywords: کلیدواژه‌های اضافی برای تشخیص نوع QUIZ.
            extra_scenario_keywords: کلیدواژه‌های اضافی برای تشخیص نوع SCENARIO.
        """
        self._impl = IntentClassifier(
            extra_quiz_keywords=extra_quiz_keywords,
            extra_scenario_keywords=extra_scenario_keywords,
        )

    def classify(self, user_input: str) -> RequestType:
        """
        تشخیص نوع درخواست و بازگرداندن مقدار Enum (RequestType).

        Args:
            user_input: متن خام درخواست کاربر.

        Returns:
            یکی از مقادیر RequestType.QA / RequestType.SCENARIO / RequestType.QUIZ.
        """
        return self._impl.classify(user_input)

    def classify_as_string(self, user_input: str) -> str:
        """
        تشخیص نوع درخواست و بازگرداندن آن به شکل رشته ساده (مثلاً برای Log یا API Response).

        Args:
            user_input: متن خام درخواست کاربر.

        Returns:
            یکی از رشته‌های "QA"، "SCENARIO" یا "QUIZ".
        """
        return self._impl.classify(user_input).value


# --------------------------------------------------------------------------- #
# نمونه پیش‌فرض سطح ماژول (Singleton سبک)
# --------------------------------------------------------------------------- #
_default_classifier_instance: Optional[RequestClassifier] = None


def get_default_classifier() -> RequestClassifier:
    """
    بازگرداندن یک نمونه مشترک و از‌پیش‌ساخته‌شده از RequestClassifier.

    استفاده از این تابع در کل Backend توصیه می‌شود تا الگوهای Regex داخلی
    Classifier فقط یک‌بار در طول عمر برنامه ساخته شوند.
    """
    global _default_classifier_instance
    if _default_classifier_instance is None:
        logger.info("در حال ساخت نمونه پیش‌فرض RequestClassifier ...")
        _default_classifier_instance = RequestClassifier()
    return _default_classifier_instance


def classify_request(user_input: str) -> str:
    """
    تابع کمکی سطح‌بالا برای تشخیص سریع نوع درخواست بدون نیاز به ساخت دستی نمونه.

    Args:
        user_input: متن خام درخواست کاربر.

    Returns:
        یکی از رشته‌های "QA"، "SCENARIO" یا "QUIZ".
    """
    return get_default_classifier().classify_as_string(user_input)


# --------------------------------------------------------------------------- #
# اجرای مستقل جهت تست سریع (python -m utils.classifier)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_inputs = [
        "اقدامات اولیه در زلزله چیست؟",
        "یک سناریوی آموزشی برای امدادرسانی به زائران در ایام شلوغی مشهد طراحی کن",
        "برای مبحث کمک‌های اولیه ۵ سؤال ارزیابی چهارگزینه‌ای بساز",
    ]

    for text in test_inputs:
        print(f"ورودی: {text}\n=> نوع تشخیص داده‌شده: {classify_request(text)}\n")
