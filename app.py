"""
app.py

فایل اصلی اجرای پروژه «امدادیار رضوی» (هلال احمر خراسان رضوی).

این فایل نقطه ورود واحد (Single Entry Point) پروژه است و مسئولیت‌های زیر
را بر عهده دارد:

    1. اجرای رابط کاربری Streamlit (با بازاستفاده از اجزای بصری آماده‌شده
       در ui/streamlit_app.py: هدر، CSS سازمانی، نقشه ساده استان)
    2. اتصال کامل Backend: بارگذاری/ساخت ایندکس RAG (rag/loader.py با
       PyMuPDF + LangChain FAISS در utils/generator.py)
    3. اجرای Pipeline کامل RAG برای هر درخواست کاربر
    4. فراخوانی ماژول Classifier (utils/classifier.py) برای تشخیص نوع
       درخواست (QA / SCENARIO / QUIZ) — این فراخوانی به‌صورت داخلی توسط
       utils.generator.RAGGenerator انجام می‌شود
    5. انتخاب و پرکردن Prompt Template مناسب (generation/prompt_templates.py)
       — نیز به‌صورت داخلی توسط RAGGenerator انجام می‌شود
    6. نمایش پاسخ نهایی تولیدشده در رابط کاربری
    7. اجرای دقیق قانون Fallback: اگر اطلاعات کافی در Context بازیابی‌شده
       از RAG یافت نشود، فقط و فقط پیام زیر نمایش داده می‌شود:
       "پاسخی برای این سوال در منابع موجود یافت نشد."

اجرای برنامه:
    streamlit run app.py

---------------------------------------------------------------------------
یادداشت اصلاحات این نسخه:
---------------------------------------------------------------------------
- تمام «کارت‌های سفید» صفحه (نوع درخواست، نقشه، فصل/نوع حادثه، ورودی متن،
  نتیجه) اکنون با emdad_card(...) (تعریف‌شده همین‌جا در app.py، تا وابسته
  به نسخهٔ ui/streamlit_app.py نباشد) ساخته می‌شوند. این تابع به‌جای
  باز/بسته‌کردن دستی <div> در دو فراخوانی جدای
  st.markdown (که باعث می‌شد محتوای واقعی، از جمله تیتر «متن درخواست خود
  را وارد کنید» و «موضوع سؤالات ارزیابی» و نیز نقشه استان و select‌باکس‌های
  فصل/نوع حادثه، بیرون از کادر سفید رندر شود)، از st.container(key=...)
  استفاده می‌کند که یک عنصر واقعی و پایدار در DOM می‌سازد؛ در نتیجه هر
  چیزی که داخل بلوک with نوشته شود، واقعاً داخل همان کادر سفید قرار
  می‌گیرد.
- هدر (render_header) اکنون راست‌چین است، هلال ماه با رنگ قرمز برند و
  لوگوی سازمانی از assets/logo.png (به‌صورت Base64) نمایش داده می‌شود؛
  این تغییرات در ui/streamlit_app.py اعمال شده‌اند.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import streamlit as st

from generation.intent_classifier import RequestType
from generation.prompt_templates import NOT_FOUND_MESSAGE
from ui.streamlit_app import (
    COUNTY_POSITIONS,
    INCIDENT_TYPES,
    REQUEST_TYPE_LABELS,
    SEASONS,
    build_province_map_svg,
    inject_custom_css,
    render_header,
)
from utils.generator import EmptyQueryError, GenerationError, IndexNotReadyError, RAGGenerator

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------- #
# کارت سازمانی (کمکی صرفاً چیدمانی — بدون هیچ منطق/داده‌ای)
# --------------------------------------------------------------------------- #
# این تابع این‌جا (در app.py) تعریف شده تا خطای:
#   ImportError: cannot import name 'emdad_card' from 'ui.streamlit_app'
# رفع شود، بدون این‌که نیازی به تغییر ui/streamlit_app.py باشد. کارکرد آن
# دقیقاً همان چیدمان قبلی است: یک کادر سفید واقعی در DOM (با
# st.container) می‌سازد تا تیتر و ویجت‌های داخل بلوک with، واقعاً درون
# همان کادر سفید نمایش داده شوند.
@contextmanager
def emdad_card(key: str):
    with st.container(key=key):
        yield


# --------------------------------------------------------------------------- #
# رفع مشکل نمایش نقشه به‌صورت کد خام (بدون تغییر ui/streamlit_app.py)
# --------------------------------------------------------------------------- #
# build_province_map_svg (تعریف‌شده در ui/streamlit_app.py) رشته HTML/SVG را
# به‌صورت تو‌رفته (Indented؛ چون داخل یک تابع نوشته شده) برمی‌گرداند. اگر یک
# خط از ورودی st.markdown با ۴ فاصله یا بیشتر شروع شود، موتور Markdown آن را
# یک بلوک کد (Code Block) در نظر می‌گیرد و به‌جای رندر HTML/SVG، متن خام آن
# را عیناً روی صفحه نمایش می‌دهد؛ همین موضوع باعث می‌شد نقشه به‌صورت کد خام
# دیده شود. این تابع فاصله‌های ابتدای هر خط را حذف می‌کند تا Markdown آن را
# به‌عنوان HTML/SVG رندر کند، نه یک بلوک کد.
def _render_html_block(html: str) -> None:
    dedented_html = "\n".join(line.lstrip() for line in html.strip().splitlines())
    st.markdown(dedented_html, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# پیکربندی صفحه
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="امدادیار رضوی | هلال احمر خراسان رضوی",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
# اتصال Backend: بارگذاری یک‌باره RAGGenerator (RAG + Classifier + Prompt + LLM)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="در حال آماده‌سازی پایگاه دانش امدادیار رضوی ...")
def load_generator() -> RAGGenerator:
    """
    ساخت و آماده‌سازی یک‌باره (Cached) RAGGenerator برای کل عمر Session.

    RAGGenerator خودش مسئول اجرای زنجیره کامل RAG است:
        rag/loader.py (PyMuPDF) -> LangChain FAISS -> Retrieval
        -> utils/classifier.py (تشخیص نوع درخواست)
        -> generation/prompt_templates.py (انتخاب و پرکردن قالب)
        -> LLM (ChatAnthropic) -> پاسخ نهایی

    این تابع فقط یک‌بار اجرا می‌شود تا از بارگذاری مجدد مدل Embedding و
    ساخت مکرر ایندکس در هر تعامل کاربر جلوگیری شود.
    """
    generator = RAGGenerator()
    generator.ensure_ready()
    return generator


# --------------------------------------------------------------------------- #
# نمایش پاسخ نهایی
# --------------------------------------------------------------------------- #
def render_result(result: dict) -> None:
    """
    نمایش خروجی RAGGenerator.generate_with_sources() در رابط کاربری.

    اگر is_answerable برابر False باشد (یعنی هیچ Context مرتبطی از RAG
    بازیابی نشده)، طبق الزام سیستم، فقط پیام ثابت زیر نمایش داده می‌شود:
        "پاسخی برای این سوال در منابع موجود یافت نشد."
    و هیچ منبع یا متن اضافه‌ای نمایش داده نمی‌شود.
    """
    with emdad_card("emdad_card_result"):
        st.markdown('<p class="emdad-card-title">📋 نتیجه</p>', unsafe_allow_html=True)

        if not result["is_answerable"]:
            # نمایش دقیق و بدون هیچ متن اضافه‌ی پیام ثابت Fallback
            st.markdown(
                f'<div class="emdad-fallback-box" style="text-align: right">{NOT_FOUND_MESSAGE}</div>',
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            f'<div class="emdad-answer-box" style="text-align: right">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )

        if result["sources"]:
            st.markdown("<div style='margin-top:14px; text-align: right;'>منابع مورد استفاده:</div>", unsafe_allow_html=True)
            tags_html = "".join(
                f'<span class="emdad-source-tag" style="text-align: right">{src["source_file"]} - ص {src["page_number"]}</span>'
                for src in result["sources"]
            )
            st.markdown(tags_html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# برنامه اصلی
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_custom_css()
    render_header()

    if "selected_county" not in st.session_state:
        st.session_state.selected_county = None

    # ---------------- ۱. انتخاب نوع درخواست ---------------- #
    with emdad_card("emdad_card_request_type"):
        st.markdown(
            '<p class="emdad-card-title">۱. نوع درخواست را انتخاب کنید</p>',
            unsafe_allow_html=True,
        )
        request_type_label = st.radio(
            label="نوع درخواست",
            options=list(REQUEST_TYPE_LABELS.keys()),
            horizontal=False,
            label_visibility="collapsed",
        )
    request_type = REQUEST_TYPE_LABELS[request_type_label]

    # ---------------- ۲. تنظیمات اختصاصی حالت سناریو ---------------- #
    county = None
    season = None
    incident_type = None

    if request_type == RequestType.SCENARIO:
        map_col, form_col = st.columns([1.1, 1], gap="large")

        with map_col:
            with emdad_card("emdad_card_map"):
                st.markdown(
                    '<p class="emdad-card-title">شهرستان مورد نظر خود را از نقشه انتخاب کنید</p>',
                    unsafe_allow_html=True,
                )
                _render_html_block(build_province_map_svg(st.session_state.selected_county))
                county = st.selectbox(
                    "انتخاب شهرستان",
                    options=list(COUNTY_POSITIONS.keys()),
                    index=None,
                    placeholder="یک شهرستان را انتخاب کنید ...",
                )
                st.session_state.selected_county = county

        with form_col:
            with emdad_card("emdad_card_scenario_details"):
                st.markdown(
                    '<p class="emdad-card-title">فصل و نوع حادثه رو انتخاب کنید</p>',
                    unsafe_allow_html=True,
                )
                season = st.selectbox(
                    "فصل وقوع حادثه", options=SEASONS, index=None, placeholder="انتخاب فصل ..."
                )
                incident_type = st.selectbox(
                    "نوع حادثه",
                    options=INCIDENT_TYPES,
                    index=None,
                    placeholder="انتخاب نوع حادثه ...",
                )

    # ---------------- ۴. ورودی متن ---------------- #
    num_questions = 5
    with emdad_card("emdad_card_text_input"):
        if request_type == RequestType.SCENARIO:
            st.markdown('<p class="emdad-card-title">۴. توضیحات تکمیلی (اختیاری)</p>', unsafe_allow_html=True)
            placeholder = "در صورت نیاز، جزئیات یا محدودیت‌های خاص سناریو را اینجا بنویسید ..."
        elif request_type == RequestType.QUIZ:
            st.markdown('<p class="emdad-card-title">۲. موضوع سؤالات ارزیابی</p>', unsafe_allow_html=True)
            placeholder = "مثال: برای مبحث کمک‌های اولیه، سؤال ارزیابی طراحی کن."
            num_questions = st.slider("تعداد سؤالات", min_value=3, max_value=10, value=5)
        else:
            st.markdown('<p class="emdad-card-title">۲. متن درخواست خود را وارد کنید</p>', unsafe_allow_html=True)
            placeholder = "مثال: اقدامات اولیه در مواجهه با زلزله چیست؟"

        user_text = st.text_area(
            "متن درخواست",
            placeholder=placeholder,
            height=110,
            label_visibility="collapsed",
        )
        submit_clicked = st.button("🔍 ارسال درخواست", use_container_width=False)

    # ---------------- منطق ارسال به Backend (اجرای کامل RAG) ---------------- #
    if submit_clicked:
        if request_type == RequestType.SCENARIO and not (county and season and incident_type):
            st.warning("لطفاً پیش از ارسال، شهرستان، فصل و نوع حادثه را انتخاب کنید.")
            return

        if request_type == RequestType.SCENARIO:
            final_query = (
                f"یک سناریوی آموزشی واقع‌گرایانه برای حادثه «{incident_type}» "
                f"در شهرستان {county} از استان خراسان رضوی، در فصل {season}، طراحی کن. "
                f"سناریو باید متناسب با شرایط جغرافیایی و اقلیمی واقعی همین شهرستان باشد."
            )
            if user_text.strip():
                final_query += f" جزئیات تکمیلی درخواست‌شده توسط کاربر: {user_text.strip()}"
        else:
            if not user_text.strip():
                st.warning("لطفاً متن درخواست خود را وارد کنید.")
                return
            final_query = user_text.strip()

        try:
            generator = load_generator()
            with st.spinner("در حال بازیابی از منابع و تولید پاسخ ..."):
                result = generator.generate_with_sources(
                    user_input=final_query,
                    top_k=5,
                    num_questions=num_questions,
                )
            render_result(result)
        except EmptyQueryError:
            st.warning("لطفاً متن درخواست خود را وارد کنید.")
        except IndexNotReadyError as exc:
            st.error(
                "پایگاه دانش هنوز آماده نیست. لطفاً از وجود فایل‌های PDF در پوشه "
                f"knowledge_base اطمینان حاصل کنید.\n\nجزئیات فنی: {exc}"
            )
        except GenerationError as exc:
            st.error(f"خطایی در فرآیند تولید پاسخ رخ داد: {exc}")

    # ---------------- فوتر ---------------- #
    st.markdown(
        '<div class="emdad-footer">امدادیار رضوی — جمعیت هلال احمر خراسان رضوی | '
        "تمام پاسخ‌ها صرفاً بر اساس منابع آموزشی رسمی سازمان تولید می‌شوند.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
