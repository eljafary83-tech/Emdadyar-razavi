"""
ui/streamlit_app.py

رابط کاربری Streamlit پروژه «امدادیار رضوی» (هلال احمر خراسان رضوی).

این فایل تنها لایه نمایش (Presentation Layer) است و مستقیماً به ماژول‌های
Backend ساخته‌شده در مراحل قبل متصل می‌شود:

    rag.vectorstore.EmbeddingModel / FAISSVectorStore
    rag.retriever.Retriever
    generation.intent_classifier.RequestType
    generation.prompt_builder.PromptBuilder

اجرای برنامه:
    streamlit run ui/streamlit_app.py

---------------------------------------------------------------------------
یادداشت اصلاحات این نسخه:
---------------------------------------------------------------------------
۱) هدر: عبارت «امدادیار رضوی» راست‌چین (RTL) شد و هلال ماه کنار آن با یک
   آیکون SVG واقعی (به‌جای اموجی) به رنگ قرمز برند نمایش داده می‌شود
   (رنگ اموجی توسط CSS قابل تغییر نیست، به همین دلیل از SVG استفاده شد).
   آیکون سمت چپ هدر نیز از فایل assets/logo.png خوانده و به‌صورت Base64
   درون تگ <img> جای‌گذاری می‌شود (خواندن مستقیم مسیر نسبی توسط مرورگر
   در Streamlit کار نمی‌کند؛ به همین علت فایل باید Base64 شود). اگر فایل
   لوگو یافت نشود، آیکون پیش‌فرض (هلال ماه) به‌عنوان جایگزین نمایش داده
   می‌شود تا برنامه خطا ندهد.

۲) باگ اصلی «خارج افتادن محتوا از کادر سفید»: در نسخه قبلی، برای رسم هر
   «کارت»، یک تگ <div> باز در یک فراخوانی st.markdown و تگ بستن آن در
   فراخوانی دیگری از st.markdown نوشته شده بود. از آن‌جا که هر فراخوانی
   st.markdown یک عنصر مستقل و جدا در DOM می‌سازد، مرورگر بلافاصله همان
   <div> باز-نشده را در همان فراخوانی می‌بندد؛ در نتیجه تیتر/ویجت‌هایی که
   در فراخوانی‌های بعدی می‌آمدند (مثلاً «متن درخواست خود را وارد کنید» یا
   «موضوع سؤالات ارزیابی» و نیز نقشه استان و select‌باکس‌های فصل/نوع
   حادثه) واقعاً داخل کادر سفید قرار نمی‌گرفتند.
   راه‌حل: تابع emdad_card() که از st.container(key=...) استفاده می‌کند؛
   این یک عنصر واقعی در DOM می‌سازد که تیتر و ویجت‌های بومی Streamlit به
   شکل واقعی و تودرتو (nested) داخل همان عنصر قرار می‌گیرند.

۳) راست‌چین‌شدن select باکس‌ها (فصل/نوع حادثه) و رادیو با افزودن CSS
   عمومی روی عناصر BaseWeb Streamlit.
"""

from __future__ import annotations

import base64
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import streamlit as st

from generation.intent_classifier import RequestType
from generation.prompt_builder import PromptBuildError, PromptBuilder, PromptResult
from rag.retriever import Retriever, RetrieverNotReadyError
from rag.vectorstore import EmbeddingModel, FAISSVectorStore, VectorStoreError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# تنظیمات ثابت
# --------------------------------------------------------------------------- #
VECTOR_STORE_DIR = "data/vector_store"

BRAND = {
    "red": "#C8102E",
    "red_dark": "#8C0F22",
    "gold": "#C9A24B",
    "gold_light": "#E7D4A1",
    "cream": "#FBF7EF",
    "text_dark": "#2B2B2B",
    "border": "#E7DFCF",
}

SEASONS = ["بهار", "تابستان", "پاییز", "زمستان"]

INCIDENT_TYPES = [
    "امدادرسانی به زائران در ایام شلوغی",
    "تصادف جاده‌ای",
    "کولاک و بهمن در محورهای کوهستانی",
    "سیلاب",
    "گم‌شدن افراد در مناطق طبیعی",
    "آتش‌سوزی",
    "زلزله",
    "سایر حوادث",
]

REQUEST_TYPE_LABELS = {
    "پاسخ‌گویی به سؤال": RequestType.QA,
    "طراحی سناریوی آموزشی": RequestType.SCENARIO,
    "تولید سؤالات ارزیابی": RequestType.QUIZ,
}

# مختصات نمایشی (غیر جغرافیایی و ساده‌شده) شهرستان‌های خراسان رضوی
# روی یک بوم SVG با ابعاد ۳۶۰x۴۴۰ برای ترسیم نقشه ساده استان
COUNTY_POSITIONS = {
    "مشهد": (215, 210),
    "چناران": (170, 165),
    "قوچان": (185, 105),
    "درگز": (230, 55),
    "سرخس": (300, 150),
    "فریمان": (150, 235),
    "تربت جام": (245, 300),
    "تایباد": (200, 335),
    "خواف": (170, 360),
    "تربت حیدریه": (120, 255),
    "کاشمر": (75, 220),
    "بردسکن": (95, 175),
    "نیشابور": (110, 130),
    "سبزوار": (55, 110),
    "گناباد": (85, 300),
    "فیض‌آباد": (115, 305),
}

# مسیرهای احتمالی فایل لوگو (برای اجرای مستقل یا از طریق app.py در ریشه پروژه)
LOGO_CANDIDATE_PATHS = (
    Path("assets/logo.png"),
    Path(__file__).resolve().parent.parent / "assets" / "logo.png",
    Path(__file__).resolve().parent / "assets" / "logo.png",
)

# --------------------------------------------------------------------------- #
# پیکربندی صفحه
# --------------------------------------------------------------------------- #
try:
    st.set_page_config(
        page_title="امدادیار رضوی | هلال احمر خراسان رضوی",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
except Exception:
    # این فایل هم به‌صورت مستقل (streamlit run ui/streamlit_app.py) و هم
    # به‌صورت Import‌شده در app.py (نقطه ورود اصلی پروژه) قابل استفاده است.
    # اگر app.py قبلاً st.set_page_config را فراخوانی کرده باشد، Streamlit
    # در فراخوانی دوم خطا می‌دهد؛ این try/except از آن جلوگیری می‌کند.
    pass


# --------------------------------------------------------------------------- #
# خواندن لوگوی سازمانی (assets/logo.png) به‌صورت Base64
# --------------------------------------------------------------------------- #
def _load_logo_base64() -> str:
    """
    خواندن فایل assets/logo.png و بازگرداندن محتوای آن به‌صورت Base64.

    خواندن مستقیم یک مسیر نسبی (مثلاً src="assets/logo.png") در HTML
    تزریق‌شده با st.markdown، در بسیاری از حالت‌های اجرای Streamlit کار
    نمی‌کند (چون مرورگر آن را نسبت به URL صفحه جست‌وجو می‌کند، نه نسبت به
    پوشه پروژه روی دیسک). راه مطمئن، خواندن بایت‌های فایل در سمت سرور و
    جای‌گذاری آن به‌صورت Data URI (Base64) در تگ <img> است.

    اگر فایل در هیچ‌کدام از مسیرهای احتمالی یافت نشود، رشته خالی
    بازگردانده می‌شود تا رابط کاربری بدون خطا و با آیکون جایگزین
    (هلال ماه) نمایش داده شود.
    """
    for candidate in LOGO_CANDIDATE_PATHS:
        try:
            if candidate.is_file():
                return base64.b64encode(candidate.read_bytes()).decode("utf-8")
        except OSError:
            continue
    logger.warning(
        "فایل لوگو در مسیرهای مورد انتظار (assets/logo.png) یافت نشد؛ "
        "آیکون پیش‌فرض هدر نمایش داده می‌شود."
    )
    return ""


# --------------------------------------------------------------------------- #
# CSS سازمانی (رنگ‌بندی، تایپوگرافی، راست‌به‌چپ)
# --------------------------------------------------------------------------- #
def inject_custom_css() -> None:
    """تزریق استایل سازمانی هلال احمر خراسان رضوی (رنگ، فونت، جهت راست‌به‌چپ)."""
    st.markdown(
        f"""
        <style>
        @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

        html, body, [class*="css"] {{
            font-family: 'Vazirmatn', Tahoma, sans-serif !important;
            direction: rtl;
        }}

        .stApp {{
            background-color: {BRAND['cream']};
            color: {BRAND['text_dark']};
        }}

        /* ---------- هدر مینیمال ---------- */
        .emdad-header {{
            background: linear-gradient(90deg, {BRAND['red_dark']} 0%, {BRAND['red']} 55%, {BRAND['gold']} 100%);
            padding: 22px 32px;
            border-radius: 10px;
            margin-bottom: 28px;
            display: flex;
            flex-direction: row-reverse;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        }}
        .emdad-header-text {{
            direction: rtl;
            text-align: right;
            width: 100%;
        }}
        .emdad-header-title {{
            color: #FFFFFF;
            font-size: 26px;
            font-weight: 800;
            margin: 0;
            display: flex;
            flex-direction: row-reverse;
            align-items: center;
            justify-content: flex-end;
            gap: 8px;
            direction: rtl;
            text-align: right;
            width: 100%;
        }}
        .emdad-header-subtitle {{
            color: {BRAND['gold_light']};
            font-size: 14px;
            margin-top: 4px;
            text-align: right;
            direction: rtl;
        }}
        .emdad-header-crescent {{
            display: inline-flex;
            align-items: center;
            line-height: 0;
        }}
        .emdad-header-crescent svg {{
            display: block;
        }}
        .emdad-header-icon {{
            font-size: 34px;
            color: {BRAND['red']};
        }}
        .emdad-header-icon-wrap {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .emdad-header-logo {{
            height: 48px;
            border-radius: 8px;
            background-color: #FFFFFF;
            padding: 3px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.25);
        }}

        /* ---------- کارت‌ها (نسخه قدیمی؛ برای سازگاری با HTML خام نگه داشته شده) ---------- */
        .emdad-card {{
            background-color: #FFFFFF;
            border: 1px solid {BRAND['border']};
            border-radius: 12px;
            padding: 22px 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
            direction: rtl;
            text-align: right;
        }}

        /* ---------- کارت‌های واقعی ساخته‌شده با st.container(key=...) ----------
           این انتخاب‌گر روی هر container ای که کلید آن با «emdad_card»
           شروع شود اعمال می‌شود (Streamlit به‌صورت خودکار کلاس
           st-key-<key> را به آن container اضافه می‌کند). با این روش،
           تیتر و ویجت‌های بومی که داخل همان container قرار می‌گیرند،
           واقعاً داخل کادر سفید رندر می‌شوند. */
        div[class*="st-key-emdad_card"] {{
            background-color: #FFFFFF;
            border: 1px solid {BRAND['border']};
            border-radius: 12px;
            padding: 22px 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }}
        div[class*="st-key-emdad_card"],
        div[class*="st-key-emdad_card"] * {{
            direction: rtl;
        }}
        .emdad-card-title {{
            color: {BRAND['red_dark']};
            font-weight: 800;
            font-size: 17px;
            margin-bottom: 12px;
            border-right: 4px solid {BRAND['gold']};
            padding-right: 10px;
            direction: rtl;
            text-align: right;
            width: 100%;
            display: block;
        }}

        /* ---------- راست‌چین کردن ویجت‌های بومی Streamlit ---------- */
        div[data-baseweb="select"] {{
            direction: rtl;
        }}
        div[data-baseweb="select"] > div {{
            text-align: right;
            direction: rtl;
        }}
        div[data-baseweb="popover"] li,
        ul[role="listbox"] li,
        div[data-baseweb="menu"] li {{
            direction: rtl !important;
            text-align: right !important;
        }}
        div[role="radiogroup"] {{
            direction: rtl;
            align-items: flex-end;
        }}
        div[role="radiogroup"] label {{
            direction: rtl;
        }}
        .stSelectbox label,
        .stRadio label,
        .stTextArea label,
        .stSlider label {{
            direction: rtl;
            text-align: right;
            width: 100%;
        }}
        .stTextArea textarea {{
            direction: rtl;
            text-align: right;
        }}

        /* ---------- پاسخ ---------- */
        .emdad-answer-box {{
            background-color: #FFFDF8;
            border: 1px solid {BRAND['gold_light']};
            border-right: 5px solid {BRAND['gold']};
            border-radius: 10px;
            padding: 20px 22px;
            line-height: 2.1;
            font-size: 15.5px;
            white-space: pre-wrap;
            direction: rtl;
            text-align: right;
        }}
        .emdad-fallback-box {{
            background-color: #FDECEC;
            border: 1px solid #F3B9B9;
            border-right: 5px solid {BRAND['red']};
            border-radius: 10px;
            padding: 18px 20px;
            color: {BRAND['red_dark']};
            font-weight: 700;
            direction: rtl;
            text-align: right;
        }}
        .emdad-source-tag {{
            display: inline-block;
            background-color: {BRAND['gold_light']};
            color: {BRAND['red_dark']};
            border-radius: 6px;
            padding: 3px 10px;
            font-size: 12.5px;
            margin: 3px 4px 0 0;
            direction: rtl;
            text-align: right;
        }}

        /* ---------- دکمه‌ها ---------- */
        div.stButton > button {{
            background-color: {BRAND['red']};
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 10px 26px;
            font-weight: 700;
            font-size: 15px;
            transition: background-color 0.2s ease-in-out;
        }}
        div.stButton > button:hover {{
            background-color: {BRAND['red_dark']};
            color: #FFFFFF;
        }}

        /* ---------- فوتر ---------- */
        .emdad-footer {{
            text-align: center;
            color: #8A8A8A;
            font-size: 12.5px;
            margin-top: 30px;
            padding-top: 14px;
            border-top: 1px solid {BRAND['border']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# کارت سازمانی واقعی (Container واقعی در DOM)
# --------------------------------------------------------------------------- #
@contextmanager
def emdad_card(key: str) -> Iterator[st.delta_generator.DeltaGenerator]:
    """
    Context Manager برای رسم یک «کارت سازمانی» واقعی و صحیح.

    برخلاف روش قبلی (باز و بستن دستی <div> در دو فراخوانی جدای
    st.markdown که باعث می‌شد محتوای واقعی بیرون از کادر سفید قرار
    بگیرد)، این تابع از st.container(key=...) استفاده می‌کند که یک عنصر
    واقعی و پایدار در DOM می‌سازد. هر چیزی که داخل بلوک `with emdad_card(...)`
    نوشته شود (تیتر HTML، selectbox، text_area، slider، دکمه و ...)
    به‌صورت واقعی فرزند همان عنصر DOM خواهد بود و داخل کادر سفید نمایش
    داده می‌شود.

    نکته: کلید (key) باید در کل صفحه یکتا باشد و طبق قرارداد این پروژه
    با پیشوند «emdad_card» شروع شود تا CSS مربوطه در inject_custom_css
    به‌درستی روی آن اعمال شود.
    """
    if not key.startswith("emdad_card"):
        key = f"emdad_card_{key}"
    with st.container(key=key) as container:
        yield container


# --------------------------------------------------------------------------- #
# هدر مینیمال
# --------------------------------------------------------------------------- #
def render_header() -> None:
    """رسم هدر رسمی و مینیمال بالای صفحه (راست‌چین، لوگوی سازمانی)."""
    logo_base64 = _load_logo_base64()
    if logo_base64:
        icon_html = (
            f'<img src="data:image/png;base64,{logo_base64}" '
            f'alt="لوگوی هلال احمر خراسان رضوی" class="emdad-header-logo">'
        )
    else:
        # آیکون جایگزین در صورتی که assets/logo.png یافت نشود
        icon_html = '<span class="emdad-header-icon">🌙</span>'

    st.markdown(
        f"""
        <div class="emdad-header">
            <div class="emdad-header-text">
                <p class="emdad-header-title">
                    <span>امدادیار رضوی</span>
                </p>
                <p class="emdad-header-subtitle">دستیار هوشمند آموزشی جمعیت هلال احمر خراسان رضوی</p>
            </div>
            <div class="emdad-header-icon-wrap">{icon_html}</div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# نقشه ساده استان (SVG)
# --------------------------------------------------------------------------- #
def build_province_map_svg(selected_county: Optional[str]) -> str:
    """
    ساخت یک نقشه ساده و سبک‌وزن (SVG) از شهرستان‌های خراسان رضوی.

    این نقشه از مختصات جغرافیایی دقیق پیروی نمی‌کند و صرفاً یک نمایش
    بصری ساده و سازمانی برای کمک به انتخاب شهرستان است. شهرستان انتخاب‌شده
    با رنگ طلایی و افکت درخشش (Glow) برجسته می‌شود.

    Args:
        selected_county: نام شهرستانی که کاربر انتخاب کرده (برای Highlight).

    Returns:
        رشته HTML/SVG آماده رندر با st.markdown.
    """
    markers = []
    for county, (x, y) in COUNTY_POSITIONS.items():
        is_selected = county == selected_county
        fill = BRAND["gold"] if is_selected else BRAND["red"]
        radius = 9 if is_selected else 6
        glow = (
            f'<circle cx="{x}" cy="{y}" r="16" fill="{BRAND["gold"]}" opacity="0.25" />'
            if is_selected
            else ""
        )
        label_weight = "700" if is_selected else "400"
        label_fill = BRAND["red_dark"] if is_selected else BRAND["text_dark"]

        markers.append(
            f"""
            {glow}
            <circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="#FFFFFF" stroke-width="1.5" />
            <text x="{x + 12}" y="{y + 4}" font-size="11" font-weight="{label_weight}"
                  fill="{label_fill}" font-family="Vazirmatn, Tahoma, sans-serif">{county}</text>
            """
        )

    markers_svg = "\n".join(markers)

    svg = f"""
    <div style="background-color:#FFFFFF; border:1px solid {BRAND['border']};
                border-radius:12px; padding:14px; display:flex; justify-content:center;
                width:100%;">
        <svg viewBox="0 0 360 440" width="100%" height="420"
             preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="340" height="420" rx="18"
                  fill="{BRAND['cream']}" stroke="{BRAND['gold_light']}" stroke-width="2" />
            <text x="180" y="34" text-anchor="middle" font-size="14" font-weight="700"
                  fill="{BRAND['red_dark']}" font-family="Vazirmatn, Tahoma, sans-serif">
                نقشه ساده استان خراسان رضوی
            </text>
            {markers_svg}
        </svg>
    </div>
    """
    return svg


# --------------------------------------------------------------------------- #
# اتصال به Backend (RAG Pipeline)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="در حال بارگذاری پایگاه دانش امدادیار رضوی ...")
def load_prompt_builder() -> PromptBuilder:
    """
    بارگذاری یک‌باره (Cached) اجزای Backend: مدل Embedding، ایندکس FAISS، Retriever و PromptBuilder.

    این تابع فقط یک‌بار در طول عمر Session اجرا می‌شود تا از بارگذاری مجدد
    مدل سنگین Embedding در هر تعامل کاربر جلوگیری شود.
    """
    embedder = EmbeddingModel()
    store = FAISSVectorStore(embedding_model=embedder, index_dir=VECTOR_STORE_DIR)
    store.load()
    retriever = Retriever(vector_store=store, default_top_k=5)
    return PromptBuilder(retriever=retriever)


def generate_llm_response(prompt: str) -> str:
    """
    ارسال Prompt نهایی به LLM و دریافت پاسخ.

    این تابع محل اتصال به ماژول generation/llm_client.py است که در مرحله
    بعد پیاده‌سازی می‌شود. تا پیش از آماده‌شدن آن ماژول، یک پیام موقت
    نمایش داده می‌شود تا رابط کاربری بدون خطا قابل تست باشد.
    """
    try:
        from generation.llm_client import get_completion  # وارد کردن تنبل (Lazy Import)

        return get_completion(prompt)
    except ImportError:
        logger.warning("ماژول generation.llm_client هنوز پیاده‌سازی نشده است.")
        return (
            "⚠️ ماژول اتصال به مدل زبانی (llm_client) هنوز به پروژه اضافه نشده است.\n\n"
            "Prompt نهایی با موفقیت ساخته شد و آماده ارسال به LLM است."
        )


# --------------------------------------------------------------------------- #
# رندر بخش نمایش پاسخ
# --------------------------------------------------------------------------- #
def render_response(result: PromptResult) -> None:
    """نمایش پاسخ نهایی یا پیام Fallback، به‌همراه برچسب منابع استفاده‌شده."""
    with emdad_card("emdad_card_response"):
        st.markdown('<p class="emdad-card-title">📋 نتیجه</p>', unsafe_allow_html=True)

        if not result.is_answerable:
            st.markdown(
                f'<div class="emdad-fallback-box">{result.fallback_message}</div>',
                unsafe_allow_html=True,
            )
            return

        with st.spinner("در حال تولید پاسخ توسط مدل زبانی ..."):
            answer_text = generate_llm_response(result.prompt)

        st.markdown(f'<div class="emdad-answer-box">{answer_text}</div>', unsafe_allow_html=True)

        if result.retrieved_chunks:
            st.markdown("<div style='margin-top:14px;'>منابع مورد استفاده:</div>", unsafe_allow_html=True)
            tags_html = "".join(
                f'<span class="emdad-source-tag">{chunk.source_file} - ص {chunk.page_number}</span>'
                for chunk in result.retrieved_chunks
            )
            st.markdown(tags_html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# برنامه اصلی
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_custom_css()
    render_header()

    # مقداردهی اولیه Session State
    if "selected_county" not in st.session_state:
        st.session_state.selected_county = None

    # ---------------- ۱. انتخاب نوع درخواست ---------------- #
    with emdad_card("emdad_card_request_type"):
        st.markdown('<p class="emdad-card-title">۱. نوع درخواست را انتخاب کنید</p>', unsafe_allow_html=True)
        request_type_label = st.radio(
            label="نوع درخواست",
            options=list(REQUEST_TYPE_LABELS.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
    request_type = REQUEST_TYPE_LABELS[request_type_label]

    # ---------------- ۲. تنظیمات اختصاصی حالت سناریو ---------------- #
    county: Optional[str] = None
    season: Optional[str] = None
    incident_type: Optional[str] = None

    if request_type == RequestType.SCENARIO:
        map_col, form_col = st.columns([1.1, 1], gap="large")

        with map_col:
            with emdad_card("emdad_card_map"):
                st.markdown(
                    '<p class="emdad-card-title">۲. شهرستان مورد نظر را از نقشه انتخاب کنید</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    build_province_map_svg(st.session_state.selected_county),
                    unsafe_allow_html=True,
                )
                county = st.selectbox(
                    "انتخاب شهرستان",
                    options=list(COUNTY_POSITIONS.keys()),
                    index=None,
                    placeholder="یک شهرستان را انتخاب کنید ...",
                )
                st.session_state.selected_county = county

        with form_col:
            with emdad_card("emdad_card_scenario_details"):
                st.markdown('<p class="emdad-card-title">۳. فصل و نوع حادثه</p>', unsafe_allow_html=True)
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
    with emdad_card("emdad_card_text_input"):
        if request_type == RequestType.SCENARIO:
            st.markdown(
                '<p class="emdad-card-title">۴. توضیحات تکمیلی (اختیاری)</p>', unsafe_allow_html=True
            )
            placeholder = "در صورت نیاز، جزئیات یا محدودیت‌های خاص سناریو را اینجا بنویسید ..."
        else:
            st.markdown('<p class="emdad-card-title">۲. متن درخواست خود را وارد کنید</p>', unsafe_allow_html=True)
            placeholder = (
                "مثال: اقدامات اولیه در مواجهه با زلزله چیست؟"
                if request_type == RequestType.QA
                else "مثال: برای مبحث کمک‌های اولیه، ۵ سؤال ارزیابی طراحی کن."
            )

        user_text = st.text_area(
            "متن درخواست",
            placeholder=placeholder,
            height=110,
            label_visibility="collapsed",
        )
        submit_clicked = st.button("🔍 ارسال درخواست", use_container_width=False)

    # ---------------- منطق ارسال به Backend ---------------- #
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
            prompt_builder = load_prompt_builder()
            result = prompt_builder.build(user_input=final_query)
            render_response(result)
        except (VectorStoreError, RetrieverNotReadyError) as exc:
            st.error(
                "پایگاه دانش هنوز آماده نیست. لطفاً ابتدا Pipeline ساخت ایندکس "
                f"(scripts/build_index.py) را اجرا کنید.\n\nجزئیات فنی: {exc}"
            )
        except PromptBuildError as exc:
            st.error(f"خطایی در پردازش درخواست رخ داد: {exc}")

    # ---------------- فوتر ---------------- #
    st.markdown(
        '<div class="emdad-footer">امدادیار رضوی — جمعیت هلال احمر خراسان رضوی | '
        "تمام پاسخ‌ها صرفاً بر اساس منابع آموزشی رسمی سازمان تولید می‌شوند.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
