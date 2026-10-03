"""Streamlit entry point for the unified RAG + LLM assistant."""
from __future__ import annotations
import streamlit as st

from config import Settings
from generation.intent_classifier import RequestType
from llm.client import APIConfig, ChatCompletionsClient
from llm.provider import GenericChatProvider
from rag.index import SentenceTransformerEmbeddings
from rag.retriever import Retriever
from services.assistant import AssistantResponse, AssistantService
from ui.streamlit_app import COUNTY_POSITIONS, INCIDENT_TYPES, REQUEST_TYPE_LABELS, SEASONS, build_province_map_svg, inject_custom_css, render_header, safe_text_to_html

st.set_page_config(page_title="امدادیار رضوی | هلال احمر خراسان رضوی", page_icon="🌙", layout="wide", initial_sidebar_state="collapsed")

def card(key: str):
    """Native Streamlit container; widgets are genuinely nested in one card."""
    return st.container(key=key)

@st.cache_resource(show_spinner="در حال بارگذاری مدل embedding ...")
def get_embeddings(model_name: str) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name)

def get_service() -> AssistantService:
    """Create a lightweight service per request so .env/index changes are observed."""
    settings = Settings.from_environment()
    embeddings = get_embeddings(settings.embedding_model)
    retriever = Retriever(settings.knowledge_base_dir, settings.index_dir, embeddings,
                          settings.retrieval_top_k, settings.retrieval_threshold, settings.chunk_size,
                          settings.chunk_overlap, settings.min_chunk_length)
    provider = GenericChatProvider(ChatCompletionsClient(APIConfig(
        settings.llm_api_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_seconds)))
    return AssistantService(retriever, provider)

def render_result(result: AssistantResponse) -> None:
    with card("emdad_result"):
        st.subheader("📋 نتیجه")
        # Escape untrusted model/PDF-derived text: only fixed markup is rendered.
        st.markdown(f"<div>{safe_text_to_html(result.answer)}</div>", unsafe_allow_html=True)
        cited_sources = [item for index, item in enumerate(result.sources, 1) if f"src_{index}" in result.citation_ids]
        if cited_sources:
            st.markdown("**منابع بازیابی‌شده:**")
            st.markdown("".join(f"<span class='emdad-source'>{safe_text_to_html(item.chunk.source_file)} — ص {item.chunk.page_number} (شباهت {item.score:.2f})</span>" for item in cited_sources), unsafe_allow_html=True)

def main() -> None:
    inject_custom_css(); render_header()
    with card("emdad_request_type"):
        label = st.radio("نوع درخواست", list(REQUEST_TYPE_LABELS), horizontal=True)
    kind = REQUEST_TYPE_LABELS[label]
    county = season = incident = None
    if kind is RequestType.SCENARIO:
        left, right = st.columns(2)
        with left:
            county = st.selectbox("شهرستان", list(COUNTY_POSITIONS), index=None)
            st.markdown(build_province_map_svg(county), unsafe_allow_html=True)
        with right:
            season = st.selectbox("فصل", SEASONS, index=None)
            incident = st.selectbox("نوع حادثه", INCIDENT_TYPES, index=None)
    with card("emdad_input"):
        count = st.slider("تعداد سؤال", 3, 10, 5) if kind is RequestType.QUIZ else 5
        text = st.text_area("متن درخواست یا توضیحات تکمیلی", height=110)
        submitted = st.button("🔍 ارسال درخواست")
    if not submitted: return
    if kind is RequestType.SCENARIO:
        if not (county and season and incident): st.warning("شهرستان، فصل و نوع حادثه را انتخاب کنید."); return
        query = f"یک سناریوی آموزشی برای {incident} در {county} در فصل {season} طراحی کن. {text.strip()}"
    else: query = text.strip()
    if not query: st.warning("متن درخواست را وارد کنید."); return
    try:
        with st.spinner("در حال بازیابی منابع و تولید پاسخ ..."):
            result = get_service().answer(query, request_type=kind, num_questions=count)
        render_result(result)
    except Exception:
        st.error("آماده‌سازی پایگاه دانش با خطا مواجه شد. تنظیمات و فایل‌های PDF را بررسی کنید.")

if __name__ == "__main__": main()
