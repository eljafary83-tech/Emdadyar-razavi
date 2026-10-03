"""Reusable visual components for the Streamlit entry point; no backend imports."""
from __future__ import annotations
import html
from typing import Optional
import streamlit as st
from generation.intent_classifier import RequestType

BRAND = {"red":"#C8102E", "red_dark":"#8C0F22", "gold":"#C9A24B", "cream":"#FBF7EF", "border":"#E7DFCF"}
SEASONS = ["بهار", "تابستان", "پاییز", "زمستان"]
INCIDENT_TYPES = ["امدادرسانی به زائران در ایام شلوغی", "تصادف جاده‌ای", "کولاک و بهمن در محورهای کوهستانی", "سیلاب", "گم‌شدن افراد در مناطق طبیعی", "آتش‌سوزی", "زلزله", "سایر حوادث"]
REQUEST_TYPE_LABELS = {"پاسخ‌گویی به سؤال":RequestType.QA, "طراحی سناریوی آموزشی":RequestType.SCENARIO, "تولید سؤالات ارزیابی":RequestType.QUIZ}
COUNTY_POSITIONS = {"مشهد":(215,210), "نیشابور":(110,130), "قوچان":(185,105), "تربت جام":(245,300), "سبزوار":(55,110), "تربت حیدریه":(120,255), "خواف":(170,360), "سرخس":(300,150)}

def safe_text_to_html(value: str) -> str:
    """Escape untrusted LLM/PDF text while preserving line breaks."""
    return html.escape(value).replace("\n", "<br>")

def inject_custom_css() -> None:
    st.markdown(f"""<style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    html,body,[class*='css']{{font-family:Vazirmatn,Tahoma,sans-serif!important;direction:rtl}} .emdad-header,.emdad-card,[class*='st-key-emdad_']{{background:#fff;border:1px solid {BRAND['border']};border-radius:12px;padding:18px;margin:10px 0}} .emdad-header{{border-top:5px solid {BRAND['red']};display:flex;justify-content:space-between}} .emdad-title{{color:{BRAND['red_dark']};font-weight:800;font-size:29px;margin:0}} .emdad-source{{display:inline-block;background:{BRAND['cream']};padding:4px 8px;margin:3px;border-radius:5px}} </style>""", unsafe_allow_html=True)

def render_header() -> None:
    st.markdown("<div class='emdad-header'><div><p class='emdad-title'>امدادیار رضوی</p><span>دستیار هوشمند آموزشی جمعیت هلال احمر خراسان رضوی</span></div><div>🌙</div></div>", unsafe_allow_html=True)

def build_province_map_svg(selected_county: Optional[str]) -> str:
    circles = "".join(f"<circle cx='{x}' cy='{y}' r='{9 if c == selected_county else 6}' fill='{BRAND['gold'] if c == selected_county else BRAND['red']}'/><text x='{x+10}' y='{y+4}' font-size='10'>{c}</text>" for c,(x,y) in COUNTY_POSITIONS.items())
    return f"<svg viewBox='0 0 360 440' width='100%' height='300'><rect x='5' y='5' width='350' height='430' rx='18' fill='{BRAND['cream']}'/>{circles}</svg>"
