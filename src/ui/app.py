"""
رابط کاربری Streamlit (فارسی، راست‌به‌چپ) — طراحی مدرن
======================================================
اجرا از ریشهٔ پروژه:  python run_app.py

چیدمان:
  - سایدبار (سمتِ چپ صفحه): برند، مدیریت گفتگوها، تنظیمات پاسخ‌دهی،
    وضعیت دانش‌نامه و منابعِ آخرین پاسخ
  - صفحهٔ اصلی: چت با حباب‌های پیام مدرن + نمایش جریانی پاسخ مدل
  - جهت همهٔ متن‌ها راست‌به‌چپ است؛ کد و بلوک‌های فنی چپ‌به‌راست می‌مانند.

نکتهٔ معماری: جست‌وجوی دانش‌نامه فقط با تصمیم خودِ مدل و به‌صورت
tool call انجام می‌شود؛ متن پیام‌ها هرگز به‌صورت خودکار امبد نمی‌شود.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ریشهٔ پروژه به sys.path اضافه می‌شود تا پکیج src قابل import باشد
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging

import streamlit as st

from src.config import Settings
from src.core.api_client import EmbeddingClient, LLMClient
from src.core.embedder import EmbeddingService
from src.core.history import ChatHistory
from src.core.rag_pipeline import (EVENT_LLM_ROUND, EVENT_THINKING,
                                   EVENT_TOOL_CALL, EVENT_TOOL_RESULT,
                                   EVENT_WARNING, RAGPipeline)
from src.core.retriever import Retriever
from src.core.tools import ToolRegistry
from src.core.vector_store import VectorStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# ═══════════════════════════ تنظیمات صفحه ═══════════════════════════
st.set_page_config(page_title="دانش‌نامه", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════ آیکون‌های SVG داخلی ═══════════════════════════
# آیکون‌ها به‌صورت SVG خطی (سبک feather) تولید می‌شوند تا هیچ وابستگی‌ای به
# فونت ایموجی دستگاه کاربر یا CDN های فیلترشده نباشد — همه‌جا یکسان رندر می‌شوند.
_SVG_PATHS = {
    "check":  '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "x":      '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    "box":    '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>',
    "file":   '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "clock":  '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "book":   '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "chat":   '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
}


def icon(name: str, size: int = 15, color: str = "currentColor") -> str:
    """ساخت یک آیکون SVG خطی (بدون هیچ وابستگی خارجی)."""
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round" style="flex-shrink:0;vertical-align:-2px">'
            f'{_SVG_PATHS[name]}</svg>')


# ═══════════════════════════ فونت وزیرمتن (باندل‌شده، بدون CDN) ═══════════════════════════
@st.cache_data(show_spinner=False)
def vazirmatn_font_face() -> str:
    """ساخت @font-face با data-URI از فایل باندل‌شدهٔ پروژه.

    قبلاً فونت از fonts.googleapis.com لود می‌شد که در ایران در دسترس نیست و
    باعث می‌شد ظاهر اپ خراب شود؛ الان فونت داخل خود پروژه است (assets/fonts)
    و به‌صورت base64 داخل CSS قرار می‌گیرد؛ اگر فایل نبود، فونت‌های جایگزین
    سیستم استفاده می‌شوند.
    """
    import base64
    font_path = ROOT / "assets" / "fonts" / "Vazirmatn-var.woff2"
    try:
        data = font_path.read_bytes()
        if data[:4] != b"wOF2" or len(data) < 50000:
            raise ValueError("فایل فونت نامعتبر است")
        encoded = base64.b64encode(data).decode("ascii")
        return ("@font-face{font-family:'Vazirmatn';font-style:normal;"
                "font-weight:300 900;font-display:swap;"
                f"src:url('data:font/woff2;base64,{encoded}') format('woff2');}}")
    except Exception:
        return ""


# ═══════════════════════════ استایل مدرن ═══════════════════════════
# نکتهٔ مهم چیدمان: روی .stApp جهتِ ltr حفظ شده تا سایدبار سمتِ چپ بماند؛
# جهتِ rtl فقط به محتوای داخل (متن‌ها، حباب‌های چت، سایدبار) داده می‌شود.
APP_CSS = """
<style>
/*__FONT_FACE__*/

:root{
  --bg:#F4F6FB;
  --surface:#FFFFFF;
  --border:#E5E9F4;
  --text:#000000;
  --text2:#64748B;
  --accent:#6366F1;
  --accent2:#8B5CF6;
  --accent-soft:#EEF0FF;
  --radius:18px;
}

/* ─────────────── پایه و فونت ─────────────── */
html, body, .stApp, button, input, textarea, select, label,
.stMarkdown, .stButton, p, span, li, td, th, summary {
    font-family:'Vazirmatn','Segoe UI',Tahoma,sans-serif !important;
}
.stApp{
    direction:ltr; /* چیدمان کلی LTR بماند تا سایدبار سمت چپ قرار گیرد */
    background:
        radial-gradient(1100px 480px at 88% -10%, rgba(99,102,241,.10), transparent 60%),
        radial-gradient(900px 420px at -8% 108%, rgba(139,92,246,.09), transparent 55%),
        var(--bg);
}

/* ─────────────── مخفی‌کردن کروم استریملیت ─────────────── */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"]{ display:none !important; }
header[data-testid="stHeader"]{ background:transparent !important; }

/* ⚠️ نکتهٔ بسیار مهم: stToolbar عمداً مخفی نمی‌شود چون در Streamlit 1.65
   دکمهٔ «بازکردن سایدبار» بعد از جمع‌شدن، داخل همان stToolbar رندر می‌شود؛
   مخفی‌کردن کل آن باعث می‌شد سایدبار جمع‌شده دیگر قابل بازگشت نباشد. */
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"]{
    visibility:visible !important; opacity:1 !important; display:flex !important;
}
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] button{
    background:var(--surface) !important;
    border:1px solid var(--border) !important;
    border-radius:12px !important;
    box-shadow:0 8px 22px -10px rgba(15,23,42,.30) !important;
    color:var(--accent) !important;
}
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] button:hover{
    border-color:var(--accent) !important;
}

/* ─────────────── محتوای اصلی: راست‌به‌چپ ─────────────── */
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .block-container{
    direction:rtl;
}
h1,h2,h3,h4,h5,h6{ direction:rtl; text-align:right; }
p,li,td,th,summary{ direction:rtl; text-align:right; }
[data-testid="stMainBlockContainer"]{
    max-width:920px;
    margin-inline:auto;
}

/* کد و بلوک‌های فنی همیشه چپ‌به‌راست */
pre, code, kbd, samp{
    direction:ltr !important;
    text-align:left !important;
}

/* ⚠️ آیکون‌های Material استریملیت (فلش اکسپندر، آواتار چت، فلش سایدبار و…)
   با فونت «Material Symbols Rounded» رندر می‌شوند؛ قانون span بالاتر این
   فونت را بازنویسی می‌کرد و به‌جای آیکون، متن خام (مثل expand_more) دیده
   می‌شد. این قانون بعدی، فونت آیکون‌ها را برمی‌گرداند. */
span[data-testid="stIconMaterial"],
section[data-testid="stSidebar"] span[data-testid="stIconMaterial"]{
    font-family:'Material Symbols Rounded' !important;
    font-feature-settings:'liga';
}

/* ─────────────── حباب‌های چت ─────────────── */
[data-testid="stChatMessage"]{
    direction:rtl;
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:14px 18px;
    margin:12px 0;
    box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px -14px rgba(15,23,42,.10);
    animation:msgIn .25s ease;
}
@keyframes msgIn{
    from{ opacity:0; transform:translateY(8px); }
    to{ opacity:1; transform:none; }
}
[data-testid="stChatMessage"] .stMarkdown p,
[data-testid="stChatMessage"] li{
    direction:rtl; text-align:right;
}
[data-testid="stChatMessage"] p:has(.marker-user),
[data-testid="stChatMessage"] p:has(.marker-assistant){
    margin:0 !important; height:0 !important; line-height:0 !important;
}
.marker-user, .marker-assistant{ display:none; }

/* پیام کاربر: حباب گرادیانی بنفش-نیلی */
[data-testid="stChatMessage"]:has(.marker-user){
    background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 100%);
    border:none;
    box-shadow:0 12px 30px -12px rgba(99,102,241,.55);
}
[data-testid="stChatMessage"]:has(.marker-user) .stMarkdown,
[data-testid="stChatMessage"]:has(.marker-user) p,
[data-testid="stChatMessage"]:has(.marker-user) li,
[data-testid="stChatMessage"]:has(.marker-user) span{
    color:#00000 !important;
}

/* نشانه‌گر زندهٔ تایپ در حباب دستیار */
.stStreamingText p:last-child::after{
    content:'▍';
    color:var(--accent);
    animation:blink 1s steps(1) infinite;
}
@keyframes blink{ 50%{ opacity:0; } }

/* ─────────────── ورودی چت ─────────────── */
[data-testid="stBottom"], [data-testid="stBottom"] > div{
    background:transparent !important;
}
[data-testid="stBottomBlockContainer"]{
    max-width:920px; margin-inline:auto;
}
[data-testid="stChatInput"]{
    direction:rtl;
    border-radius:16px !important;
    border:1px solid var(--border) !important;
    background:var(--surface) !important;
    box-shadow:0 14px 34px -16px rgba(15,23,42,.22) !important;
}
[data-testid="stChatInput"] textarea{
    direction:rtl; text-align:right;
    font-family:'Vazirmatn',Tahoma,sans-serif !important;
}
[data-testid="stChatInput"]:focus-within{
    border-color:var(--accent) !important;
    box-shadow:0 0 0 4px rgba(99,102,241,.16) !important;
}

/* ─────────────── دکمه‌ها ─────────────── */
.stButton > button{
    border-radius:12px;
    font-weight:600;
    transition:all .18s ease;
}
[data-testid="stMain"] .stButton > button{
    background:var(--surface);
    border:1px solid var(--border);
    color:var(--text);
}
[data-testid="stMain"] .stButton > button:hover{
    border-color:var(--accent);
    color:var(--accent);
    transform:translateY(-1px);
    box-shadow:0 8px 20px -10px rgba(99,102,241,.45);
}

/* ─────────────── نوار وضعیت و هشدارها ─────────────── */
[data-testid="stStatus"]{
    direction:rtl;
    border:1px solid var(--border);
    border-radius:14px;
    background:#FBFCFF;
}
[data-testid="stStatus"] p, [data-testid="stStatus"] div{ text-align:right; }
[data-testid="stAlert"]{ direction:rtl; border-radius:14px; }
[data-testid="stAlert"] p{ text-align:right; }

/* ─────────────── سایدبار (سمت چپ، تم تیره) ─────────────── */
section[data-testid="stSidebar"]{
    direction:ltr;              /* خودِ ستون LTR بماند تا جای سایدبار عوض نشود */
    background:linear-gradient(180deg,#0B1220 0%,#131C36 100%) !important;
    border:none;
    box-shadow:1px 0 0 rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] *{
    font-family:'Vazirmatn',Tahoma,sans-serif !important;
}
[data-testid="stSidebarContent"]{ direction:rtl; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
    color:#F1F5F9 !important; text-align:right; font-size:1rem !important;
    font-weight:700 !important; margin-bottom:.35rem;
}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] summary, section[data-testid="stSidebar"] span{
    color:#C7D0E4;
}
section[data-testid="stSidebar"] p{ text-align:right; }
section[data-testid="stSidebar"] hr{
    border-color:rgba(255,255,255,.10) !important; margin:14px 0 !important;
}
section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb{
    background:rgba(255,255,255,.16);
}

/* دکمه‌های سایدبار */
section[data-testid="stSidebar"] .stButton > button{
    background:rgba(255,255,255,.06);
    color:#E2E8F0;
    border:1px solid rgba(255,255,255,.12);
    border-radius:12px;
    font-weight:500;
}
section[data-testid="stSidebar"] .stButton > button:hover{
    background:rgba(255,255,255,.13);
    border-color:rgba(255,255,255,.24);
    transform:translateY(-1px);
}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] button[kind="primary"]{
    background:linear-gradient(135deg,#6366F1,#8B5CF6) !important;
    color:#fff !important;
    border:none !important;
    box-shadow:0 10px 24px -10px rgba(99,102,241,.65) !important;
    font-weight:700;
}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover,
section[data-testid="stSidebar"] button[kind="primary"]:hover{
    filter:brightness(1.08);
}

/* اسلایدرها: خود اسلایدر LTR، برچسب فارسی RTL */
section[data-testid="stSidebar"] [data-testid="stSlider"]{ direction:ltr; }
section[data-testid="stSidebar"] [data-testid="stSlider"] label{
    direction:rtl; text-align:right; color:#E2E8F0 !important;
}

/* اکسپندر منابع در سایدبار */
section[data-testid="stSidebar"] [data-testid="stExpander"]{
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.12) !important;
    border-radius:12px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span{
    color:#E2E8F0;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] p{
    color:#9AA7C4; font-size:.85rem;
}

/* ─────────────── اجزای سفارشی ─────────────── */
.brand{ display:flex; align-items:center; gap:12px; padding:8px 4px 16px; direction:rtl; }
.brand-logo{
    width:46px; height:46px; border-radius:14px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    font-size:23px; background:linear-gradient(135deg,#6366F1,#8B5CF6);
    box-shadow:0 12px 26px -8px rgba(99,102,241,.6);
}
.brand-name{ color:#F8FAFC; font-weight:800; font-size:1.06rem; line-height:1.3; }
.brand-sub{ color:#7C8AA5; font-size:.76rem; }

.hero{ direction:rtl; text-align:center; padding:34px 10px 6px; }
.hero h1{
    font-size:2.3rem; font-weight:900; text-align:center;
    background:linear-gradient(90deg,#4F46E5,#9333EA);
    -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent;
    margin-bottom:.4rem;
}
.hero p{ color:var(--text2); text-align:center; font-size:1.02rem; }
.empty-hint{
    direction:rtl; text-align:center; color:var(--text2);
    margin:22px 0 10px; font-weight:600;
}

.kb-card{
    direction:rtl;
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.12);
    border-radius:14px; padding:12px 14px;
}
.kb-title{ color:#F1F5F9; font-weight:700; font-size:.9rem; margin-bottom:8px;
           display:flex; align-items:center; gap:6px; }
.kb-row{ display:flex; justify-content:space-between; align-items:center;
         padding:4px 0; color:#C7D0E4; font-size:.86rem; }
.kb-row b{ color:#FFFFFF; font-weight:700; }
.kb-empty{ color:#FCA5A5; font-size:.86rem; line-height:1.9; }

.src-badge{
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(99,102,241,.16); color:#C7D2FE;
    border-radius:999px; padding:2px 10px; font-size:.78rem; font-weight:600;
}

/* ─────────────── اسکرول‌بار ─────────────── */
::-webkit-scrollbar{ width:8px; height:8px; }
::-webkit-scrollbar-thumb{ background:#C3C9D9; border-radius:8px; }
::-webkit-scrollbar-thumb:hover{ background:#9AA3B8; }
::-webkit-scrollbar-track{ background:transparent; }
</style>
"""
# جای‌گذاری @font-face فونت باندل‌شده (اگر فایل فونت موجود باشد)
APP_CSS = APP_CSS.replace("/*__FONT_FACE__*/", vazirmatn_font_face())
st.markdown(APP_CSS, unsafe_allow_html=True)

# ═══════════════════════════ ساخت سرویس‌ها ═══════════════════════════
@st.cache_resource(show_spinner=False)
def build_services(settings: Settings) -> dict:
    """ساخت یک‌بارهٔ اجزای سیستم (cache شده بین rerun ها)."""
    vector_store = VectorStore(settings.index_dir)
    vector_store.try_load()
    embedding_client = EmbeddingClient(settings)
    embedding_service = EmbeddingService(embedding_client, settings.embed_batch_size)
    retriever = Retriever(embedding_service, vector_store)
    tools = ToolRegistry(retriever, vector_store,
                         settings.tool_result_max_chars,
                         default_top_k=settings.default_top_k)
    llm = LLMClient(settings)
    pipeline = RAGPipeline(llm, tools, settings)
    history = ChatHistory(settings.history_db_path)
    return {
        "vector_store": vector_store,
        "tools": tools,
        "pipeline": pipeline,
        "history": history,
    }


SETTINGS = Settings.load()
SERVICES = build_services(SETTINGS)
history: ChatHistory = SERVICES["history"]
pipeline: RAGPipeline = SERVICES["pipeline"]
tools: ToolRegistry = SERVICES["tools"]
vector_store: VectorStore = SERVICES["vector_store"]


# ═══════════════════════════ وضعیت سشن ═══════════════════════════
def init_session_state() -> None:
    """مقداردهی اولیهٔ وضعیت سشن؛ آخرین گفتگو (یا یکی جدید) فعال می‌شود."""
    if "session_id" not in st.session_state:
        sessions = history.list_sessions()
        if sessions:
            st.session_state.session_id = sessions[0].id
        else:
            st.session_state.session_id = history.create_session().id
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = history.last_assistant_meta(
            st.session_state.session_id
        ).get("sources", [])


def load_sources_for(session_id: int) -> list:
    """منابعِ ذخیره‌شدهٔ آخرین پاسخِ یک گفتگو."""
    return history.last_assistant_meta(session_id).get("sources", [])


init_session_state()

SUGGESTIONS = [
    "سلام! چه کمکی از تو برمی‌آید؟",
    "در اسناد من چه موضوعاتی وجود دارد؟",
    "مهم‌ترین نکات گزارش‌ها را خلاصه کن",
    "دربارهٔ محتوای فایل‌های من توضیح بده",
]

AVATARS = {"user": ":material/person:", "assistant": ":material/smart_toy:"}
MARKERS = {"user": "marker-user", "assistant": "marker-assistant"}


def render_message(role: str, content: str) -> None:
    """نمایش یک پیام چت با حباب و نشانهٔ نقش (برای استایل متفاوت CSS)."""
    with st.chat_message(role, avatar=AVATARS.get(role, ":material/smart_toy:")):
        st.markdown(f'<span class="{MARKERS.get(role, "")}"></span>',
                    unsafe_allow_html=True)
        st.markdown(content)


# ═══════════════════════════ سایدبار (سمت چپ) ═══════════════════════════
with st.sidebar:
    # ── برند ──
    st.markdown(
        f"""
        <div class="brand">
            <div class="brand-logo">{icon('chat', 24, '#FFFFFF')}</div>
            <div>
                <div class="brand-name">دانش‌نامه</div>
                <div class="brand-sub">RAG محلی · Qwen3 + bge-m3</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── مدیریت گفتگوها ──
    if st.button("+ گفتگوی جدید", use_container_width=True, type="primary"):
        new_session = history.create_session()
        st.session_state.session_id = new_session.id
        st.session_state.last_sources = []
        st.rerun()

    for session in history.list_sessions():
        active = session.id == st.session_state.session_id
        col_pick, col_del = st.columns([4.2, 1])
        if col_pick.button(session.title, key=f"pick_{session.id}",
                           use_container_width=True,
                           type="primary" if active else "secondary"):
            if not active:
                st.session_state.session_id = session.id
                st.session_state.last_sources = load_sources_for(session.id)
                st.rerun()
        if col_del.button("✕", key=f"del_{session.id}", help="حذف این گفتگو"):
            history.delete_session(session.id)
            if active:
                remaining = history.list_sessions()
                st.session_state.session_id = (
                    remaining[0].id if remaining else history.create_session().id
                )
                st.session_state.last_sources = load_sources_for(
                    st.session_state.session_id
                )
            st.rerun()

    # ── تنظیمات پاسخ‌دهی ──
    st.divider()
    st.subheader("تنظیمات پاسخ‌دهی")
    memory_depth = st.slider(
        "عمق حافظه (تعداد پیام‌های گذشته)", 0, 20, 10,
        help="چند پیام از تاریخچهٔ این گفتگو به مدل داده شود"
    )
    top_k = st.slider(
        "تعداد نتایج جست‌وجو (top-k)", 1, 15, SETTINGS.default_top_k,
        help="پیش‌فرضِ تعداد نتایج؛ وقتی مدل خودش عددی تعیین نکند استفاده می‌شود"
    )
    temperature = st.slider(
        "میزان خلاقیت (temperature)", 0.0, 1.0, SETTINGS.temperature, 0.05
    )

    # ── وضعیت دانش‌نامه ──
    st.divider()
    st.subheader("دانش‌نامه")
    kb_stats = vector_store.stats()
    if kb_stats["ready"]:
        created = kb_stats.get("created_at", "-")
        st.markdown(
            f"""
            <div class="kb-card">
                <div class="kb-title">{icon('check', 16, '#4ADE80')} ایندکس آماده است</div>
                <div class="kb-row"><span>{icon('box')} تعداد چانک‌ها</span><b>{kb_stats['count']:,}</b></div>
                <div class="kb-row"><span>{icon('file')} تعداد فایل‌ها</span><b>{len(kb_stats.get('sources', []))}</b></div>
                <div class="kb-row"><span>{icon('layers')} ابعاد بردار</span><b>{kb_stats['dimension']}</b></div>
                <div class="kb-row"><span>{icon('clock')} ساخته‌شده</span><b>{created}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="kb-card">
                <div class="kb-title" style="color:#FCA5A5;">{icon('x', 16, '#FCA5A5')} ایندکس FAISS هنوز ساخته نشده است</div>
                <div class="kb-empty">ابتدا دیتاست را embed کنید:<br>
                <code style="direction:ltr;">python scripts/embed_dataset.py --data data/data.csv</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if st.button("بازخوانی ایندکس", use_container_width=True,
                 help="پس از ساخت مجدد ایندکس روی دیسک"):
        st.cache_resource.clear()
        st.rerun()

    # ── منابع آخرین پاسخ ──
    st.divider()
    sources = st.session_state.get("last_sources", [])
    if sources:
        st.markdown(
            f'<span class="src-badge">{icon("book")} منابع آخرین پاسخ · {len(sources)}</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        for index, source in enumerate(sources, start=1):
            label = (f"{index}. {source.get('source', '-')} — "
                     f"شباهت {source.get('score', 0):.2f}")
            with st.expander(label):
                text = source.get("text", "")
                st.caption(text[:700] + ("…" if len(text) > 700 else ""))
    else:
        st.subheader("منابع")
        st.caption("هر وقت دستیار از دانش‌نامه جست‌وجو کند، منابعِ استفاده‌شده "
                   "همین‌جا نمایش داده می‌شوند.")


# ═══════════════════════════ صفحهٔ چت ═══════════════════════════
messages = history.get_messages(st.session_state.session_id)

if not messages:
    # ── حالت خالی: هیرو + پیشنهادها ──
    st.markdown(
        """
        <div class="hero">
            <h1>دانش‌نامه</h1>
            <p>سوالتان را بپرسید؛ اگر پاسخ در اسناد شما باشد، دستیار خودش
            در دانش‌نامه جست‌وجو می‌کند و با ذکر منبع پاسخ می‌دهد.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not kb_stats["ready"]:
        st.warning(
            "دانش‌نامه خالی است. پس از ساخت ایندکس، پاسخ‌های مرتبط با اسناد شما "
            "بر اساس آن‌ها داده می‌شود؛ تا آن موقع مدل فقط با دانش عمومی پاسخ می‌دهد."
        )
    st.markdown('<div class="empty-hint">برای شروع یکی از این‌ها را امتحان کنید:</div>',
                unsafe_allow_html=True)
    suggestion = None
    cols = st.columns(2)
    for i, item in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(item, key=f"sug_{i}", use_container_width=True):
                suggestion = item
else:
    suggestion = None
    for message in messages:
        render_message(message.role, message.content)

# ورودی سوال
prompt = st.chat_input("سوال خود را بنویسید…")
if not prompt and suggestion:
    prompt = suggestion

if prompt:
    # تاریخچه برای مدل (با عمق حافظهٔ انتخاب‌شده)
    history_rows = history.get_messages(st.session_state.session_id,
                                        limit=memory_depth)
    chat_history = ChatHistory.as_chat_messages(history_rows)

    # ثبت پیام کاربر + نمایش
    history.add_message(st.session_state.session_id, "user", prompt)
    render_message("user", prompt)

    # پیش‌فرضِ تعداد نتایج ابزار از تنظیمات سایدبار
    tools.default_top_k = top_k

    # اجرای خط لوله + نمایش جریانی
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown('<span class="marker-assistant"></span>', unsafe_allow_html=True)
        status = st.status("در حال فکر کردن…", expanded=False)

        def on_event(event: str, **data) -> None:
            """به‌روزرسانی نوار وضعیت بر اساس رویدادهای خط لوله."""
            if event == EVENT_THINKING:
                if data.get("active"):
                    status.update(label="در حال تفکر…")
                else:
                    status.update(label="در حال نوشتن پاسخ…")
            elif event == EVENT_TOOL_CALL:
                status.update(label="در حال جست‌وجو در دانش‌نامه…")
                status.write(
                    f"**فراخوانی ابزار:** `{data['name']}` "
                    f"با آرگومان‌های `{data['arguments']}`"
                )
            elif event == EVENT_TOOL_RESULT:
                status.write(f"**نتیجهٔ {data['name']}:** {data['summary']}…")
            elif event == EVENT_LLM_ROUND and data.get("round", 1) > 1:
                status.update(label="در حال نوشتن پاسخ…")
            elif event == EVENT_WARNING:
                status.write(f"هشدار: {data['message']}")

        runner = pipeline.create_run(prompt, chat_history, temperature=temperature)
        runner.on_event = on_event
        streamed_answer = st.write_stream(runner.stream())
        status.update(label="پاسخ آماده شد", state="complete", expanded=False)

    result = runner.result
    if result.error:
        st.error(result.error)

    # متن نهایی پاسخ
    final_answer = (streamed_answer or "").strip() \
        or result.answer.strip() \
        or "پاسخی تولید نشد."

    # ذخیرهٔ پاسخ + منابع (برای نمایش در سایدبار و اجراهای بعدی)
    sources_payload = [
        {
            "source": item.source,
            "score": item.score,
            "text": item.text[:800],
        }
        for item in result.sources
    ]
    history.add_message(
        st.session_state.session_id, "assistant", final_answer,
        meta={"sources": sources_payload, "tool_calls": len(result.tool_calls)},
    )
    st.session_state.last_sources = sources_payload
    st.rerun()
