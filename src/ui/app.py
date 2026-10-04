"""
رابط کاربری Streamlit (فارسی، راست‌به‌چپ)
=========================================
اجرا از ریشهٔ پروژه:  python run_app.py

چیدمان:
  - صفحهٔ اصلی: چت + نمایش جریانی پاسخ مدل
  - سایدبار: مدیریت گفتگوها، تنظیمات (عمق حافظه/top-k/دما)،
    وضعیت دانش‌نامه و منابعِ آخرین پاسخ
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
from src.core.rag_pipeline import (EVENT_LLM_ROUND, EVENT_RETRIEVAL_DONE,
                                   EVENT_TOOL_CALL, EVENT_TOOL_RESULT,
                                   EVENT_WARNING, RAGPipeline)
from src.core.retriever import Retriever
from src.core.tools import ToolRegistry
from src.core.vector_store import VectorStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# ═══════════════════════════ تنظیمات صفحه ═══════════════════════════
st.set_page_config(page_title="دستیار دانش‌نامه", layout="wide",
                   initial_sidebar_state="expanded")

PERSIAN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap');

html, body, .stApp, .block-container,
button, input, textarea, select, label {
    font-family: 'Vazirmatn', 'Tahoma', 'Segoe UI', sans-serif;
}
.stApp, .block-container, section[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    direction: rtl;
}
[data-testid="stChatMessage"], [data-testid="stChatInput"] {
    direction: rtl;
}
[data-testid="stChatMessage"] div.stMarkdown,
[data-testid="stChatMessage"] p {
    direction: rtl;
    text-align: right;
}
h1, h2, h3, h4, h5 {
    text-align: right;
    direction: rtl;
}
pre, code, [data-testid="stCodeBlock"] {
    direction: ltr;
    text-align: left;
}
.stSlider > div {
    direction: ltr;
}
hr {
    border: 0.5px solid #d0d0d0;
}
</style>
"""
st.markdown(PERSIAN_CSS, unsafe_allow_html=True)


# ═══════════════════════════ ساخت سرویس‌ها ═══════════════════════════
@st.cache_resource(show_spinner=False)
def build_services(settings: Settings) -> dict:
    """ساخت یک‌بارهٔ اجزای سیستم (cache شده بین rerun ها)."""
    vector_store = VectorStore(settings.index_dir)
    vector_store.try_load()
    embedding_client = EmbeddingClient(settings)
    embedding_service = EmbeddingService(embedding_client, settings.embed_batch_size)
    retriever = Retriever(embedding_service, vector_store)
    tools = ToolRegistry(retriever, vector_store, settings.tool_result_max_chars)
    llm = LLMClient(settings)
    pipeline = RAGPipeline(llm, retriever, tools, settings)
    history = ChatHistory(settings.history_db_path)
    return {
        "vector_store": vector_store,
        "pipeline": pipeline,
        "history": history,
    }


SETTINGS = Settings.load()
SERVICES = build_services(SETTINGS)
history: ChatHistory = SERVICES["history"]
pipeline: RAGPipeline = SERVICES["pipeline"]
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


init_session_state()


def load_sources_for(session_id: int) -> list:
    """منابعِ ذخیره‌شدهٔ آخرین پاسخِ یک گفتگو."""
    return history.last_assistant_meta(session_id).get("sources", [])


# ═══════════════════════════ سایدبار ═══════════════════════════
with st.sidebar:
    st.subheader("گفتگوها")

    if st.button("+ گفتگوی جدید", use_container_width=True):
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
        if col_del.button("حذف", key=f"del_{session.id}",
                          help="حذف این گفتگو"):
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

    st.divider()
    st.subheader("تنظیمات پاسخ‌دهی")
    memory_depth = st.slider(
        "عمق حافظه (تعداد پیام‌های گذشته)", 0, 20, 10,
        help="چند پیام از تاریخچهٔ این گفتگو به مدل داده شود"
    )
    top_k = st.slider(
        "تعداد نتایج جست‌وجو (top-k)", 1, 15, SETTINGS.default_top_k,
        help="چند چانک مرتبط هر بار بازیابی شود"
    )
    temperature = st.slider(
        "میزان خلاقیت (temperature)", 0.0, 1.0, SETTINGS.temperature, 0.05
    )

    st.divider()
    st.subheader("دانش‌نامه")
    kb_stats = vector_store.stats()
    if kb_stats["ready"]:
        st.success(
            f"ایندکس آماده است\n\n"
            f"- چانک‌ها: {kb_stats['count']}\n"
            f"- فایل‌ها: {len(kb_stats.get('sources', []))}\n"
            f"- ابعاد بردار: {kb_stats['dimension']}\n"
            f"- ساخته‌شده: {kb_stats.get('created_at', '-')}"
        )
    else:
        st.error("ایندکس FAISS هنوز ساخته نشده است.")
        st.caption("ابتدا دیتاست را embed کنید:")
        st.code("python scripts/embed_dataset.py --data data/data.csv",
                language="bash")
    if st.button("بازخوانی ایندکس", use_container_width=True,
                 help="پس از ساخت مجدد ایندکس روی دیسک"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("منابع آخرین پاسخ")
    sources = st.session_state.get("last_sources", [])
    if not sources:
        st.caption("هنوز پاسخی تولید نشده است.")
    for index, source in enumerate(sources, start=1):
        label = f"{index}. {source.get('source', '-')} — شباهت {source.get('score', 0):.2f}"
        with st.expander(label):
            text = source.get("text", "")
            st.caption(text[:700] + ("…" if len(text) > 700 else ""))


# ═══════════════════════════ صفحهٔ چت ═══════════════════════════
st.title("دستیار دانش‌نامه")
st.caption("سیستم RAG محلی — Qwen3 + bge-m3 + FAISS")

if not kb_stats["ready"]:
    st.warning(
        "هنوز هیچ سندی در دانش‌نامه نیست. پس از ساخت ایندکس، پاسخ‌ها بر اساس "
        "اسناد شما داده می‌شود؛ تا آن موقع مدل فقط با دانش عمومی و ابزار جست‌وجو "
        "پاسخ می‌دهد."
    )

# نمایش تاریخچهٔ گفتگوی فعال
for message in history.get_messages(st.session_state.session_id):
    with st.chat_message(message.role):
        st.markdown(message.content)

# ورودی سوال
prompt = st.chat_input("سوال خود را بپرسید…")

if prompt:
    # تاریخچه برای مدل (با عمق حافظهٔ انتخاب‌شده)
    history_rows = history.get_messages(st.session_state.session_id,
                                        limit=memory_depth)
    chat_history = ChatHistory.as_chat_messages(history_rows)

    # ثبت پیام کاربر
    history.add_message(st.session_state.session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # اجرای خط لوله + نمایش جریانی
    with st.chat_message("assistant"):
        status = st.status("در حال جست‌وجو در دانش‌نامه…", expanded=False)

        def on_event(event: str, **data) -> None:
            """به‌روزرسانی نوار وضعیت بر اساس رویدادهای خط لوله."""
            if event == EVENT_RETRIEVAL_DONE:
                status.update(label=f"بافت اولیه: {data['count']} نتیجه بازیابی شد")
            elif event == EVENT_TOOL_CALL:
                status.update(label=f"مدل در حال استفاده از ابزار {data['name']} است")
                status.write(f"فراخوانی ابزار: {data['name']}({data['arguments']})")
            elif event == EVENT_TOOL_RESULT:
                status.write(f"نتیجهٔ ابزار {data['name']}: {data['summary']}…")
            elif event == EVENT_LLM_ROUND and data.get("forced_final"):
                status.update(label="تولید پاسخ نهایی…")
            elif event == EVENT_WARNING:
                status.write(f"هشدار: {data['message']}")

        runner = pipeline.create_run(prompt, chat_history, top_k=top_k,
                                     temperature=temperature)
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
