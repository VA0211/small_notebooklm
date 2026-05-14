import httpx
import streamlit as st

from config import settings

_API = settings.api_url

def _api(method, path, **kwargs):
    return httpx.request(method, f"{_API}{path}", timeout=120, **kwargs)

def run():
    st.set_page_config(page_title="RAG Learning System", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    filenames, page = _sidebar()
    tabs = st.tabs(["Hỏi đáp", "Tóm tắt", "Quiz", "Flashcards"])

    for tab, fn in zip(tabs, [_tab_chat, _tab_summary, _tab_quiz, _tab_flashcards]):
        with tab:
            fn(filenames, page)

run()