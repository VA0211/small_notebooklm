import streamlit as st
from src.indexing import ingest, save_and_ingest_pdf
from src.rag import answer
from src.learning import summarize, generate_quiz, generate_flashcards
from src.export import to_markdown

st.set_page_config(page_title="Simple NotebookLM", layout="wide")

st.title("Simple NotebookLM")

# Sidebar - Quản lý tài liệu
with st.sidebar:
    st.header("Tài liệu của bạn")
    uploaded_file = st.file_uploader("Tải lên PDF mới", type="pdf")
    if uploaded_file:
        with st.spinner("Đang xử lý tài liệu..."):
            res = save_and_ingest_pdf(uploaded_file.read(), uploaded_file.name)
            st.success(f"Đã index {res['chunks_indexed']} đoạn văn bản!")
    
    if st.button("Index lại toàn bộ dữ liệu"):
        count = ingest(recreate=True)
        st.success(f"Đã xử lý xong {count} chunks!")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Hỏi đáp (Chat)", "Tóm tắt", "Quiz", "Flashcards"])

with tab1:
    query = st.text_input("Nhập câu hỏi về tài liệu:")
    if query:
        with st.spinner("Đang tìm câu trả lời..."):
            res = answer(query)
            st.markdown(res.answer)
            with st.expander("Xem trích dẫn"):
                for cite in res.citations:
                    st.write(f"**{cite.source_marker}**: {cite.filename} (Trang {cite.page})")

with tab2:
    if st.button("Tạo tóm tắt toàn bộ"):
        with st.spinner("Đang tóm tắt..."):
            s = summarize()
            st.subheader("Bản tóm tắt")
            st.write(s.summary)
            st.subheader("Các ý chính")
            for kp in s.key_points:
                st.write(f"- {kp}")

with tab3:
    count = st.slider("Số lượng câu hỏi", 1, 10, 5)
    if st.button("Tạo bộ câu hỏi"):
        with st.spinner("Đang tạo quiz..."):
            qset = generate_quiz(count=count)
            for i, item in enumerate(qset.items):
                st.write(f"**Câu {i+1}: {item.question}**")
                ans = st.radio(f"Chọn đáp án cho câu {i+1}", item.options, key=f"q_{i}")
                if st.button(f"Kiểm tra câu {i+1}"):
                    if item.options.index(ans) == item.correct_index:
                        st.success("Đúng!")
                    else:
                        st.error(f"Sai! Đáp án đúng là: {item.options[item.correct_index]}")
                    st.info(item.explanation)

with tab4:
    if st.button("Tạo Flashcards"):
        with st.spinner("Đang tạo thẻ..."):
            fset = generate_flashcards()
            for card in fset.cards:
                with st.expander(f"Mặt trước: {card.front}"):
                    st.write(f"**Mặt sau:** {card.back}")
                    if card.hint: st.caption(f"Gợi ý: {card.hint}")