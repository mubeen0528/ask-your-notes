"""Run with: python -m streamlit run app.py"""
import streamlit as st

from notes_engine import answer_question, build_index

st.set_page_config(page_title="Ask Your Notes", page_icon="📚")
st.title("Ask Your Notes")
st.write("Ask a question about the .txt and .md notes in your library folder.")
st.caption("Retrieval happens locally. When you click Ask and passages match, your question "
           "and selected excerpts are sent to Google to generate an answer.")

if "notes_index" not in st.session_state:
    st.session_state.notes_index = build_index()

if st.button("Reload Notes"):
    with st.spinner("Reloading local notes…"):
        st.session_state.notes_index = build_index()
    st.success("Notes reloaded. Your last result is kept below.")

index = st.session_state.notes_index
st.write(f"Loaded files: {index.file_count}")
for warning in index.warnings:
    st.warning(warning)
if not index.file_count:
    st.info("Add non-empty .txt or .md files to library, then click Reload Notes.")

with st.form("ask_form"):
    question = st.text_input("Your question", placeholder="What is a Git commit?")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Finding passages and preparing your answer…"):
            st.session_state.last_result = answer_question(index, question.strip())
            st.session_state.last_question = question.strip()

if "last_result" in st.session_state:
    result = st.session_state.last_result
    st.subheader("Answer")
    st.caption(f"Last submitted question: {st.session_state.last_question}")
    if result.is_error:
        st.error(result.answer)
    else:
        st.markdown(result.answer)
    st.subheader("Retrieved sources")
    st.caption("These are the passages selected for the last question. Retrieval does not "
               "prove every statement in the answer; check citations against the excerpts.")
    sources = sorted({doc.metadata["source"] for doc in result.passages})
    if sources:
        for source in sources:
            st.text(source)
        for number, doc in enumerate(result.passages, start=1):
            with st.expander(f"[S{number}] {doc.metadata['source']}"):
                st.text(doc.page_content)
    else:
        st.write("No sources matched.")
