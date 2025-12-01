import streamlit as st
from rag_pipeline import run_guarded_rag

st.title("RAG Chatbot with Guardrails")

query = st.text_input("Ask your question:")
context = st.text_area("Provide context (from retrieved docs):")

if st.button("Get Answer"):
    answer, validation = run_guarded_rag(query, context)
    st.subheader("Answer:")
    st.write(answer)
    if validation.failures:
        st.warning("Validation Issues Found:")
        st.json(validation.failures)