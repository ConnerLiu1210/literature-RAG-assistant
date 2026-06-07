import os
import tempfile

import streamlit as st

from src.pdf_loader import load_pdf
from src.text_splitter import split_pages_into_chunks
from src.vector_store import VectorStore
from src.prompt_builder import build_rag_prompt
from src.llm_client import generate_answer


st.set_page_config(
    page_title="Literature RAG Assistant",
    page_icon="📄",
    layout="wide"
)


st.title("📄 Literature RAG Assistant")
st.write("Upload a research paper PDF and ask questions about it.")


with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=5
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=100,
        max_value=800,
        value=250,
        step=50
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10
    )


uploaded_file = st.file_uploader(
    "Upload a PDF paper",
    type=["pdf"]
)


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    try:
        pages = load_pdf(pdf_path)

        chunks = split_pages_into_chunks(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        vector_store = VectorStore()
        vector_store.add_chunks(chunks)

        st.success(f"Loaded {len(pages)} pages and created {len(chunks)} chunks.")

        question = st.text_input(
            "Ask a question",
            value="What dataset or data did this paper use?"
        )

        if st.button("Ask"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                results = vector_store.search(question, top_k=top_k)

                st.subheader("Retrieved Sources")
                st.caption("Lower distance means the retrieved chunk is more relevant.")

                for i, result in enumerate(results, start=1):
                    page = result["page"]
                    distance = result["distance"]
                    text = result["text"]

                    with st.expander(
                        f"Source {i} | Page {page} | Distance: {distance:.4f}"
                    ):
                        st.write(text)

                prompt = build_rag_prompt(question, results)

                answer = generate_answer(prompt)

                st.subheader("Answer")
                st.write(answer)

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

else:
    st.info("Please upload a PDF file to start.")