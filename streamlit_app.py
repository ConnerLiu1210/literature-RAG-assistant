import os
import tempfile

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.pdf_loader import load_pdf
from src.text_splitter import split_pages_into_chunks
from src.vector_store import build_vector_store, search
from src.prompt_builder import build_rag_prompt, build_sources
from src.llm_client import generate_answer
from src.query_rewriter import rewrite_query, generate_multi_queries
from src.metadata_extractor import extract_paper_metadata, answer_metadata_question


st.set_page_config(
    page_title="Literature RAG Assistant",
    page_icon="📄",
    layout="wide"
)


def clear_chat_state():
    st.session_state["messages"] = []
    st.session_state.pop("last_sources", None)
    st.session_state.pop("last_results", None)


def is_comparison_question(question):
    keywords = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "different",
        "between these two",
        "between the two",
        "these two",
        "two papers",
        "两篇",
        "两个文章",
        "这两个",
        "区别",
        "差异",
        "比较",
        "不同"
    ]

    question_lower = question.lower()
    return any(keyword.lower() in question_lower for keyword in keywords)


def is_metadata_question(question):
    keywords = [
        "author",
        "authors",
        "title",
        "venue",
        "year",
        "conference",
        "谁写的",
        "作者",
        "标题",
        "题目",
        "会议",
        "年份",
        "发表",
        "哪一年"
    ]

    question_lower = question.lower()
    return any(keyword.lower() in question_lower for keyword in keywords)


def fix_result_sources(results, all_chunks):
    fixed_results = []

    for result in results:
        chunk_id = result.get("chunk_id")

        if chunk_id is not None and 0 <= chunk_id < len(all_chunks):
            original_chunk = all_chunks[chunk_id]

            result["source"] = original_chunk.get("source", "Unknown")
            result["page"] = original_chunk.get("page", result.get("page", "Unknown"))
            result["text"] = original_chunk.get("chunk", result.get("text", ""))
            result["chunk"] = original_chunk.get("chunk", result.get("chunk", ""))

        fixed_results.append(result)

    return fixed_results


def multi_query_search(vector_store, queries, all_chunks, top_k):
    all_results = []
    seen_chunk_ids = set()

    for query in queries:
        results = search(vector_store, query, top_k=top_k)
        fixed_results = fix_result_sources(results, all_chunks)

        for result in fixed_results:
            chunk_id = result.get("chunk_id")

            if chunk_id not in seen_chunk_ids:
                all_results.append(result)
                seen_chunk_ids.add(chunk_id)

    return all_results


st.title("📄 Literature RAG Assistant")
st.write("Upload one or more research paper PDFs and chat with them using page-level citations.")


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

    show_debug = st.checkbox(
        "Show retrieved chunks",
        value=False
    )

    show_rewritten_query = st.checkbox(
        "Show rewritten query",
        value=True
    )

    show_metadata = st.checkbox(
        "Show paper metadata",
        value=False
    )

    if st.button("Clear chat"):
        clear_chat_state()


uploaded_files = st.file_uploader(
    "Upload PDF papers",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:
    file_signature = tuple(file.name for file in uploaded_files)

    if st.session_state.get("file_signature") != file_signature:
        st.session_state["file_signature"] = file_signature
        clear_chat_state()
        st.session_state.pop("paper_metadata", None)

    all_chunks = []
    temp_paths = []
    paper_metadata = []

    try:
        for file_index, uploaded_file in enumerate(uploaded_files, start=1):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name
                temp_paths.append(pdf_path)

            pages = load_pdf(pdf_path)

            source_name = f"Paper {file_index}: {uploaded_file.name}"

            if pages:
                metadata = extract_paper_metadata(
                    first_page_text=pages[0]["text"],
                    source_name=source_name
                )
                paper_metadata.append(metadata)

            chunks = split_pages_into_chunks(
                pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_name=source_name
            )

            all_chunks.extend(chunks)

        st.session_state["paper_metadata"] = paper_metadata

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        vector_store = build_vector_store(all_chunks, embedding_model)

        st.success(
            f"Loaded {len(uploaded_files)} PDF(s) and created {len(all_chunks)} chunks."
        )

        if show_metadata:
            st.subheader("Paper Metadata")
            for item in st.session_state["paper_metadata"]:
                with st.expander(item["source"]):
                    st.write(item["metadata_text"])

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_question = st.chat_input("Ask a question about the uploaded papers")

        if user_question:
            st.session_state["messages"].append({
                "role": "user",
                "content": user_question
            })

            with st.chat_message("user"):
                st.write(user_question)

            comparison_mode = is_comparison_question(user_question)
            metadata_mode = is_metadata_question(user_question)

            if comparison_mode and len(uploaded_files) < 2:
                answer = (
                    "You uploaded only one paper, so I cannot compare two papers yet. "
                    "Please upload a second PDF if you want a comparison."
                )

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer
                })

                with st.chat_message("assistant"):
                    st.write(answer)

            elif metadata_mode:
                answer = answer_metadata_question(
                    question=user_question,
                    paper_metadata=st.session_state["paper_metadata"]
                )

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer
                })

                st.session_state.pop("last_sources", None)
                st.session_state.pop("last_results", None)

                with st.chat_message("assistant"):
                    st.write(answer)

            else:
                uploaded_file_names = [file.name for file in uploaded_files]

                rewritten_query = rewrite_query(
                    user_question=user_question,
                    chat_history=st.session_state["messages"],
                    uploaded_file_names=uploaded_file_names
                )

                search_queries = generate_multi_queries(
                    user_question=user_question,
                    rewritten_query=rewritten_query,
                    is_comparison=comparison_mode
                )

                if show_rewritten_query:
                    st.caption(f"Rewritten query: {rewritten_query}")

                retrieval_k = max(top_k, 8) if comparison_mode else top_k

                results = multi_query_search(
                    vector_store=vector_store,
                    queries=search_queries,
                    all_chunks=all_chunks,
                    top_k=retrieval_k
                )

                if comparison_mode:
                    results = results[:25]
                else:
                    results = results[:top_k]

                recent_history = "\n".join(
                    [
                        f"{msg['role']}: {msg['content']}"
                        for msg in st.session_state["messages"][-4:]
                    ]
                )

                if comparison_mode:
                    chat_question = f"""
Conversation history:
{recent_history}

Current question:
{user_question}

Rewritten retrieval query:
{rewritten_query}

The user is asking for a comparison between papers.

Answer using ONLY the retrieved excerpts.

When possible, structure the answer as:

| Category | Paper 1 | Paper 2 |
|---|---|---|
| Goal / Main idea | ... | ... |
| Architecture | ... | ... |
| Retrieval strategy | ... | ... |
| Training method | ... | ... |
| Datasets / Experiments | ... | ... |
| Main results | ... | ... |

If some categories are not supported by the excerpts, write "Not found in retrieved excerpts."

Always cite file name and page number.
"""
                else:
                    chat_question = f"""
Conversation history:
{recent_history}

Current question:
{user_question}

Rewritten retrieval query:
{rewritten_query}

Use the conversation history only to understand follow-up references.
Answer the current question using only the retrieved paper excerpts.
"""

                prompt = build_rag_prompt(chat_question, results)
                answer = generate_answer(prompt)
                sources = build_sources(results)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer
                })

                st.session_state["last_sources"] = sources
                st.session_state["last_results"] = results

                with st.chat_message("assistant"):
                    st.write(answer)

        if "last_sources" in st.session_state:
            st.subheader("Sources")
            st.caption("These are the retrieved paper excerpts used to generate the latest answer.")

            for i, source in enumerate(st.session_state["last_sources"], start=1):
                source_name = source.get("source", "Unknown")
                page = source.get("page", "Unknown")
                preview = source.get("preview", "")

                with st.expander(f"Source {i} | {source_name} | Page {page}"):
                    st.write(preview)

        if show_debug and "last_results" in st.session_state:
            st.subheader("Retrieved Chunks")
            st.caption("Higher score means the retrieved chunk is more relevant.")

            for i, result in enumerate(st.session_state["last_results"], start=1):
                source_name = result.get("source", "Unknown")
                page = result.get("page", "Unknown")
                score = result.get("score", 0)
                text = result.get("text", "")

                with st.expander(
                    f"Chunk {i} | {source_name} | Page {page} | Score: {score:.4f}"
                ):
                    st.write(text)

    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

else:
    clear_chat_state()
    st.session_state.pop("paper_metadata", None)
    st.info("Please upload one or more PDF files to start.")