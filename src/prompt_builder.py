def build_rag_prompt(question, retrieved_chunks):
    """
    Build a RAG prompt from the user question and retrieved PDF chunks.

    question: str
    retrieved_chunks: list of {"chunk": text, "page": page_number, ...}
    """

    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        page = item.get("page", "unknown")
        text = item.get("chunk") or item.get("text", "")

        context_parts.append(
            f"[Source {i} | Page {page}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a research paper assistant.

Answer the user's question using ONLY the provided paper excerpts.
If the excerpts do not contain enough information, say that the provided context is insufficient.
Cite the page numbers when you use information from the excerpts.

Question:
{question}

Paper excerpts:
{context}

Answer:
""".strip()

    return prompt