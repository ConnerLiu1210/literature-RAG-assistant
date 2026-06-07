def build_context(retrieved_chunks):
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        page = chunk["page"]
        text = chunk["text"]

        context_parts.append(
            f"[Source {i} | Page {page}]\n{text}"
        )

    return "\n\n".join(context_parts)


def build_rag_prompt(question, retrieved_chunks):
    context = build_context(retrieved_chunks)

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
"""
    return prompt