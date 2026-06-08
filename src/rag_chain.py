def build_context(retrieved_chunks):
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        page = chunk.get("page", "Unknown")
        text = chunk.get("text") or chunk.get("chunk", "")

        context_parts.append(
            f"[Source {i} | Page {page}]\n{text}"
        )

    return "\n\n".join(context_parts)


def build_sources(retrieved_chunks):
    sources = []

    seen_pages = set()

    for chunk in retrieved_chunks:
        page = chunk.get("page", "Unknown")
        text = chunk.get("text") or chunk.get("chunk", "")

        if page not in seen_pages:
            sources.append({
                "page": page,
                "preview": text[:300]
            })
            seen_pages.add(page)

    return sources


def build_rag_prompt(question, retrieved_chunks):
    context = build_context(retrieved_chunks)

    prompt = f"""
You are a research paper assistant.

Answer the user's question using ONLY the provided paper excerpts.

Rules:
1. Do not use outside knowledge.
2. If the provided excerpts do not contain enough information, say:
   "The provided context is insufficient to answer this question."
3. When you use information from a source, cite the page number in this format: [Page X].
4. Keep the answer clear, concise, and grounded in the excerpts.

Question:
{question}

Paper excerpts:
{context}

Answer:
"""
    return prompt


def build_rag_response(answer, retrieved_chunks):
    return {
        "answer": answer,
        "sources": build_sources(retrieved_chunks)
    }