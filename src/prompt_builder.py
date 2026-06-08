def build_context(retrieved_chunks):
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "Unknown")
        text = chunk.get("text") or chunk.get("chunk", "")

        context_parts.append(
            f"[Source {i} | File: {source} | Page {page}]\n{text}"
        )

    return "\n\n".join(context_parts)


def build_sources(retrieved_chunks):
    sources = []
    seen = set()

    for chunk in retrieved_chunks:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "Unknown")
        text = chunk.get("text") or chunk.get("chunk", "")

        key = (source, page)

        if key not in seen:
            sources.append({
                "source": source,
                "page": page,
                "preview": text[:300]
            })
            seen.add(key)

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
3. When you use information from a source, cite both the file name and page number.
   Use this format: [File: filename.pdf, Page X]
4. If the question asks for comparison, compare the papers only using evidence from the provided excerpts.
5. Keep the answer clear, concise, and grounded in the excerpts.

Question:
{question}

Paper excerpts:
{context}

Answer:
"""
    return prompt