def split_pages_into_chunks(pages, chunk_size=250, chunk_overlap=50):
    """
    Split PDF pages into word-based chunks.

    pages: [{"page": page_number, "text": text}, ...]
    return: [{"page": page_number, "chunk": chunk_text}, ...]
    """
    chunks = []

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        words = text.split()

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append({
                    "page": page_number,
                    "chunk": chunk_text
                })

            start += chunk_size - chunk_overlap

    return chunks


# Alias for Streamlit app import
split_pages = split_pages_into_chunks