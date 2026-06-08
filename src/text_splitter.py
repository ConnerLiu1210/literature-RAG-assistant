def split_pages_into_chunks(
    pages,
    chunk_size=250,
    chunk_overlap=50,
    source_name="Unknown"
):
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
                    "source": source_name,
                    "page": page_number,
                    "chunk": chunk_text
                })

            start += chunk_size - chunk_overlap

    return chunks