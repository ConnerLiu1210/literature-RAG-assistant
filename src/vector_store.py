import numpy as np


def build_vector_store(chunks, embedding_model):
    texts = [item["chunk"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "embedding_model": embedding_model
    }


def search(vector_store, query, top_k=5):
    embedding_model = vector_store["embedding_model"]
    chunks = vector_store["chunks"]
    embeddings = vector_store["embeddings"]

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        item = chunks[idx].copy()

        results.append({
            "chunk": item.get("chunk", ""),
            "text": item.get("chunk", ""),
            "page": item.get("page", "Unknown"),
            "source": item.get("source", "Unknown"),
            "chunk_id": int(idx),
            "score": float(scores[idx])
        })

    return results