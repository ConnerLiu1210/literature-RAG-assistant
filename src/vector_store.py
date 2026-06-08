import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, collection_name="research_papers"):
        self.client = chromadb.Client()
        self.collection_name = collection_name

        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_chunks(self, chunks):
        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for i, item in enumerate(chunks):
            text = item["chunk"]
            page = item["page"]
            source = item.get("source", "Unknown")

            documents.append(text)
            embeddings.append(self.embedding_model.encode(text).tolist())
            metadatas.append({
                "page": page,
                "source": source,
                "chunk_id": i
            })
            ids.append(f"chunk_{i}")

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query, top_k=5):
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_chunks = []

        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]

            retrieved_chunks.append({
                "chunk": results["documents"][0][i],
                "text": results["documents"][0][i],
                "page": metadata.get("page", "Unknown"),
                "source": metadata.get("source", "Unknown"),
                "chunk_id": metadata.get("chunk_id"),
                "distance": results["distances"][0][i]
            })

        return retrieved_chunks


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
        item["text"] = item.get("chunk", "")
        item["page"] = item.get("page", "Unknown")
        item["source"] = item.get("source", "Unknown")
        item["chunk_id"] = int(idx)
        item["score"] = float(scores[idx])
        results.append(item)

    return results