from sentence_transformers import SentenceTransformer


def get_embedding_model():
    """
    Load the embedding model used to convert text chunks into vectors.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")