from src.pdf_loader import load_pdf
from src.text_splitter import split_pages_into_chunks
from src.vector_store import VectorStore
from src.rag_chain import build_rag_prompt
from src.llm_client import generate_answer

def main():
    pdf_path = "data/sample_paper.pdf"

    pages = load_pdf(pdf_path)
    print("Total pages:", len(pages))

    chunks = split_pages_into_chunks(pages)
    print("Total chunks:", len(chunks))

    vector_store = VectorStore()
    vector_store.add_chunks(chunks)

    question = "What dataset or data did this paper use?"
    results = vector_store.search(question, top_k=5)

    print("\nQuestion:")
    print(question)

    print("\nRetrieved Sources:")
    for i, result in enumerate(results, start=1):
        print("\n" + "=" * 50)
        print(f"Source {i}")
        print("Page:", result["page"])
        print("Distance:", result["distance"])
        print(result["text"][:800])

    prompt = build_rag_prompt(question, results)

    print("\n" + "#" * 80)
    print("ANSWER")
    print("#" * 80)

    answer = generate_answer(prompt)
    print(answer)


if __name__ == "__main__":
    main()