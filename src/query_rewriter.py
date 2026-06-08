from src.llm_client import generate_answer


def rewrite_query(user_question, chat_history, uploaded_file_names):
    papers = "\n".join(
        [f"Paper {i + 1}: {name}" for i, name in enumerate(uploaded_file_names)]
    )

    history_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]]
    )

    prompt = f"""
You are a query rewriting assistant for a multi-paper RAG system.

Rewrite the user's question into a clear retrieval query.
Do not answer the question.

Uploaded papers:
{papers}

Conversation history:
{history_text}

User question:
{user_question}

Return only the rewritten query.
"""

    return generate_answer(prompt).strip()


def generate_multi_queries(user_question, rewritten_query, is_comparison=False):
    if is_comparison:
        return [
            rewritten_query,
            "main contribution and goal of each paper",
            "architecture and model design differences",
            "retrieval strategy differences",
            "training method and learning objective differences",
            "datasets experiments and evaluation results",
            "limitations and conclusions of each paper"
        ]

    return [
        rewritten_query,
        user_question
    ]