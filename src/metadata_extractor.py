from src.llm_client import generate_answer


def extract_paper_metadata(first_page_text, source_name):
    prompt = f"""
You are a research paper metadata extraction assistant.

Extract the paper metadata from the first page text below.

Return the result in this exact format:

Title: ...
Authors: ...
Venue/Year: ...

If something is not found, write "Unknown".

Source file:
{source_name}

First page text:
{first_page_text[:4000]}
"""

    metadata_text = generate_answer(prompt).strip()

    return {
        "source": source_name,
        "metadata_text": metadata_text
    }


def answer_metadata_question(question, paper_metadata):
    metadata_context = "\n\n".join(
        [
            f"{item['source']}\n{item['metadata_text']}"
            for item in paper_metadata
        ]
    )

    prompt = f"""
You are a research paper assistant.

Answer the user's question using ONLY the paper metadata below.

Paper metadata:
{metadata_context}

User question:
{question}

Answer clearly and concisely.
"""

    return generate_answer(prompt).strip()