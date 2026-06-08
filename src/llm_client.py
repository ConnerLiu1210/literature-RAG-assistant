import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


def generate_answer(prompt):
    api_key = os.getenv("AZURE_ANTHROPIC_API_KEY")
    base_url = os.getenv("AZURE_ANTHROPIC_BASE_URL")
    model = os.getenv("AZURE_ANTHROPIC_MODEL", "claude-sonnet-4-5")

    if not api_key:
        return "Missing AZURE_ANTHROPIC_API_KEY in Streamlit Secrets."

    if not base_url:
        return "Missing AZURE_ANTHROPIC_BASE_URL in Streamlit Secrets."

    client = Anthropic(
        api_key=api_key,
        base_url=base_url
    )

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text