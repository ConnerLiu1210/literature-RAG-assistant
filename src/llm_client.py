import os
from anthropic import Anthropic

from dotenv import load_dotenv

load_dotenv()


def generate_answer(prompt):
    client = Anthropic(
        api_key=os.getenv("AZURE_ANTHROPIC_API_KEY"),
        base_url=os.getenv("AZURE_ANTHROPIC_BASE_URL")
    )

    response = client.messages.create(
        model=os.getenv("AZURE_ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text