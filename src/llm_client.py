import os
from dotenv import load_dotenv
from anthropic import AnthropicFoundry


load_dotenv()


def generate_answer(prompt: str) -> str:
    client = AnthropicFoundry(
        api_key=os.getenv("ANTHROPIC_FOUNDRY_API_KEY"),
        resource=os.getenv("ANTHROPIC_FOUNDRY_RESOURCE"),
    )

    response = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.content[0].text