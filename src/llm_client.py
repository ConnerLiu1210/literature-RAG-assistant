import os
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


def get_secret(name, default=None):
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def generate_answer(prompt):
    api_key = get_secret("AZURE_ANTHROPIC_API_KEY")
    base_url = get_secret("AZURE_ANTHROPIC_BASE_URL")
    model = get_secret("AZURE_ANTHROPIC_MODEL", "claude-sonnet-4-5")

    if not api_key:
        return "Missing AZURE_ANTHROPIC_API_KEY. Please check Streamlit Secrets."

    if not base_url:
        return "Missing AZURE_ANTHROPIC_BASE_URL. Please check Streamlit Secrets."

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