"""
Module for interfacing with the target LLM agent using Gemini API.
"""
import os
from google import genai

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def query_gemini_agent(prompt: str, model: str = "gemini-flash-lite-latest") -> str:
    """
    Send a prompt to the Gemini API target agent and return the response.
    """
    # API keys must be kept in environment variables, never hardcoded.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Target model selection can be configured via environment variable or parameter.
    model_name = os.environ.get("GEMINI_MODEL", model)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=dict(automatic_function_calling=dict(disable=True)),
    )
    return response.text

