"""
Module for interfacing with the target LLM agent using the Gemini API.

Supports two backends via GEMINI_BACKEND env var:
  - "developer" (default): Gemini Developer API via an API key
    (GEMINI_API_KEY). This is what's been used so far - free tier or
    AI Studio prepay billing, depending on account state.
  - "vertex": Vertex AI's Gemini endpoint, billed through the same GCP
    project as BigQuery and eligible for Google Cloud trial credit
    (unlike AI Studio's Gemini API, which explicitly excludes trial
    credit - see https://ai.google.dev/gemini-api/docs/billing#prepay).
    Authenticates via Application Default Credentials - the same
    GOOGLE_APPLICATION_CREDENTIALS service account key already set up
    for BigQuery, so no separate credential file is needed.

Switching backends only requires setting env vars - query_gemini_agent()'s
signature and return value are unchanged either way, so run_audit.py and
app.py don't need to know which backend is active.
"""
import os
from google import genai

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_client = None  # lazily built and reused across calls in the same process


def _get_client() -> genai.Client:
    """
    Build (once) and return the genai.Client for whichever backend is
    configured. Cached at module level so repeated calls in a batch run
    (run_audit.py) or across Streamlit reruns don't rebuild the client -
    and, for Vertex, don't repeatedly re-resolve Application Default
    Credentials - on every single prompt.
    """
    global _client
    if _client is not None:
        return _client

    backend = os.environ.get("GEMINI_BACKEND", "developer").strip().lower()

    if backend == "vertex":
        project_id = os.environ.get("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError(
                "GEMINI_BACKEND=vertex requires GCP_PROJECT_ID to be set (same project "
                "as BigQuery). Authentication uses Application Default Credentials - "
                "locally, GOOGLE_APPLICATION_CREDENTIALS; on Streamlit Cloud, the "
                "gcp_service_account_json secret."
            )
        location = os.environ.get("GEMINI_VERTEX_LOCATION", "global")
        _client = genai.Client(vertexai=True, project=project_id, location=location)
    elif backend == "developer":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key)
    else:
        raise ValueError(f"Unknown GEMINI_BACKEND '{backend}' - expected 'developer' or 'vertex'")

    return _client


def query_gemini_agent(prompt: str, model: str = "gemini-flash-lite-latest") -> str:
    """
    Send a prompt to the Gemini target agent (Developer API or Vertex AI,
    per GEMINI_BACKEND) and return the response text.
    """
    # Target model selection can be configured via environment variable or parameter.
    # Vertex AI model IDs don't always match Developer API ones 1:1 (e.g. version
    # suffixes can differ) - if switching to vertex and this errors with a
    # not-found/invalid-model message, check the exact ID in Vertex AI Model Garden
    # for your project/region and set GEMINI_MODEL to that instead.
    model_name = os.environ.get("GEMINI_MODEL", model)

    client = _get_client()
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=dict(automatic_function_calling=dict(disable=True)),
    )
    return response.text

