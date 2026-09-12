"""
Standalone diagnostic: confirms whether the genai.Client is really hitting
Vertex AI or still falling back to the Developer API, before trusting any
batch run to it. Run this directly - not through target_agent.py - so we
can inspect the client's internal state.
"""
import os
from google import genai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

project_id = os.environ.get("GCP_PROJECT_ID")
location = os.environ.get("GEMINI_VERTEX_LOCATION", "global")

print(f"GEMINI_BACKEND env var: {os.environ.get('GEMINI_BACKEND')!r}")
print(f"GEMINI_API_KEY is set:  {bool(os.environ.get('GEMINI_API_KEY'))}")
print(f"GCP_PROJECT_ID:         {project_id!r}")
print(f"GEMINI_VERTEX_LOCATION: {location!r}")
print()

client = genai.Client(vertexai=True, project=project_id, location=location)

# Inspect the client's own belief about its mode - this is the ground truth,
# not an assumption.
try:
    print(f"client.vertexai property: {client.vertexai}")
except AttributeError:
    pass
try:
    print(f"client._api_client.vertexai: {client._api_client.vertexai}")
except AttributeError:
    pass
try:
    print(f"client._api_client.project: {client._api_client.project}")
    print(f"client._api_client.location: {client._api_client.location}")
except AttributeError:
    pass

print("\nAttempting one generate_content call...\n")
model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
print(f"Using model: {model_name}")

response = client.models.generate_content(
    model=model_name,
    contents="Say OK if you can read this.",
    config=dict(automatic_function_calling=dict(disable=True)),
)
print("\nSUCCESS:")
print(response.text)
