"""
Test script to verify Gemini API connection via target_agent.query_gemini_agent().
"""
import os
import sys

# Ensure workspace root is on the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from biaslens.target_agent import query_gemini_agent


def main():
    test_prompt = "Say hello and give a one-sentence greeting."
    print(f"Sending prompt to Gemini API: '{test_prompt}'\n")

    try:
        response = query_gemini_agent(test_prompt)
        print("Gemini Agent Response:")
        print("----------------------")
        print(response.strip())
        print("----------------------")
        print("\nGemini API connection confirmed working end-to-end!")
    except Exception as e:
        print(f"Error querying Gemini API: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
