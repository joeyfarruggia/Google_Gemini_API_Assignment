"""
Google Gemini API Assignment

This program uses Google's Gemini API to accept a question from the
user and generate an AI response.
"""

import os
import sys

from google import genai


MODEL_NAME = "gemini-3.5-flash"


def create_client() -> genai.Client:
    """
    Create and return a Gemini API client.

    The Google Gen AI SDK automatically reads GEMINI_API_KEY from the
    computer's environment variables.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found.\n"
            "Set it in PowerShell with:\n"
            '$env:GEMINI_API_KEY="your-api-key-here"'
        )

    return genai.Client(api_key=api_key)


def generate_response(client: genai.Client, prompt: str) -> str:
    """
    Send the user's prompt to Gemini and return the generated response.
    """
    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt,
    )

    return interaction.output_text


def main() -> None:
    """Run the Gemini command-line application."""
    print("=" * 60)
    print("Google Gemini AI Assistant")
    print("=" * 60)
    print("Enter a question for Gemini.")
    print("Type 'quit' to close the program.\n")

    try:
        client = create_client()
    except ValueError as error:
        print(f"Configuration error:\n{error}")
        sys.exit(1)

    while True:
        user_prompt = input("You: ").strip()

        if user_prompt.lower() in {"quit", "exit", "q"}:
            print("\nProgram closed.")
            break

        if not user_prompt:
            print("Please enter a question.\n")
            continue

        try:
            response = generate_response(client, user_prompt)

            print("\nGemini:")
            print(response)
            print()

        except Exception as error:
            print("\nThe Gemini API request was unsuccessful.")
            print(f"Error details: {error}\n")


if __name__ == "__main__":
    main()