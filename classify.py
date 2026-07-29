"""
Classifies a member's free-text "why are you leaving" answer into one of
our fixed categories using the Gemini API.

SETUP:
1. Get a free API key at https://aistudio.google.com/app/apikey
2. Locally: create a file .streamlit/secrets.toml with:
       GEMINI_API_KEY = "your-key-here"
   On Streamlit Community Cloud: add the same key in
   App settings -> Secrets, in the deployed dashboard.
3. pip install google-generativeai
"""

import streamlit as st
import google.generativeai as genai

from actions import CATEGORIES

_MODEL_NAME = "gemini-flash-lite-latest"  # Google's alias for the current free-tier Flash-Lite model


def _get_model():
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in st.secrets. Add it to "
            ".streamlit/secrets.toml locally, or in your Streamlit Cloud "
            "app's Secrets settings."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(_MODEL_NAME)


def classify_reason(free_text: str) -> str:
    """
    Sends the member's free-text answer to Gemini and returns exactly one
    of the strings in actions.CATEGORIES. Falls back to "Other" if the
    model response doesn't cleanly match a category or the API call fails.
    """
    if not free_text or not free_text.strip():
        return "Other"

    categories_list = ", ".join(CATEGORIES)
    prompt = (
        "A gym member was asked why they are considering cancelling their "
        "membership. Classify their answer into EXACTLY ONE of the "
        f"following categories: {categories_list}.\n\n"
        f'Member answer: "{free_text.strip()}"\n\n'
        "Respond with ONLY the category name, exactly as written above, "
        "and nothing else."
    )

    try:
        model = _get_model()
        response = model.generate_content(prompt)
        raw = (response.text or "").strip()

        # Match the model's output back to one of our known categories,
        # in case it adds punctuation/extra words despite instructions.
        for category in CATEGORIES:
            if category.lower() in raw.lower():
                return category

        return "Other"

    except Exception as e:
        st.warning(f"Classification service unavailable, defaulting to "
                    f"'Other'. ({e})")
        return "Other"
