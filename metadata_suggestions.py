# metadata_suggestions.py

import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("OPENROUTER_API_KEY")


def suggest_metadata(field_name, dataset_name, description):
    """Use OpenRouter's Llama 3.1 8B Instruct model to suggest missing metadata fields."""
    prompt = f"""
    The user is creating metadata for a dataset named '{dataset_name}'.
    Description: {description}

    The field '{field_name}' is missing or insufficient.
    Please suggest a reasonable value for this field.
    """

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
