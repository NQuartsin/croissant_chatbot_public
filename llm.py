# llm.py

import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("OPENROUTER_API_KEY")



def get_metadata_info_for_prompt(metadata):
    metadata_info = f"""
        name: {metadata.get("name", "No name provided")},
        author: {metadata.get("author", "No author provided")},
        year: {metadata.get("year", "No year provided")},
        title: {metadata.get("title", "No title provided")},
        description: {metadata.get("description", "No description provided")},
        license: {metadata.get("license", "No license provided")},
        url: {metadata.get("url", "No URL provided")},
        publisher: {metadata.get("publisher", "No publisher provided")},
        version: {metadata.get("version", "No version provided")},
        keywords: {metadata.get("keywords", "No keywords provided")},
        date_modified: {metadata.get("date_modified", "No date modified provided")},
        date_created: {metadata.get("date_created", "No date created provided")},
        date_published: {metadata.get("date_published", "No date published provided")},
        language: {metadata.get("language", "No language provided")},
        cite_as: {metadata.get("cite_as", "No citation provided")},
        task: {metadata.get("tasks", "No tasks provided")},
        modality: {metadata.get("modality", "No modality provided")}
    """
    return metadata_info

def suggest_field_value(metadata, informal_description, field):
    # for the fields: name, title, description, publisher, keywords, task, modality
    prompt = f"""
    The user is creating metadata for a dataset with the following information:
    {get_metadata_info_for_prompt(metadata)}
    Additional information: {informal_description}
    
    The field '{field}' is missing or insufficient.
    Please provide 1-3 reasonable suggestions for this field only.
    If there is no information available, prompt the user to provide more information.
    """

    return prompt

def suggest_ways_to_fill_field(metadata, informal_description, field):
    # for the fields: author, year, license, url, version, date_modified, date_created, date_published, language
    prompt = f"""
    The user is creating metadata for a dataset with the following information:
    {get_metadata_info_for_prompt(metadata)}
    Additional information: {informal_description}

    The field '{field}' is missing or insufficient.
    Please suggest at most 5 ways for the user to figure out how to fill this field.
    """

    return prompt


def suggest_citation(metadata):
    """Use OpenRouter's Llama 3.1 8B Instruct model to suggest a citation for a dataset."""
    prompt = f"""
    The user is creating metadata for a dataset with the following information:
    {get_metadata_info_for_prompt(metadata)}
    Please suggest a citation for this dataset in bibtex format. Also make suggestions for how best the user can create a citation for this dataset.
    """

    return prompt

def ask_user_for_informal_description():

    prompt = f"""
    The user is creating metadata for a dataset.
    Please ask the user probing questions to get an informal description of the dataset.
    """

    return create_llm_response(prompt)

def suggest_metadata(metadata, informal_description, field):

    if field == "cite_as":
        prompt = suggest_citation(metadata)
    elif field in ["name", "title", "description", "publisher", "keywords", "task", "modality"]:
        prompt = suggest_field_value(metadata, informal_description, field)
    else:
        prompt = suggest_ways_to_fill_field(metadata, informal_description, field)

    return create_llm_response(prompt)
    
def handle_conversation(metadata, user_prompt, history, informal_description):
    """Handle a conversation with the user to suggest metadata fields."""
    prompt =f"""
    The user is creating metadata for a dataset with the following information:
    {get_metadata_info_for_prompt(metadata)}
    Additional information: {informal_description}
    The last thing said by the user was: {user_prompt}
    The conversation history is: {history}

    Please suggest the next step in the conversation to help the user fill in the missing metadata fields.
    """

    return create_llm_response(prompt)

def create_llm_response(prompt):
    """Use OpenRouter's Llama 3.1 8B Instruct model to suggest missing metadata fields."""

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


