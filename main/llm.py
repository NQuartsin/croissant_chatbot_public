# llm.py

# necessary imports
from math import e
import requests
from dotenv import load_dotenv
import os
from typing import Dict

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("OPENROUTER_API_KEY")  # Get API key from environment variable
if api_key is None:
    raise EnvironmentError("OPENROUTER_API_KEY is not set in the environment variables. Please set it in the .env file.")

"""
This module contains functions to interact with the OpenRouter API for generating metadata suggestions.
"""

def get_metadata_info_for_prompt(metadata: Dict[str, str]) -> str:
    """
    Generate a formatted string containing metadata information.

    Args:
        metadata: A Dictionary where keys are metadata attributes and values are their respective values.

    Returns:
        metadata_info: A formatted string containing metadata information.
    """
    metadata_info = f"""
        name: {metadata.get("name", "No name provided")},
        creators: {metadata.get("creators", "No creators provided")},
        description: {metadata.get("description", "No description provided")},
        license: {metadata.get("license", "No license provided")},
        url: {metadata.get("url", "No URL provided")},
        publisher: {metadata.get("publisher", "No publisher provided")},
        version: {metadata.get("version", "No version provided")},
        keywords: {metadata.get("keywords", "No keywords provided")},
        date_modified: {metadata.get("date_modified", "No date modified provided")},
        date_created: {metadata.get("date_created", "No date created provided")},
        date_published: {metadata.get("date_published", "No date published provided")},
        in_language: {metadata.get("in_language", "No language provided")},
        cite_as: {metadata.get("cite_as", "No citation provided")},
        task: {metadata.get("tasks", "No tasks provided")},
        modality: {metadata.get("modality", "No modality provided")}
    """
    return metadata_info 

def create_prompt_to_suggest_attribute_value(metadata: Dict[str, str], informal_description: str, attribute: str) -> str:
    """
    Suggest reasonable values for a specific metadata attribute.

    Args:
        metadata: A Dictionary of metadata attributes and their values.
        informal_description: Additional informal description of the dataset.
        attribute: The metadata attribute for which suggestions are needed.

    Returns:
        prompt: A prompt string to generate suggestions for the specified attribute.
    """
    prompt = f"""
    The user is creating metadata for a dataset with the following information:
    {get_metadata_info_for_prompt(metadata)}
    Additional information for the dataset is the following informal description (ignore if empty): {informal_description}
    
    The attribute '{attribute}' is missing or insufficient.
    Please provide 1-3 reasonable suggestions for this attribute only.
    """

    return prompt

def create_prompt_to_suggest_description(metadata: Dict[str, str], informal_description: str) -> str:
    """
    Suggest diverse descriptions for the dataset.

    Args:
        metadata: A Dictionary of metadata attributes and their values.
        informal_description: Additional informal description of the dataset.

    Returns:
        prompt: A prompt string to generate diverse descriptions for the dataset.
    """
    prompt = f"""
    The user is creating metadata for a dataset with the following information:
    {get_metadata_info_for_prompt(metadata)}
    Additional information for the dataset is the following informal description (ignore if empty): {informal_description}

    The attribute 'description' is missing or insufficient.
    Please provide 1-3 diverse, non-repetitive descriptions that are at least 2 sentences long.
    """

    return prompt

def create_prompt_to_suggest_ways_to_fill_attribute(metadata: Dict[str, str], informal_description: str, attribute: str) -> str:
    """
    Suggest ways to fill a specific metadata attribute.

    Args:
        metadata: A Dictionary of metadata attributes and their values.
        informal_description: Additional informal description of the dataset.
        attribute: The metadata attribute for which suggestions are needed.

    Returns:
        prompt: A prompt string to generate suggestions for filling the specified attribute.
    """
    prompt = f"""
    The user is creating metadata for a dataset with the following information:
    {get_metadata_info_for_prompt(metadata)}
    Additional information for the dataset is the following informal description (ignore if empty): {informal_description}

    The attribute '{attribute}' is missing or insufficient.
    Please suggest at most 5 ways for the user to figure out how to fill this attribute.
    """

    return prompt


def create_prompt_to_suggest_citation(metadata: Dict[str, str]) -> str:
    """
    Suggest a citation for the dataset in bibtex format.

    Args:
        metadata: A Dictionary of metadata attributes and their values.

    Returns:
        prompt: A prompt string to generate a citation for the dataset.
    """
    prompt = f"""
    The user is creating metadata for a dataset with the following information:
    {get_metadata_info_for_prompt(metadata)}
    Please suggest a citation for this dataset in bibtex format. Also make suggestions for how best the user can create a citation for this dataset.
    """

    return prompt

def ask_user_for_informal_description() -> str:
    """
    Generate probing questions to gather an informal description of the dataset.

    Returns:
        A string containing probing questions for the user.
    """
    try:
        prompt = f"""
        The user is creating metadata for a dataset.
        Please ask the user probing questions to get an informal description of the dataset.
        Ask 1-5 questions.
        """
        return str(create_llm_response(prompt))
    except Exception as e:
        raise Exception(f"An error occurred while trying to use the LLM model.\n {e}")


def suggest_metadata(metadata: Dict[str, str], informal_description: str, attribute: str) -> str:
    """
    Suggest metadata for a specific attribute based on existing metadata and informal description.

    Args:
        metadata: A Dictionary of metadata attributes and their values.
        informal_description: Additional informal description of the dataset.
        attribute: The metadata attribute for which suggestions are needed.

    Returns:
        A string containing suggestions for the specified attribute.
    """
    try:
        if attribute == "cite_as":
            prompt = create_prompt_to_suggest_citation(metadata)
        elif attribute in ["name", "publisher", "keywords", "task", "modality", "license", "in_in_language"]:
            prompt = create_prompt_to_suggest_attribute_value(metadata, informal_description, attribute)
        elif attribute == "description":
            prompt = create_prompt_to_suggest_description(metadata, informal_description)
        else:
            prompt = create_prompt_to_suggest_ways_to_fill_attribute(metadata, informal_description, attribute)

        return str(create_llm_response(prompt))
    except Exception as e:
        raise Exception(f"An error occurred while trying to use the LLM model.\n {e}")


def create_llm_response(prompt: str) -> str:
    """
    Use OpenRouter's Llama 3.1 8B Instruct model to generate a response for the given prompt.
    Source: https://openrouter.ai/mistralai/mistral-7b-instruct/api 

    Args:
        prompt: The input prompt string for the LLM model.

    Returns:
        The response generated by the LLM model.
    """
    model_propmt = "You are helping a user create metadata for a dataset." + prompt

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": model_propmt}],
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"An error occured while trying to use the LLM model.\n {response.status_code}: {response.text}")
    except Exception as e:
        raise Exception(f"An error occured while trying to use the LLM model.\n {e}")

