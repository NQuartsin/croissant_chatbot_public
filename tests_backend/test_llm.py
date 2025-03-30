# test_llm.py

#necessary imports
import pytest
from unittest.mock import patch
from main.llm import (
    get_metadata_info_for_prompt,
    create_prompt_to_suggest_attribute_value,
    create_prompt_to_suggest_description,
    create_prompt_to_suggest_ways_to_fill_attribute,
    create_prompt_to_suggest_citation,
    ask_user_for_informal_description,
    suggest_metadata,
    create_llm_response,
)

@pytest.fixture
def sample_metadata():
    """Fixture to provide sample metadata."""
    return {
        "name": "Sample Dataset",
        "author": "John Doe",
    }

def test_get_metadata_info_for_prompt(sample_metadata):
    """Test the get_metadata_info_for_prompt function."""
    metadata_info = get_metadata_info_for_prompt(sample_metadata)
    assert "name: Sample Dataset" in metadata_info
    assert "author: John Doe" in metadata_info


def test_create_prompt_to_suggest_attribute_value(sample_metadata):
    """Test the create_prompt_to_suggest_attribute_value function."""
    prompt = create_prompt_to_suggest_attribute_value(sample_metadata, "This is a test dataset.", "keywords")
    assert "The user is creating metadata for a dataset with the following information:" in prompt
    assert "name: Sample Dataset" in prompt
    assert "author: John Doe" in prompt
    assert "This is a test dataset." in prompt
    assert "The attribute 'keywords' is missing or insufficient." in prompt
    assert "Please provide 1-3 reasonable suggestions for this attribute only." in prompt

def test_create_prompt_to_suggest_description(sample_metadata):
    """Test the create_prompt_to_suggest_description function."""
    prompt = create_prompt_to_suggest_description(sample_metadata, "This is a test dataset.")
    assert "The attribute 'description' is missing or insufficient." in prompt
    assert "name: Sample Dataset" in prompt
    assert "author: John Doe" in prompt
    assert "This is a test dataset." in prompt
    assert "Please provide 1-3 diverse, non-repetitive descriptions that are at least 2 sentences long." in prompt

def test_create_prompt_to_suggest_ways_to_fill_attribute(sample_metadata):
    """Test the create_prompt_to_suggest_ways_to_fill_attribute function."""
    prompt = create_prompt_to_suggest_ways_to_fill_attribute(sample_metadata, "This is a test dataset.", "date_created")
    assert "The attribute 'date_created' is missing or insufficient." in prompt
    assert "name: Sample Dataset" in prompt
    assert "author: John Doe" in prompt
    assert "This is a test dataset." in prompt
    assert "Please suggest at most 5 ways for the user to figure out how to fill this attribute." in prompt


def test_create_prompt_to_suggest_citation(sample_metadata):
    """Test the create_prompt_to_suggest_citation function."""
    prompt = create_prompt_to_suggest_citation(sample_metadata)
    assert "name: Sample Dataset" in prompt
    assert "author: John Doe" in prompt
    assert "Please suggest a citation for this dataset in bibtex format." in prompt

@patch("main.llm.create_llm_response")
def test_ask_user_for_informal_description(mock_create_llm_response):
    """Test the ask_user_for_informal_description function."""
    mock_create_llm_response.return_value = "What is the purpose of the dataset?"
    response = ask_user_for_informal_description()
    assert response == "What is the purpose of the dataset?"
    mock_create_llm_response.assert_called_once()

@patch("main.llm.create_llm_response")
def test_ask_user_for_informal_description_exception(mock_create_llm_response):
    """Test ask_user_for_informal_description when create_llm_response raises an exception."""
    mock_create_llm_response.side_effect = Exception("Mocked exception")
    with pytest.raises(Exception, match="An error occurred while trying to use the LLM model"):
        ask_user_for_informal_description()
    mock_create_llm_response.assert_called_once()

@patch("main.llm.create_llm_response")
def test_suggest_metadata_cite_as(mock_create_llm_response, sample_metadata):
    """Test the suggest_metadata function for cite_as."""
    mock_create_llm_response.return_value = "Suggested metadata response."
    response = suggest_metadata(sample_metadata, "This is a test dataset.", "cite_as")
    assert response == "Suggested metadata response."
    mock_create_llm_response.assert_called_once()

@patch("main.llm.create_llm_response")
def test_suggest_metadata_suggest_attribute_value(mock_create_llm_response, sample_metadata):
    """Test the suggest_metadata function for suggest_attribute_value."""
    mock_create_llm_response.return_value = "Suggested metadata response."
    response = suggest_metadata(sample_metadata, "This is a test dataset.", "keywords")
    assert response == "Suggested metadata response."
    mock_create_llm_response.assert_called_once()

@patch("main.llm.create_llm_response")
def test_suggest_metadata_suggest_description(mock_create_llm_response, sample_metadata):
    """Test the suggest_metadata function for suggest_description."""
    mock_create_llm_response.return_value = "Suggested metadata response."
    response = suggest_metadata(sample_metadata, "This is a test dataset.", "description")
    assert response == "Suggested metadata response."
    mock_create_llm_response.assert_called_once()

@patch("main.llm.create_llm_response")
def test_suggest_metadata_suggest_ways_to_fill_attribute(mock_create_llm_response, sample_metadata):
    """Test the suggest_metadata function for suggest_ways_to_fill_attribute."""
    mock_create_llm_response.return_value = "Suggested metadata response."
    response = suggest_metadata(sample_metadata, "This is a test dataset.", "date_created")
    assert response == "Suggested metadata response."
    mock_create_llm_response.assert_called_once()

@patch("main.llm.create_llm_response")
def test_suggest_metadata_exception(mock_create_llm_response, sample_metadata):
    """Test suggest_metadata when create_llm_response raises an exception."""
    mock_create_llm_response.side_effect = Exception("Mocked exception")
    with pytest.raises(Exception, match="An error occurred while trying to use the LLM model"):
        suggest_metadata(sample_metadata, "This is a test dataset.", "keywords")
    mock_create_llm_response.assert_called_once()

@patch("main.llm.requests.post")
def test_create_llm_response(mock_post):
    """Test the create_llm_response function."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "This is a test response."}}]
    }

    prompt = "Test prompt"
    response = create_llm_response(prompt)
    assert response == "This is a test response."
    mock_post.assert_called_once()

@patch("main.llm.requests.post")
def test_create_llm_response_error(mock_post):
    """Test create_llm_response when the API returns an error."""
    mock_post.return_value.status_code = 400
    mock_post.return_value.text = "Bad Request"

    prompt = "Test prompt"
    with pytest.raises(Exception, match="An error occured while trying to use the LLM model"):
        create_llm_response(prompt)
        