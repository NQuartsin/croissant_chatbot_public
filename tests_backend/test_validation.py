# test_validation.py

# necessary imports
from unittest.mock import patch, mock_open
import pytest
from main.validation import MetadataValidator

"""
Test cases for the MetadataValidator class.
"""

@pytest.fixture
def validator():
    """Fixture to create a MetadataValidator instance."""
    return MetadataValidator()

def test_validate_url(validator):
    """Test the validate_url method."""
    valid, message = validator.validate_url("https://example.com")
    assert valid is True
    assert message == "URL is valid."
    valid, message = validator.validate_url("http://example.com")
    assert valid is True
    valid, message = validator.validate_url("invalid-url")
    assert valid is False
    assert message == "Invalid URL format."
    valid, message = validator.validate_url("ftp://example.com")
    assert valid is False
    assert message == "Invalid URL format."

    with patch("re.compile", side_effect=Exception("Mocked exception")):
        valid, message = validator.validate_url("https://example.com")
        assert valid is False
        assert message == "Error validating URL: Mocked exception"

@patch("builtins.open", new_callable=mock_open, read_data='{"licenses": [{"licenseId": "BSD Zero Clause License"}, {"licenseId": "MIT"}]}')
def test_validate_license(mock_file, validator):
    """Test the validate_license method."""
    valid, message = validator.validate_license("BSD Zero Clause License")
    assert valid is True
    assert message == "License is valid."
    valid, message = validator.validate_license("mit")
    assert valid is True
    assert message == "License is valid."
    valid, message = validator.validate_license("Invalid License")
    assert valid is False
    assert message == "Invalid License: liecence must be from the SPDX License List"

    with patch("builtins.open", side_effect=FileNotFoundError):
        valid, message = validator.validate_license("BSD Zero Clause License")
        assert valid is False
        assert "Error validating license:" in message

def test_check_non_empty_string(validator):
    """Test the check_non_empty_string method."""
    attribute_name = "test_attribute"
    valid, message = validator.check_non_empty_string("Hello", attribute_name)
    assert valid is True
    assert message == "Value is a non-empty string."
    valid, message = validator.check_non_empty_string("", attribute_name)
    assert valid is False
    assert message == "test_attribute must be a non-empty string."

def test_validate_keywords(validator):
    """Test the validate_keywords method."""
    valid, message = validator.validate_keywords("keyword1")
    assert valid is True
    assert message == "Keywords are valid."
    valid, message = validator.validate_keywords("keyword1, keyword2")
    assert valid is True
    assert message == "Keywords are valid."
    valid, message = validator.validate_keywords([])
    assert valid is False
    assert message == "Keywords must be a comma-separated string."
    valid, message = validator.validate_keywords(["keyword1", "keyword2"])
    assert valid is False
    assert message == "Keywords must be a comma-separated string."

    # Mock the all function to raise an exception
    with patch("builtins.all", side_effect=Exception("Mocked exception")):
        valid, message = validator.validate_keywords("keyword1, keyword2")
        assert valid is False
        assert "Error validating keywords" in message
        assert "Mocked exception" in message

def test_validate_date(validator):
    """Test the validate_date method."""
    attribute_name = "test_attribute"
    valid, message = validator.validate_date("2023-01-01", attribute_name)
    assert valid is True
    assert message == "Date is valid."
    valid, message = validator.validate_date("2023-12-31", attribute_name)
    assert valid is True
    assert message == "Date is valid."
    valid, message = validator.validate_date("2023-13-01", attribute_name)
    assert valid is False
    assert message == "test_attribute must be in the format YYYY-MM-DD."
    valid, message = validator.validate_date("2023-01-32", attribute_name)
    assert valid is False
    assert message == "test_attribute must be in the format YYYY-MM-DD."

def test_validate_language(validator):
    """Test the validate_language method."""
    valid, message = validator.validate_language("en")
    assert valid is True
    assert message == "All languages are valid."
    valid, message = validator.validate_language("en, fr")
    assert valid is True
    assert message == "All languages are valid."
    valid, message = validator.validate_language("French, German")
    assert valid is True
    assert message == "All languages are valid."
    valid, message = validator.validate_language("")
    assert valid is False
    assert "are not valid ISO language codes or names." in message
    valid, message = validator.validate_language("invalid_language")
    assert valid is False
    assert "are not valid ISO language codes or names." in message

def test_validate_cite_as(validator):
    """Test the validate_cite_as method."""
    valid, message = validator.validate_cite_as("@article{key, author = {Author}, title = {Title}}")
    assert valid is True
    assert message == "Citation is valid."
    valid, message = validator.validate_cite_as("")
    assert valid is False
    assert message == "Citation must be in valid BibTeX format."
    valid, message = validator.validate_cite_as("invalid_bibtex")
    assert valid is False
    assert message == "Citation must be in valid BibTeX format."

    with patch("builtins.all", side_effect=Exception("Mocked exception")):
        valid, message = validator.validate_cite_as("@article{key, author = {Author}, title = {Title}}")
        assert valid is False
        assert "Citation must be in valid BibTeX format. Error:" in message

def test_validate_all_attributes_no_errors(validator):
    """Test the validate_all_attributes method."""
    metadata = {
        "name": "Dataset Name",
        "author": "Author Name",
        "description": "Dataset Description",
        "url": "https://example.com",
        "publisher": "Publisher Name",
        "version": "1.0",
        "keywords": "keyword1, keyword2",
        "date_modified": "2023-01-01",
        "date_created": "2023-01-01",
        "date_published": "2023-01-01",
        "cite_as": "@article{key, author = {Author}, title = {Title}}",
        "language": "en, fr",
        "task": "task1, task2",
        "modality": "modality1, modality2"
    }
    errors = validator.validate_all_attributes(metadata)
    assert errors == {}

def test_validate_all_attributes_with_errors(validator):
    """Test the validate_all_attributes method with errors."""
    metadata = {
        "name": "",
        "author": "",
        "license": "invalid_license",
        "description": "",
        "url": "invalid-url",
        "publisher": "",
        "version": "",
        "keywords": "",
        "date_modified": "",
        "date_created": "",
        "date_published": "",
        "cite_as": "",
        "language": "",
        "task": "",
        "modality": ""
    }
    errors = validator.validate_all_attributes(metadata)
    print(errors)
    assert len(errors) > 0
    assert "name" in errors
    assert "author" in errors
    assert "description" in errors
    assert "license" in errors
    assert "url" in errors
    assert "publisher" in errors
    assert "version" in errors
    assert "keywords" in errors
    assert "date_modified" in errors
    assert "date_created" in errors
    assert "date_published" in errors
    assert "cite_as" in errors
    assert "language" in errors
    assert "task" in errors
    assert "modality" in errors


def test_validate_all_attributes_exception(validator):
    """Test the validate_all_attributes method when an exception occurs."""
    metadata = {
        "url": "https://example.com",
    }

    # Mock the validate_all_attributes method to raise an exception
    with patch.object(validator, "validate_url", side_effect=Exception("Mocked exception")):
        errors = validator.validate_all_attributes(metadata)
        assert "error" in errors
        assert errors["error"] == "An error occurred during validation: Mocked exception"


