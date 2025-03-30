# test_attribute_quality.py

# necessary imports
from unittest.mock import patch
import pytest
from main.attribute_quality import AttributeQualityChecker
import warnings
from lexical_diversity import lex_div as ld 
import spacy  

nlp = spacy.load("en_core_web_sm")


# Suppress DeprecationWarning for pkg_resources
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lexical_diversity")
"""
Test cases for the AttributeQualityChecker class.
"""

@pytest.fixture
def checker():
    """Fixture to create an AttributeQualityChecker instance."""
    return AttributeQualityChecker()

def test_check_description(checker):
    """Test the check_description method."""
    # description too short
    valid, message = checker.check_description("This is an invalid description with insufficient detail.")
    assert valid is False
    assert "The description lacks lexical diversity and may be repetitive." in message
    assert "The description has limited sentence variety and may be monotonous." in message

    # description long enough to calculate lexical diversity
    valid, message = checker.check_description("The unique strategy boosts productivity. The unique method boosts productivity. The unique method boosts productivity. The unique method boosts productivity.")
    assert valid is False
    assert "The description lacks lexical diversity and may be repetitive." in message
    assert "The description has limited sentence variety and may be monotonous." in message

    # good description
    valid, message = checker.check_description("The innovative model enhances efficiency, while researchers explore new optimization techniques. By leveraging cutting-edge technology, the system adapts dynamically to changing conditions. As a result, overall productivity sees a significant boost.")
    assert valid is True
    assert "Description quality is acceptable." in message
    assert "The description lacks lexical diversity and may be repetitive." not in message
    assert "The description has limited sentence variety and may be monotonous." not in message

def test_check_description_errors(checker):
    """Test the check_description method with error cases."""
    # error in description
    with patch("main.attribute_quality.AttributeQualityChecker.calculate_lexical_diversity", side_effect=Exception("Mocked exception")):
        valid, message = checker.check_description("This is a test description.")
        assert valid is False
        assert "Unexpected error in description quality check: Mocked exception" in message
        
    # lex error case
    with patch("main.attribute_quality.len", side_effect=Exception("Mocked exception")):
        valid, message = checker.check_description("This is a test description.")
        valid is False
        assert "Unexpected error in lexical diversity calculation: Mocked exception" in message

def test_calculate_lexical_diversity(checker):
    """Test the calculate_lexical_diversity method."""
    # empty string
    score, error = checker.calculate_lexical_diversity("")
    assert score == 0
    assert error is None

    # value too short
    score, error = checker.calculate_lexical_diversity("This is an invalid description with insufficient detail.")
    assert score == 0
    assert error is None

    # value long enough to calculate lexical diversity
    score, error = checker.calculate_lexical_diversity("The unique strategy boosts productivity. The unique method boosts productivity. The unique method boosts productivity. The unique method boosts productivity.")
    assert score > 0
    assert error is None

    # error case
    with patch("main.attribute_quality.len", side_effect=Exception("Mocked exception")):
        score, error = checker.calculate_lexical_diversity("This is a test description.")
        assert score == 0
        assert "Unexpected error in lexical diversity calculation: Mocked exception" in error

def test_calculate_sentence_variety(checker):
    """Test the calculate_sentence_variety method."""
    # one sentence
    count, error = checker.calculate_sentence_variety("This is an invalid description with insufficient detail.")
    assert count == 1
    assert error is None

    # more than one sentence
    count, error = checker.calculate_sentence_variety("The unique strategy boosts productivity. The unique method boosts productivity. The unique method boosts productivity. The unique method boosts productivity.")
    assert count > 0
    assert error is None

    # error case
    with patch("main.attribute_quality.len", side_effect=Exception("Mocked exception")):
        count, error = checker.calculate_sentence_variety("This is a test description.")
        assert count == 0
        assert "Unexpected error in sentence variety calculation: Mocked exception" in error

def test_get_sentence_structure(checker):
    """Test the get_sentence_structures method."""
    # empty string
    structure, errors = checker.get_sentence_structure("")
    assert structure == ''
    assert errors is None

    # valid value for getting sentence structure
    structure, errors = checker.get_sentence_structure(nlp("This is an invalid description with insufficient detail."))
    assert len(structure) > 0
    assert errors is None

    # invalid value for getting sentence structure
    structure, errors = checker.get_sentence_structure("Invalid_input")
    assert structure == ''
    assert "Error in extracting sentence structure:" in errors

def test_check_keywords(checker):
    """Test the check_keywords method."""
    # empty string
    valid, message = checker.check_keywords("")
    assert valid is False
    assert "Please provide at least 3 keywords." in message

    # not enough keywords
    valid, message = checker.check_keywords("keyword1, keyword2")
    assert valid is False
    assert "Please provide at least 3 keywords." in message

    # repeated keywords
    valid, message = checker.check_keywords("keyword1, keyword2, keyword1")
    assert valid is False
    assert "Keywords should not have any repeated words." in message

    # valid keywords
    valid, message = checker.check_keywords("keyword1, keyword2, keyword3")
    assert valid is True
    assert "Keywords quality is acceptable." in message

    # error case
    with patch("main.attribute_quality.len", side_effect=Exception("Mocked exception")):
        valid, message = checker.check_keywords("keyword1, keyword2")
        assert valid is False
        assert "Error in keywords quality check: Mocked exception" in message

def test_check_attribute_quality(checker):
    """Test the check_attribute_quality method."""
    # empty string
    valid, message = checker.check_attribute_quality("keywords", "")
    assert valid is False
    assert "Please provide at least 3 keywords." in message

    # invalid attribute name
    valid, message = checker.check_attribute_quality("invalid_attribute", "value")
    assert valid is False
    assert "Invalid attribute name: invalid_attribute" in message

    # valid attribute name and value
    valid, message = checker.check_attribute_quality("description", "The innovative model enhances efficiency, while researchers explore new optimization techniques. By leveraging cutting-edge technology, the system adapts dynamically to changing conditions. As a result, overall productivity sees a significant boost.")
    assert valid is True
    assert "Description quality is acceptable." in message

    # attribute name not description or keywords
    valid, message = checker.check_attribute_quality("name", "value")
    assert valid is True
    assert "Attribute quality is acceptable." in message

def test_check_quality_of_all_attributes(checker):
    """Test the check_quality_of_all_attributes method."""
    # empty metadata
    metadata = {}
    result = checker.check_quality_of_all_attributes(metadata)
    assert result == {}

    # valid metadata
    metadata = {
        "description": "The innovative model enhances efficiency, while researchers explore new optimization techniques. By leveraging cutting-edge technology, the system adapts dynamically to changing conditions. As a result, overall productivity sees a significant boost.",
        "keywords": "keyword1, keyword2, keyword3"
    }
    result = checker.check_quality_of_all_attributes(metadata)
    assert result == {}

    # invalid metadata
    metadata = {
        "description": "This is an invalid description with insufficient detail.",
        "keywords": ""
    }
    result = checker.check_quality_of_all_attributes(metadata)
    assert result == {
        "description": "The description lacks lexical diversity and may be repetitive.The description has limited sentence variety and may be monotonous.",
        "keywords": "Please provide at least 3 keywords."
    }

    # error case
    metadata = {
        "description": "This is a test description.",
        "keywords": "keyword1, keyword2"
    }
    with patch("main.attribute_quality.AttributeQualityChecker.check_attribute_quality", side_effect=Exception("Mocked exception")):
        result = checker.check_quality_of_all_attributes(metadata)
        assert result == {"error": "Error in checking quality of all attributes: Mocked exception"}





