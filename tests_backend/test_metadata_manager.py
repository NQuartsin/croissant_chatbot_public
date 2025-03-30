# test_metadata_manager.py

# necessary imports
import pytest
from unittest.mock import patch, mock_open
from main.metadata_manager import MetadataManager
from main.attribute_quality import AttributeQualityChecker
from main.validation import MetadataValidator
import os
import json
import datetime

"""
Test cases for the MetadataManager class.
"""

@pytest.fixture
def metadata_manager(sample_metadata, sample_final_metadata, sample_confirmed_metadata, sample_temporary_metadata):
    """Fixture to create a MetadataManager instance with sample metadata."""
    manager = MetadataManager()
    manager.metadata = sample_metadata  # Assign sample_metadata to self.metadata
    manager.final_metadata = sample_final_metadata
    manager.confirmed_metadata = sample_confirmed_metadata
    manager.temporary_metadata = sample_temporary_metadata
    return manager

@pytest.fixture
def sample_metadata():
    """Fixture to provide sample metadata."""
    return {
        "name": "Sample Dataset",
        "creators": "John Doe",
    }

@pytest.fixture
def sample_final_metadata():
    """Fixture to provide sample final metadata."""
    return {
        "name": "Sample Dataset",
        "creators": "John Doe",
        "description": "This is a sample dataset.",
        "license": "MIT",
        "url": "https://example.com",
        "publisher": "Sample Publisher",
        "version": "1.0",
        "keywords": "sample, dataset",
        "date_modified": "2023-01-01",
        "date_created": "2023-01-01",
        "date_published": "2023-01-01",
        "cite_as": "@article{sample2023}",
        "in_language": "en",
        "task": "sample task",
        "modality": "text",
    }

@pytest.fixture
def sample_confirmed_metadata():
    """Fixture to provide sample confirmed metadata."""
    return {
        "description": "This is a sample dataset.",
    }

@pytest.fixture
def sample_temporary_metadata():
    """Fixture to provide sample temporary metadata."""
    return {
        "description": "This is a sample dataset.",
    }

@pytest.fixture
def metadata_manager_with_final_metadata(sample_final_metadata):
    """Fixture to create a MetadataManager instance with sample final metadata."""
    manager = MetadataManager()
    manager.metadata = sample_final_metadata  # Assign sample_final_metadata to self.metadata
    return manager

def test_reset_metadata(metadata_manager):
    """Test the reset_metadata method."""
    # Check initial state
    assert metadata_manager.metadata != {}
    assert metadata_manager.final_metadata != {}
    assert metadata_manager.confirmed_metadata != {}
    assert metadata_manager.temporary_metadata != {}
    # Reset metadata
    metadata_manager.reset_metadata()
    # Check if all metadata variables are reset to empty dictionaries
    assert metadata_manager.metadata == {}
    assert metadata_manager.final_metadata == {}
    assert metadata_manager.confirmed_metadata == {}
    assert metadata_manager.temporary_metadata == {}

def test_is_all_attributes_filled(metadata_manager_with_final_metadata):
    """Test the is_all_attributes_filled method."""
    # All attributes filled
    assert metadata_manager_with_final_metadata.is_all_attributes_filled() is True

    # Missing attributes
    incomplete_metadata = metadata_manager_with_final_metadata.get_metadata().copy()
    incomplete_metadata.pop("creators")
    metadata_manager_with_final_metadata.metadata = incomplete_metadata
    assert metadata_manager_with_final_metadata.is_all_attributes_filled() is False

def test_get_metadata(metadata_manager):
    """Test the get_metadata method."""
    # Check if the metadata is returned correctly
    assert metadata_manager.get_metadata() == metadata_manager.metadata

def test_get_metadata_value(metadata_manager):
    """Test the get_metadata_value method."""
    # Check if the metadata value is returned correctly
    assert metadata_manager.get_metadata_value("name") == "Sample Dataset"
    assert metadata_manager.get_metadata_value("creators") == "John Doe"

    # Test with a non-existent key
    assert metadata_manager.get_metadata_value("non_existent_key") == ""

def test_update_metadata(metadata_manager):
    """Test the update_metadata method."""
    # Check initial state
    assert metadata_manager.metadata == {
        "name": "Sample Dataset",
        "creators": "John Doe",
    }
    # Update metadata
    new_metadata = {
        "name": "Updated Dataset",
        "creators": "Jane Doe",
    }
    metadata_manager.update_metadata(new_metadata)
    # Check if the metadata is updated correctly
    assert metadata_manager.get_metadata() == new_metadata

def test_set_metadata_value(metadata_manager):
    """Test the set_metadata_value method."""
    # Check initial state
    assert metadata_manager.metadata["name"] == "Sample Dataset"
    # Set new value
    metadata_manager.set_metadata_value("name", "Updated Dataset")
    # Check if the value is updated correctly
    assert metadata_manager.metadata["name"] == "Updated Dataset"
    # Test with a non-existent key
    metadata_manager.set_metadata_value("non_existent_key", "New Value")
    # Check if the non-existent key is added to metadata
    assert metadata_manager.metadata["non_existent_key"] == "New Value"

def test_get_temporary_metadata_value(metadata_manager):
    """Test the get_temporary_metadata_value method."""
    assert metadata_manager.get_temporary_metadata_value("description") == "This is a sample dataset."
    assert metadata_manager.get_temporary_metadata_value("non_existent_key") == ""

def test_update_temporary_metadata(metadata_manager):
    """Test the update_temporary_metadata method."""
    # Check initial state
    assert metadata_manager.temporary_metadata == {
        "description": "This is a sample dataset.",
    }
    # Update temporary metadata
    new_temporary_metadata = {
        "description": "Updated description",
    }
    metadata_manager.update_temporary_metadata(new_temporary_metadata)
    # Check if the temporary metadata is updated correctly
    assert metadata_manager.get_temporary_metadata_value("description") == "Updated description"

    # Test with a non-existent key
    metadata_manager.update_temporary_metadata({"non_existent_key": "New Value"})
    # Check if the non-existent key is added to temporary metadata
    assert metadata_manager.temporary_metadata["non_existent_key"] == "New Value"

def test_clear_temporary_metadata(metadata_manager):
    """Test the clear_temporary_metadata method."""
    # Check initial state
    assert metadata_manager.temporary_metadata != {}
    # Clear temporary metadata
    metadata_manager.clear_temporary_metadata()
    # Check if temporary metadata is cleared
    assert metadata_manager.temporary_metadata == {}

def test_get_confirmed_metadata(metadata_manager):
    """Test the get_confirmed_metadata method."""
    # Check if the confirmed metadata is returned correctly
    assert metadata_manager.get_confirmed_metadata() == metadata_manager.confirmed_metadata
    # Test with empty confirmed metadata
    metadata_manager.confirmed_metadata = {}
    assert metadata_manager.get_confirmed_metadata() == {}

def test_confirm_metadata_value(metadata_manager):
    """Test the confirm_metadata_value method."""
    # Check initial state
    metadata_manager.confirmed_metadata = {} # clear confirmed metadata
    assert metadata_manager.get_confirmed_metadata() == {} # check if confirmed metadata is empty
    assert metadata_manager.get_metadata_value("description") == "" 
    # Confirm metadata value
    metadata_manager.confirm_metadata_value("description", "This is a sample dataset.")
    # Check if the confirmed metadata is updated correctly
    assert metadata_manager.confirmed_metadata["description"] == "This is a sample dataset."
    # Check if metadata has the confirmed value
    assert metadata_manager.metadata["description"] == "This is a sample dataset."

def test_merge_confirmed_metadata(metadata_manager):
    """Test the merge_confirmed_metadata method."""
    # Check initial state
    assert metadata_manager.metadata == {
        "name": "Sample Dataset",
        "creators": "John Doe",
    }
    assert metadata_manager.confirmed_metadata == {
        "description": "This is a sample dataset.",
    }
    # Merge confirmed metadata
    metadata_manager.merge_confirmed_metadata()
    # Check if the confirmed metadata is merged correctly
    assert metadata_manager.metadata["description"] == "This is a sample dataset."

def test_validate_and_check_quality_sucess(metadata_manager):
    """Test the validate_and_check_quality method."""
    # Mock the MetadataValidator and AttributeQualityChecker classes
    with patch.object(MetadataValidator, 'validate_all_attributes') as mock_validate_metadata, \
         patch.object(AttributeQualityChecker, 'check_quality_of_all_attributes') as mock_check_quality:
        # Set return values for the mocked methods
        mock_validate_metadata.return_value = {}
        mock_check_quality.return_value = {}

        # Call the method
        success, errors, issues  = metadata_manager.validate_and_check_quality("name", "Sample Dataset")

        # Check if the methods were called
        mock_validate_metadata.assert_called_once()
        mock_check_quality.assert_called_once()

        # Check the result
        assert success is True
        assert errors == ''
        assert issues == ''

def test_validate_and_check_quality_with_issues(metadata_manager):
    """Test the validate_and_check_quality method for failure cases."""
    # Mock the MetadataValidator and AttributeQualityChecker classes
    with patch.object(MetadataValidator, 'validate_all_attributes') as mock_validate_metadata, \
         patch.object(AttributeQualityChecker, 'check_quality_of_all_attributes') as mock_check_quality:
        # Set return values for the mocked methods
        mock_validate_metadata.return_value = {"name": "name must be a non-empty string."}

        # Call the method
        success, errors, issues  = metadata_manager.validate_and_check_quality("name", "")

        # Check if the methods were called
        mock_validate_metadata.assert_called_once()
        mock_check_quality.assert_called_once()

        # Check the result
        assert success is False
        assert errors == "name: name must be a non-empty string."
        assert issues == ""

def test_validate_and_check_quality_with_errors(metadata_manager):
    """Test the validate_and_check_quality method for failure cases."""
    # Mock the MetadataValidator and AttributeQualityChecker classes
    with patch.object(MetadataValidator, 'validate_all_attributes') as mock_validate_metadata, \
         patch.object(AttributeQualityChecker, 'check_quality_of_all_attributes') as mock_check_quality:
        # Set return values for the mocked methods
        mock_check_quality.return_value = {"keywords": "keywords should not have any repeated words."}

        # Call the method
        success, errors, issues  = metadata_manager.validate_and_check_quality("keyword", "keyword1, keyword2, keyword1")

        # Check if the methods were called
        mock_validate_metadata.assert_called_once()
        mock_check_quality.assert_called_once()

        # Check the result
        assert success is False
        assert errors == ""
        assert issues == "keywords: keywords should not have any repeated words."

def test_validate_and_check_quality_all_attributes_sucess(metadata_manager):
    """Test the validate_and_check_quality_all_attributes method for success cases."""
    # Mock the MetadataValidator and AttributeQualityChecker classes
    with patch.object(MetadataValidator, 'validate_all_attributes') as mock_validate_metadata, \
         patch.object(AttributeQualityChecker, 'check_quality_of_all_attributes') as mock_check_quality:
        # Set return values for the mocked methods
        mock_validate_metadata.return_value = {}
        mock_check_quality.return_value = {}

        # Call the method
        success, errors, issues  = metadata_manager.validate_and_check_quality_all_attributes()

        # Check if the methods were called
        mock_validate_metadata.assert_called_once()
        mock_check_quality.assert_called_once()

        # Check the result
        assert success is True
        assert errors == ''
        assert issues == ''

def test_validate_and_check_quality_all_attributes_with_issues_and_errors(metadata_manager):
    """Test the validate_and_check_quality_all_attributes method for failure cases."""
    # Mock the MetadataValidator and AttributeQualityChecker classes
    with patch.object(MetadataValidator, 'validate_all_attributes') as mock_validate_metadata, \
         patch.object(AttributeQualityChecker, 'check_quality_of_all_attributes') as mock_check_quality:
        # Set return values for the mocked methods
        metadata_manager.metadata["name"] = ""
        metadata_manager.metadata["keywords"] = "keyword1, keyword2, keyword1"
        mock_validate_metadata.return_value = {"name": "name must be a non-empty string."}
        mock_check_quality.return_value = {"keywords": "keywords should not have any repeated words."}

        # Call the method
        success, errors, issues  = metadata_manager.validate_and_check_quality_all_attributes()

        # Check if the methods were called
        mock_validate_metadata.assert_called_once()
        mock_check_quality.assert_called_once()

        # Check the result
        assert success is False
        assert errors == "name: name must be a non-empty string."
        assert issues == "keywords: keywords should not have any repeated words."

def test_save_metadata_to_file(metadata_manager, sample_metadata):
    """Test the save_metadata_to_file method."""
    # Mock os.makedirs, os.path.exists, and open
    with patch("os.makedirs") as mock_makedirs, \
         patch("os.path.exists", return_value=False) as mock_exists, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch.object(metadata_manager, "get_filename", return_value="test_metadata.json") as mock_get_filename:
        
        # Call the method
        filepath, filename = metadata_manager.save_metadata_to_file(sample_metadata)

        # Check if os.makedirs was called
        mock_makedirs.assert_called_once_with(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annotations"))

        # Check if the file was opened
        mock_file.assert_called_once_with(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annotations", "test_metadata.json"), "w")

        # Concatenate all write calls
        written_content = "".join(call.args[0] for call in mock_file().write.call_args_list)

        # Check the written content
        expected_content = json.dumps(sample_metadata, indent=2, default=metadata_manager.json_serial)
        assert written_content == expected_content

        # Check the returned values
        assert filepath.endswith("annotations/test_metadata.json")
        assert filename == "test_metadata.json"

def test_json_serial(metadata_manager):
    """Test the json_serial method."""

    # Test with a datetime object
    datetime_obj = datetime.datetime(2023, 1, 1)
    date = metadata_manager.json_serial(datetime_obj)
    assert date == "2023-01-01"

    # Test with a date string
    date = metadata_manager.json_serial("2023-01-01")
    assert date is None

def test_remove_emojis(metadata_manager):
    """Test the remove_emojis method."""
    # Test with a string containing emojis
    text_with_emojis = "This is a test 😊"
    cleaned_text = metadata_manager.remove_emojis(text_with_emojis)
    assert cleaned_text == "This is a test "

    # Test with a string without emojis
    text_without_emojis = "This is a test"
    cleaned_text = metadata_manager.remove_emojis(text_without_emojis)
    assert cleaned_text == "This is a test"

def test_get_filename(metadata_manager):
    """Test the get_filename method."""
    # Test with a valid name
    metadata_manager.metadata["name"] = "Sample Dataset"
    filename = metadata_manager.get_filename()
    assert filename == "sample_dataset_metadata.json"

    # Test with an empty name
    metadata_manager.metadata["name"] = ""
    filename = metadata_manager.get_filename()
    assert filename == "metadata.json"

    # Test with a name containing special characters
    metadata_manager.metadata["name"] = "file/Sample Dataset?😊"
    filename = metadata_manager.get_filename()
    assert filename == "file-sample_dataset-_metadata.json"

def test_finalise_metadata(metadata_manager):
    """Test the finalise_metadata method without writing files."""
    metadata_manager.metadata["task"] = "sample task"
    metadata_manager.metadata["modality"] = "text"
    metadata_manager.final_metadata = {}
    assert metadata_manager.final_metadata == {}
    with patch.object(MetadataManager, 'save_metadata_to_file', return_value=("mock_path", "mock_file.json")):
        success, final_metadata = metadata_manager.finalise_metadata()
    # Check if final metadata is updated correctly
    assert success is True
    assert "name" in final_metadata
    assert final_metadata["name"] == "Sample Dataset"
    assert "creator" in final_metadata
    assert final_metadata["creator"] == "John Doe"
    assert "task" in final_metadata
    assert final_metadata["task"] == "sample task"
    assert "modality" in final_metadata
    assert final_metadata["modality"] == "text"

def test_finalise_metadata_with_empty_metadata(metadata_manager):
    """Test the finalise_metadata method with empty metadata (without writing files)."""
    metadata_manager.metadata = {}
    metadata_manager.final_metadata = {}
    assert metadata_manager.final_metadata == {}
    with patch.object(MetadataManager, 'save_metadata_to_file', return_value=("mock_path", "mock_file.json")):
        success, final_metadata = metadata_manager.finalise_metadata()
    # Ensure final metadata is still valid
    assert success is True
    assert isinstance(final_metadata, dict)


def test_finalise_metadata_unexpected_error(metadata_manager):
    """Test the finalise_metadata method with unexpected error."""
    # Check initial state
    metadata_manager.final_metadata = {}
    assert metadata_manager.final_metadata == {}

    # Mock the save_metadata_to_file method to raise an exception
    with patch.object(metadata_manager, "save_metadata_to_file", side_effect=Exception("Mocked exception")):
        # Finalise metadata
        success, final_metadata = metadata_manager.finalise_metadata()

        # Check if final metadata is updated correctly
        assert success is False
        assert "Mocked exception" in final_metadata["error"]

def test_find_dataset_info(metadata_manager):
    """Test the find_dataset_info method."""
    # Mock the HfApi.list_datasets method to simulate fetching dataset info
    with patch("main.metadata_manager.HfApi.list_datasets", return_value=[
        type("Dataset", (object,), {
            "id": "sample_dataset_id",
            "author": "John Doe",
            "last_modified": datetime.datetime(2023, 1, 1),
            "created_at": datetime.datetime(2023, 1, 1),
            "description": "This is a sample dataset.",
            "citation": "@article{sample2023}",
            "tags": [
                "license:MIT",
                "task_categories:classification",
                "modality:text",
                "language:en"
            ]
        })()
    ]):
        # Call the actual method
        dataset_info, success = metadata_manager.find_dataset_info("sample_dataset_id")

        # Check the returned values
        assert success is True
        assert dataset_info["name"] == "sample_dataset_id"
        assert dataset_info["creators"] == "John Doe"
        assert dataset_info["date_modified"] == "2023-01-01"
        assert dataset_info["date_created"] == "2023-01-01"
        assert dataset_info["description"] == "This is a sample dataset."
        assert dataset_info["license"] == "MIT"
        assert dataset_info["task"] == "classification"
        assert dataset_info["modality"] == "text"
        assert dataset_info["in_language"] == "en"

def test_find_dataset_info_missing_tags(metadata_manager):
    """Test the find_dataset_info method with missing tags."""
    # Mock the HfApi.list_datasets method to simulate fetching dataset info
    with patch("main.metadata_manager.HfApi.list_datasets", return_value=[
        type("Dataset", (object,), {
            "id": "sample_dataset_id",
            "author": "John Doe",
            "last_modified": datetime.datetime(2023, 1, 1),
            "created_at": datetime.datetime(2023, 1, 1),
            "description": "This is a sample dataset.",
            "citation": "@article{sample2023}",
            "tags": [
                "license:MIT",
                "task_categories:classification",
            ]
        })()
    ]):
        # Call the actual method
        dataset_info, success = metadata_manager.find_dataset_info("sample_dataset_id")

        # Check the returned values
        assert success is True
        assert dataset_info["name"] == "sample_dataset_id"
        assert dataset_info["creators"] == "John Doe"
        assert dataset_info["date_modified"] == "2023-01-01"
        assert dataset_info["date_created"] == "2023-01-01"
        assert dataset_info["description"] == "This is a sample dataset."
        assert dataset_info["license"] == "MIT"
        assert dataset_info["task"] == "classification"


def test_find_dataset_info_id_returns_nothing(metadata_manager):
    """Test the find_dataset_info method with an ID that returns nothing."""
    # Mock the HfApi.list_datasets method to simulate fetching dataset info
    metadata_manager.metadata = {}
    with patch("main.metadata_manager.HfApi.list_datasets", return_value=[
        type("Dataset", (object,), {
            "id": "",
            "author": "",
            "description": "",
            "citation": "",
            "tags": []
        })()
    ]):
        # Call the actual method
        dataset_info, success = metadata_manager.find_dataset_info("sample_dataset_id")

        # Check the returned values
        assert success is True
        assert "name" not in dataset_info
        

def test_find_dataset_info_with_invalid_id(metadata_manager):
    """Test the find_dataset_info method with an invalid dataset ID."""
    # Mock the HfApi.list_datasets method to simulate fetching dataset info
    with patch("main.metadata_manager.HfApi.list_datasets", return_value=None):
        # Call the actual method
        dataset_info, success = metadata_manager.find_dataset_info("invalid_dataset_id")

        # Check the returned values
        assert success is False
        assert "error" in dataset_info