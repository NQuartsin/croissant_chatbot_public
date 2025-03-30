# metadata_manager.py

# necessary imports
from huggingface_hub import HfApi
from .attribute_quality import AttributeQualityChecker
from .validation import MetadataValidator
from .constants import METADATA_ATTRIBUTES
import json
import os
import mlcroissant as mlc
import re
import unicodedata
from datetime import datetime
from typing import Tuple, Dict

class MetadataManager:
    """A class to manage metadata attributes and values for a dataset entry."""

    def __init__(self):
        self.metadata = {}
        self.final_metadata = {}
        self.confirmed_metadata = {}
        self.temporary_metadata = {}

    def reset_metadata(self):
        """
        Reset metadata to its initial state.
        """
        self.metadata = {}
        self.final_metadata = {}
        self.confirmed_metadata = {}
        self.temporary_metadata = {}

    def is_all_attributes_filled(self) -> bool:
        """
        Check if all metadata attributes are filled.

        Returns:
            True if all attributes are filled, False otherwise.
        """
        return all(attribute in self.metadata for attribute in METADATA_ATTRIBUTES)

    # Metadata Operations
    def get_metadata(self) -> Dict[str, str]:
        """
        Return the current metadata.

        Returns:
            The current metadata Dictionary.
        """
        return self.metadata

    def get_metadata_value(self, attribute: str) -> str:
        """
        Get the value of a specific metadata attribute.

        Args:
            attribute: The name of the metadata attribute.

        Returns:
            The value of the specified metadata attribute, or an empty string if not found.
        """
        return self.metadata.get(attribute, "")

    def update_metadata(self, updates: Dict[str, str]):
        """
        Update metadata with new values.

        Args:
            updates: A Dictionary containing metadata attributes and their new values.
        """
        self.metadata.update(updates)

    def set_metadata_value(self, attribute: str, value: str):
        """
        Set the value of a specific metadata attribute.

        Args:
            attribute: The name of the metadata attribute.
            value: The value to set for the metadata attribute.
        """
        self.metadata[attribute] = value

    # Temporary Metadata Operations
    def get_temporary_metadata_value(self, attribute: str) -> str:
        """
        Get the value of a specific temporary metadata attribute.

        Args:
            attribute: The name of the temporary metadata attribute.

        Returns:
            The value of the specified temporary metadata attribute, or an empty string if not found.
        """
        return self.temporary_metadata.get(attribute, "")

    def update_temporary_metadata(self, updates: Dict[str, str]):
        """
        Update temporary metadata with new values.

        Args:
            updates: A Dictionary containing temporary metadata attributes and their new values.
        """
        self.temporary_metadata.update(updates)

    def clear_temporary_metadata(self):
        """
        Clear all temporary metadata.
        """
        self.temporary_metadata = {}

    # Confirmed Metadata Operations
    def get_confirmed_metadata(self) -> Dict[str, str]:
        """
        Return the confirmed metadata.

        Returns:
            The confirmed metadata Dictionary.
        """
        return self.confirmed_metadata

    def confirm_metadata_value(self, attribute: str, value: str):
        """
        Confirm a metadata value despite validation issues.

        Args:
            attribute: The name of the metadata attribute.
            value: The value to confirm for the metadata attribute.
        """
        self.confirmed_metadata[attribute] = value
        self.metadata[attribute] = value

    def merge_confirmed_metadata(self):
        """
        Merge confirmed metadata into the main metadata.
        """
        self.metadata.update(self.confirmed_metadata)
        self.confirmed_metadata = {}

    # Metadata Validation and Quality Checks
    def validate_and_check_quality(self, attribute: str, value: str) -> Tuple[bool, str, str]:
        """
        Validate and check the quality of the attribute value.

        Args:
            attribute: The name of the metadata attribute.
            value: The value to validate and check.

        Returns:
            A Tuple containing a boolean indicating success, error messages, and issue messages.
        """
        # Validate the attribute value
        validator = MetadataValidator()
        errors = validator.validate_all_attributes({attribute: value})
        # Check the quality of the attribute value
        issues = AttributeQualityChecker().check_quality_of_all_attributes({attribute: value})
        # Check if there are any errors or issues
        if errors or issues:
            error_messages = "\n".join([f"{attribute}: {message}" for attribute, message in errors.items()]) if errors else ""
            issue_messages = "\n".join([f"{attribute}: {message}" for attribute, message in issues.items()]) if issues else ""
            return False, error_messages, issue_messages
        # If no errors or issues, return True
        return True, "", ""

    def validate_and_check_quality_all_attributes(self) -> Tuple[bool, str, str]:
        """
        Validate and check the quality of all attributes.

        Returns:
            A Tuple containing a boolean indicating success, error messages, and issue messages.
        """
        # Validate all attributes
        validator = MetadataValidator()
        errors = validator.validate_all_attributes(self.metadata)
        # Check quality of all attributes
        issues = AttributeQualityChecker().check_quality_of_all_attributes(self.metadata)
        # Check if there are any errors or issues
        if errors or issues:
            error_messages = "\n".join([f"{attribute}: {message}" for attribute, message in errors.items()]) if errors else ""
            issue_messages = "\n".join([f"{attribute}: {message}" for attribute, message in issues.items()]) if issues else ""
            return False, error_messages, issue_messages
        # If no errors or issues, return True
        return True, "", ""

    # Metadata File Operations
    def save_metadata_to_file(self, metadata: Dict[str, str]) -> Tuple[str, str]:
        """
        Save the metadata to a file in the annotations folder.

        Args:
            metadata: The metadata Dictionary to save.

        Returns:
            A Tuple containing the file path and filename, or an error message if saving fails.
        """
        try:
            # Get the path to the annotations folder
            base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            directory = os.path.join(base_directory, "annotations")

            # Ensure the directory exists
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Generate the filename
            filename = self.get_filename()
            filepath = os.path.join(directory, filename)

            # Save the file
            with open(filepath, "w") as file:
                json.dump(metadata, file, indent=2, default=self.json_serial)

            return filepath, filename
        except Exception as e:
            return None, f"Error saving metadata to file: {str(e)}"

    def json_serial(self, obj) -> str:
        """
        Serialize objects to JSON-compatible formats.

        Args:
            obj: The object to serialize.

        Returns:
            The serialized object as a string.
        """
        # Handle datetime objects
        try:
            if isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d")
        except TypeError as e:
            return f"{e} - {obj} is not JSON serializable"
        except Exception as e:
            return f"Error serializing object: {str(e)}"

    def remove_emojis(self, text: str) -> str:
        """
        Remove emojis from a string.

        Args:
            text: The input string.

        Returns:
            The string with emojis removed.
        """
        return ''.join(c for c in text if unicodedata.category(c) != 'So')

    def get_filename(self) -> str:
        """
        Get the filename for the metadata file.

        Returns:
            The sanitized filename for the metadata file or a default name.
        """
        if self.metadata.get("name"):
            # Remove emojis
            name_without_emojis = self.remove_emojis(self.metadata['name'])
            # Replace invalid filename characters with a hyphen (-)
            sanitised_name = re.sub(r'[<>:"/\\|?*]', '-', name_without_emojis)
            return f"{sanitised_name.replace(' ', '_').lower()}_metadata.json"
        return "metadata.json"

    # Finalise Metadata
    def finalise_metadata(self) -> Tuple[bool, Dict[str, str]]:
        """
        Finalise metadata in Croissant format.

        Returns:
            A Tuple containing a boolean indicating success and the final metadata or an error message.
        """
        try:
            attributes_to_remove = ["task", "modality"]
            filtered_metadata = {k:v for k, v in self.metadata.items() if k not in attributes_to_remove}
            # Create the Croissant metadata object
            croissant_metadata = mlc.Metadata(**filtered_metadata)

            # Set the task and modality fields
            self.final_metadata = croissant_metadata.to_json()
            task = self.metadata.get("task", "")
            modality = self.metadata.get("modality", "")
            if task:
                self.final_metadata["task"] = task
            if modality:
                self.final_metadata["modality"] = modality
            # Save metadata to a file
            filepath, filename = self.save_metadata_to_file(self.final_metadata)

            return True, self.final_metadata
        except Exception as e:
            return False, {'error': str(e)}

    # Fetch Dataset Info
    def find_dataset_info(self, dataset_id_to_find: str) -> Tuple[Dict[str, str], bool]:
        """
        Fetch dataset details.

        Args:
            dataset_id_to_find: The ID of the dataset to fetch details for.

        Returns:
            A Dictionary containing the dataset metadata or an error message if fetching fails.
        """
        try:
            # Fetch the dataset details using the Hugging Face Hub API
            api = HfApi() # Hugging Face API
            found_dataset = list(api.list_datasets(dataset_name={dataset_id_to_find}, limit=1))
            if not found_dataset:
                return None, False
            found_dataset = found_dataset[0] # Get the first dataset from the list
          
            # Extract relevant metadata fields
            # Initialize empty lists for tasks, modalities, and languages and a variable for license
            lisence = ""
            tasks = []
            modalities = []
            languages = []
            # Extract tags and populate the lists and variable
            for tag in found_dataset.tags:
                if tag.startswith("license:"):
                    lisence = tag.split(":", 1)[1]
                elif tag.startswith("task_categories:"):
                    tasks.append(tag.split(":", 1)[1])
                elif tag.startswith("modality:"):
                    modalities.append(tag.split(":", 1)[1])
                elif tag.startswith("language:"):
                    languages.append(tag.split(":", 1)[1])

            # Use getattr() for all fields to handle missing attributes gracefully
            dataset_id = getattr(found_dataset, "id", None)
            author = getattr(found_dataset, "author", None)
            last_modified = getattr(found_dataset, "last_modified", None)
            created_at = getattr(found_dataset, "created_at", None)
            description = getattr(found_dataset, "description", None)
            citation = getattr(found_dataset, "citation", None)

            # Only add attributes to metadata if they have a meaningful value
            if dataset_id:
                self.metadata["name"] = dataset_id
            if author:
                self.metadata["creators"] = author
            if last_modified:
                self.metadata["date_modified"] = last_modified.strftime("%Y-%m-%d")
            if created_at:
                self.metadata["date_created"] = created_at.strftime("%Y-%m-%d")
                self.metadata["date_published"] = created_at.strftime("%Y-%m-%d") # Assuming published date is same as created date
            if description:
                self.metadata["description"] = description
            if lisence:
                self.metadata["license"] = lisence
            if dataset_id:
                self.metadata["url"] = f"https://huggingface.co/datasets/{dataset_id}"
            if author:
                self.metadata["publisher"] = author
            if tasks:
                self.metadata["task"] = ", ".join(tasks)
            if modalities:
                self.metadata["modality"] = ", ".join(modalities)
            if languages:
                self.metadata["in_language"] = ", ".join(languages)
            if citation:
                self.metadata["cite_as"] = citation

            return self.metadata, True

        except Exception as e:
            return {'error': str(e)}, False
