# metadata_manager.py
from huggingface_hub import list_datasets
from attribute_quality import AttributeQualityChecker
from validation import MetadataValidator
from constants import METADATA_ATTRIBUTES
import json
import os
import mlcroissant as mlc
import re
import unicodedata
from datetime import datetime



class MetadataManager:
    """Manage metadata attributes and values for a dataset entry."""

    def __init__(self):
        self.metadata = {}
        self.final_metadata = {}
        self.confirmed_metadata = {}
        self.temporary_metadata = {}

    
    def reset_metadata(self):
        """Reset metadata to its initial state."""
        self.metadata = {}
        self.final_metadata = {}
        self.confirmed_metadata = {}
        self.temporary_metadata = {}
    
    def is_all_attributes_filled(self):
        """Check if all metadata attributes are filled."""
        return all(attribute in self.metadata for attribute in METADATA_ATTRIBUTES)
    
    # Metadata Operations
    def get_metadata(self):
        """Return the current metadata."""
        return self.metadata

    def get_metadata_value(self, attribute):
        """Get the value of a specific metadata attribute."""
        return self.metadata.get(attribute, "")

    def update_metadata(self, updates):
        """Update metadata with new values."""
        self.metadata.update(updates)
     
    def set_metadata(self, attribute, value):
        """Set the value of a specific metadata attribute."""
        self.metadata[attribute] = value

    # Temporary Metadata Operations
    def get_temporary_metadata_value(self, attribute):
        """Get the value of a specific temporary metadata attribute."""
        return self.temporary_metadata.get(attribute, "")

    def update_temporary_metadata(self, updates):
        """Update temporary metadata with new values."""
        self.temporary_metadata.update(updates)

    def clear_temporary_metadata(self):
        """Clear all temporary metadata."""
        self.temporary_metadata = {}


    # Confirmed Metadata Operations
    def get_confirmed_metadata(self):
        """Return the confirmed metadata."""
        return self.confirmed_metadata
    
    def confirm_metadata_value(self, attribute, value):
        """Confirm a metadata value despite validation issues."""
        self.confirmed_metadata[attribute] = value
        self.metadata[attribute] = value

    def merge_confirmed_metadata(self):
        """Merge confirmed metadata into the main metadata."""
        self.metadata.update(self.confirmed_metadata)
        self.confirmed_metadata = {}
    
    def validate_and_check_quality(self, attribute, value):
        """Validate and check the quality of the attribute value."""
        # Validate the attribute value
        validator = MetadataValidator()
        errors = validator.validate_all_attributes({attribute: value})
        issues = AttributeQualityChecker().check_quality_of_all_attributes({attribute: value})

        if errors or issues:
            error_messages = "\n".join([f"{attribute}: {message}" for attribute, message in errors.items()]) if errors else ""
            issue_messages = "\n".join([f"{attribute}: {message}" for attribute, message in issues.items()]) if issues else ""
            return False, error_messages, issue_messages
        return True, "",""
    
    def validate_and_check_quality_all_attributes(self):
        """Validate and check the quality of all attributes."""
        validator = MetadataValidator()
        errors = validator.validate_all_attributes(self.metadata)
        issues = AttributeQualityChecker().check_quality_of_all_attributes(self.metadata)
        if errors or issues:
            error_messages = "\n".join([f"{attribute}: {message}" for attribute, message in errors.items()]) if errors else ""
            issue_messages = "\n".join([f"{attribute}: {message}" for attribute, message in issues.items()]) if issues else ""
            return False, error_messages, issue_messages
        return True, "", ""
    
    # def save_metadata_to_file(self, metadata):
    #     """Save the metadata to a file in the annotations folder."""
    #     directory = "annotations"  # Specify the folder where files should be saved
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)  # Create the folder if it doesn't exist

    #     # Generate the filename
    #     filename = self.get_filename()
    #     filepath = os.path.join(directory, filename)  # Save in the annotations folder

    #     # Save the file
    #     with open(filepath, "w") as file:
    #         json.dump(metadata, file, indent=2, default=self.json_serial)

    #     return filepath, filename 

    # def save_metadata_to_file(self, metadata):
    #     """Save the metadata to a file in the annotations folder inside croissant_chatbot."""
    #     # Get the path to the croissant_chatbot directory
    #     base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #     directory = os.path.join(base_directory, "annotations")  # Create the annotations folder in croissant_chatbot

    #     # Ensure the directory exists
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)

    #     # Generate the filename
    #     filename = self.get_filename()
    #     filepath = os.path.join(directory, filename)

    #     # Save the file
    #     with open(filepath, "w") as file:
    #         json.dump(metadata, file, indent=2, default=self.json_serial)

    #     return filepath, filename

    def save_metadata_to_file(self, metadata):
        """Save the metadata to a file in the annotations folder."""
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

    def json_serial(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d")
        raise TypeError(f"Type {type(obj)} not serialisable")    
    
    def remove_emojis(self, text):
        """Remove emojis from a string."""
        return ''.join(c for c in text if unicodedata.category(c) != 'So')

    def get_filename(self):
        """Get the filename for the metadata file."""
        if self.metadata.get("name"):
            # Remove emojis
            name_without_emojis = self.remove_emojis(self.metadata['name'])
            # Replace invalid filename characters with a hyphen (-)
            sanitised_name = re.sub(r'[<>:"/\\|?*]', '-', name_without_emojis)
            return f"{sanitised_name.replace(' ', '_').lower()}_metadata.json"
        return "metadata.json"

    def finalise_metadata(self):
        """Finalise metadata in Croissant format."""
        
        try:

            # Map your metadata fields to the expected Croissant Metadata fields
            metadata_mapping = {
                "name": "name",
                "author": "creators",
                "description": "description",
                "license": "license",
                "url": "url",
                "publisher": "publisher",
                "version": "version",
                "keywords": "keywords",
                "date_modified": "date_modified",
                "date_created": "date_created",
                "date_published": "date_published",
                "cite_as": "cite_as",
                "language": "in_language",
            }

            # Dynamically build the metadata dictionary for the Croissant object
            filtered_metadata = {
                croissant_field: self.metadata[original_field]
                for original_field, croissant_field in metadata_mapping.items()
                if original_field in self.metadata and self.metadata[original_field]
            }

            # Create the Croissant metadata object
            croissant_metadata = mlc.Metadata(**filtered_metadata)           

            # Set the task and modality fields
            self.final_metadata = croissant_metadata.to_json()
            task = self.metadata.get("task", "")
            modality = self.metadata.get("modality", "")
            if task:
                self.final_metadata["tasks"] = task
            if modality:
                self.final_metadata["modalitiy"] = modality

            # Save metadata to a file
            filepath, filename = self.save_metadata_to_file(self.final_metadata)

            return True, self.final_metadata
        except Exception as e:
            return False, str(e)

    def find_dataset_info(self, dataset_id):
        """Fetch dataset details."""

        datasets = list_datasets()
        dataset = next((d for d in datasets if d.id == dataset_id), None)
        if not dataset:
            return None
        lisence = ""
        tasks = []
        modalities = []
        languages = []
        
        for tag in dataset.tags:
            if tag.startswith("license:"):
                lisence = tag.split(":", 1)[1]
            elif tag.startswith("task_categories:"):
                tasks.append(tag.split(":", 1)[1])
            elif tag.startswith("modality:"):
                modalities.append(tag.split(":", 1)[1])
            elif tag.startswith("language:"):
                languages.append(tag.split(":", 1)[1])
        
        # Use getattr() for all fields to handle missing attributes gracefully
        dataset_id = getattr(dataset, "id", None)
        author = getattr(dataset, "author", None)
        last_modified = getattr(dataset, "last_modified", None)
        created_at = getattr(dataset, "created_at", None)
        description = getattr(dataset, "description", None)
        citation = getattr(dataset, "citation", None)


        # Only add attributes to metadata if they have a meaningful value
        if dataset_id:
            self.metadata["name"] = dataset_id
        if author:
            self.metadata["author"] = author
        if last_modified:
            self.metadata["year"] = last_modified.year
            self.metadata["date_modified"] = last_modified.strftime("%Y-%m-%d")
        if created_at:
            self.metadata["date_created"] = created_at.strftime("%Y-%m-%d")
            self.metadata["date_published"] = created_at.strftime("%Y-%m-%d")
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
            self.metadata["language"] = ", ".join(languages)
        if citation:
            self.metadata["cite_as"] = citation

        return self.metadata
