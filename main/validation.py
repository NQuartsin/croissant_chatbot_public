# validation.py

# Necessary imports
import re
from datetime import datetime
import langcodes 
import bibtexparser 
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding
import json
from typing import Tuple, Dict

class MetadataValidator():
    """A class to validate metadata attributes and values for a dataset entry."""

    def validate_url(self, url: str) -> Tuple[bool, str]:
        """
        Ensure the URL is valid.

        Args:
            url: The URL to validate.

        Returns:
            A Tuple containing a boolean indicating validity and a message.

        """
        try:
            regex = re.compile(r"https?://[^\s/$.?#].[^\s]*")
            if regex.match(url):
                return True, "URL is valid."
            return False, "Invalid URL format."
        except Exception as e:
            return False, f"Error validating URL: {str(e)}"


    def validate_license(self, license: str) -> Tuple[bool, str]:
        """
        Validates a license identifier against the SPDX License List.

        Args:
            license: The license identifier to validate.
        
        Returns:
            A Tuple containing a boolean indicating validity and a message.
        """
        try:
            with open("main/licences.json") as json_file:
                licenses_list = json.load(json_file)

            licenses_list = licenses_list["licenses"]

            # Normalize the input license to lowercase
            normalized_license = license.lower()

            # Check if the normalized license matches any licenseId in the list
            for license_info in licenses_list:
                if license_info.get("licenseId", "").lower() == normalized_license:
                    return True, "License is valid."
            return False, "Invalid License: licence must be from the SPDX License List"
        except Exception as e:
            return False, f"Error validating license: {str(e)}"


    def validate_date(self, date: str, attribute_name: str) -> Tuple[bool, str]:
        """
        Ensure the date is in the format YYYY-MM-DD.

        Args:
            date: The date string to validate.
            attribute_name: The name of the attribute being validated.
               
        Returns:
            A Tuple containing a boolean indicating validity and a message.
        """
        try:
            datetime.strptime(date, "%Y-%m-%d")
            return True, "Date is valid."
        except ValueError:
            return False, f"{attribute_name} must be in the format YYYY-MM-DD."

    def validate_language(self, in_language: str) -> Tuple[bool, str]:
        """
        Ensure the language(s) are valid by converting names to ISO codes and validating.

        Args:
            in_language: The language(s) string to validate.
               
        Returns:
            A Tuple containing a boolean indicating validity and a message.
        """
        try:
            # Split the input into multiple languages if it's a comma-separated string
            languages = [lang.strip() for lang in in_language.split(",")]

            # Validate each language
            invalid_languages = []
            valid_languages = []
            for lang in languages:
                try:
                    # First, try to validate as an ISO code using langcodes.Language.get
                    lang_obj = langcodes.Language.get(lang)
                    if lang_obj.is_valid():
                        valid_languages.append(lang_obj.language)  # Add the normalized ISO code
                        continue  # Skip to the next language if this is valid

                except (langcodes.tag_parser.LanguageTagError, ValueError):
                    # If langcodes.Language.get fails, fall back to langcodes.Language.find
                    try:
                        lang_obj = langcodes.Language.find(lang)
                        if lang_obj:
                            valid_languages.append(lang)  # Add the normalized ISO code

                    except ValueError:
                        # If langcodes.Language.find also fails, treat the language as invalid
                        invalid_languages.append(lang)

            if invalid_languages:
                return False, f"The following languages are invalid: {', '.join(invalid_languages)}"

            if sorted(valid_languages) == sorted(languages):
                return True, "All languages are valid."
    
            return False, f"Language(s) '{in_language}' are not valid ISO language codes or names."

        except Exception as e:
            return False, f"Language(s) '{in_language}' are not valid ISO language codes or names. Error: {str(e)}"

    def validate_cite_as(self, cite_as: str) -> Tuple[bool, str]:
        """
        Ensure the citation is in valid BibTeX format.

        Args:
            cite_as: The BibTeX citation string to validate.
                
        Returns:
            A Tuple containing a boolean indicating validity and a message.
        """
        try:
            parser = BibTexParser()
            parser.customization = homogenize_latex_encoding  # Optional: Normalize LaTeX encoding
            bib_database = bibtexparser.loads(cite_as, parser=parser)
            if bib_database.entries:
                return True, "Citation is valid."
            return False, "Citation must be in valid BibTeX format."
        except Exception as e:
            return False, f"Citation must be in valid BibTeX format. Error: {str(e)}"

    def check_non_empty_string(self, value: str, attribute_name: str) -> Tuple[bool, str]:
        """
        Ensure the value is a non-empty string.

        Args:
            value: The string value to validate.
            attribute_name: The name of the attribute being validated.
        
        Returns:
            A Tuple containing a boolean indicating validity and a message.
        """
        if isinstance(value, str) and value.strip():
            return True, "Value is a non-empty string."
        return False, f"{attribute_name} must be a non-empty string."

    def validate_comma_separated_strings(self, value: str, attribute_name: str) -> Tuple[bool, str]:
        """
        Validate that a string is a comma-separated list of non-empty values.

        Args:
            value: The string to validate.
            attribute_name: The name of the attribute being validated.

        Returns:
            A Tuple containing a boolean indicating validity and a message.
        """
        try:
            if isinstance(value, str):
                # Split the string by commas and strip whitespace
                items = [item.strip() for item in value.split(",")]
                # Check if all items are non-empty
                if all(items):
                    return True, f"{attribute_name} is valid."
                return False, f"{attribute_name} must be a comma-separated list of non-empty values."
            return False, f"{attribute_name} must be a string."
        except Exception as e:
            return False, f"Error validating {attribute_name}: {str(e)}"
        
    def validate_all_attributes(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Check if all required attributes are valid.

        Args:
            metadata: A Dictionary containing metadata attributes to validate.
                
        Returns:
            A Dictionary of errors where keys are attribute names and values are error messages.
        """
        try:
            errors = {}

            # Validate URL
            if "url" in metadata:
                valid, message = self.validate_url(metadata["url"])
                if not valid:
                    errors["url"] = message

            # Validate license
            if "license" in metadata:
                valid, message = self.validate_license(metadata["license"])
                if not valid:
                    errors["license"] = message

            # Validate dates
            for date_attribute in ["date_modified", "date_created", "date_published"]:
                if date_attribute in metadata:
                    valid, message = self.validate_date(metadata[date_attribute], date_attribute.replace("_", " ").capitalize())
                    if not valid:
                        errors[date_attribute] = message

            # Validate language
            if "in_language" in metadata:
                valid, message = self.validate_language(metadata["in_language"])
                if not valid:
                    errors["in_language"] = message

            # Validate citation
            if "cite_as" in metadata:
                valid, message = self.validate_cite_as(metadata["cite_as"])
                if not valid:
                    errors["cite_as"] = message
            
            # Validate comma-separated attributes
            for attribute in ["creators", "keywords", "task", "modality"]:
                if attribute in metadata:
                    valid, message = self.validate_comma_separated_strings(metadata[attribute], attribute.capitalize())
                    if not valid:
                        errors[attribute] = message

            # Validate non-empty string attributes
            for attribute in ["name", "description", "publisher", "version", "url", "license", "date_modified", "date_created", "date_published", "cite_as"]:
                if attribute in metadata and attribute not in errors:  # Skip if there's already an error
                    valid, message = self.check_non_empty_string(metadata[attribute], attribute.capitalize())
                    if not valid:
                        errors[attribute] = message

            return errors
        except Exception as e:
            return {"error": f"An error occurred during validation: {str(e)}"}
