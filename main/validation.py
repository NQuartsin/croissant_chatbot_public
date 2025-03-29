# validation.py
import re
from datetime import datetime
import langcodes 
import bibtexparser 
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding
import json
from typing import Tuple, Dict


class MetadataValidator():
    """Validate metadata attributes and values for a dataset entry."""

    def validate_year(self, year: int) -> Tuple[bool, str]:
        """
        Ensure the year is a four-digit number within a reasonable range.

        Args:
            year: The year to validate.

        Returns:
            A tuple containing a boolean indicating validity and a message.
        """
        current_year = datetime.now().year
        if 1900 <= int(year) <= current_year:
            return True, "Year is valid."
        return False, "Year must be between 1900 and the current year."

    def validate_url(self, url: str) -> Tuple[bool, str]:
        """
        Ensure the URL is valid.

        Args:
            url: The URL to validate.

        Returns:
            A tuple containing a boolean indicating validity and a message.

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
            A tuple containing a boolean indicating validity and a message.
        """
        try:
            with open("licences.json") as json_file:
                licenses_list = json.load(json_file)

            licenses_list = licenses_list["licenses"]

            # Normalize the input license to lowercase
            normalized_license = license.lower()

            # Check if the normalized license matches any licenseId in the list
            for license_info in licenses_list:
                if license_info.get("licenseId", "").lower() == normalized_license:
                    return True, "License is valid."
            return False, "Invalid License: liecence must be from the SPDX License List"
        except Exception as e:
            return False, f"Error validating license: {str(e)}"

    def validate_non_empty_string(self, value: str, attribute_name: str) -> Tuple[bool, str]:
        """
        Ensure the value is a non-empty string.

        Args:
            value: The string value to validate.
            attribute_name: The name of the attribute being validated.
        
        Returns:
            A tuple containing a boolean indicating validity and a message.
        """
        if isinstance(value, str) and value.strip():
            return True, "Value is a non-empty string."
        return False, f"{attribute_name} must be a non-empty string."

    def validate_keywords(self, keywords: str) -> Tuple[bool, str]:
        """
        Ensure keywords are a comma-separated list of non-empty strings.

        Args:
            keywords: The keywords string to validate.
                
        Returns:
            A tuple containing a boolean indicating validity and a message.
        """
        try:
            if isinstance(keywords, str) and all(keyword.strip() for keyword in keywords.split(",")):
                return True, "Keywords are valid."
            return False, "Keywords must be a comma-separated list of non-empty strings."
        except Exception as e:
            return False, f"Error validating keywords: {str(e)}"

    def validate_date(self, date: str, attribute_name: str) -> Tuple[bool, str]:
        """
        Ensure the date is in the format YYYY-MM-DD.

        Args:
            date: The date string to validate.
            attribute_name: The name of the attribute being validated.
               
        Returns:
            A tuple containing a boolean indicating validity and a message.
        """
        try:
            datetime.strptime(date, "%Y-%m-%d")
            return True, "Date is valid."
        except ValueError:
            return False, f"{attribute_name} must be in the format YYYY-MM-DD."

    def validate_language(self, language: str) -> Tuple[bool, str]:
        """
        Ensure the language(s) are valid by converting names to ISO codes and validating.

        Args:
            language: The language(s) string to validate.
               
        Returns:
            A tuple containing a boolean indicating validity and a message.
        """
        try:
            # Split the input into multiple languages if it's a comma-separated string
            languages = [lang.strip() for lang in language.split(",")]

            # Validate each language
            invalid_languages = []
            for lang in languages:
                try:
                    # Try creating a valid language object
                    lang_obj = langcodes.Language.make(lang)
                    # If it doesn't return a proper language code, mark as invalid
                    if not lang_obj.language:
                        invalid_languages.append(lang)
                except ValueError:
                    invalid_languages.append(lang)

            # If there are invalid languages, return an error
            if invalid_languages:
                return False, f"The following languages are invalid: {', '.join(invalid_languages)}"

            # If all languages are valid, return success
            return True, "All languages are valid."
        except Exception:
            return False, f"Language(s) '{language}' are not valid ISO language codes or names."

    def validate_bibtex(self, cite_as: str) -> Tuple[bool, str]:
        """
        Ensure the citation is in valid BibTeX format.

        Args:
            cite_as: The BibTeX citation string to validate.
                
        Returns:
            A tuple containing a boolean indicating validity and a message.
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


    def validate_all_attributes(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Check if all required attributes are valid.

        Args:
            metadata: A dictionary containing metadata attributes to validate.
                
        Returns:
            A dictionary of errors where keys are attribute names and values are error messages.
        """
        try:
            errors = {}

            # Validate year
            if "year" in metadata:
                try:
                    year = int(metadata["year"])  # Convert to integer for validation
                    valid, message = self.validate_year(year)
                    if not valid:
                        errors["year"] = message
                except ValueError:
                    errors["year"] = "Year must be a valid number."
                except Exception as e:
                    errors["year"] = f"Error validating year: {str(e)}"

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

            # Validate non-empty string attributes
            for attribute in ["name", "author", "title", "description", "publisher", "version"]:
                if attribute in metadata:
                    valid, message = self.validate_non_empty_string(metadata[attribute], attribute.capitalize())
                    if not valid:
                        errors[attribute] = message

            # Validate keywords
            if "keywords" in metadata:
                valid, message = self.validate_keywords(metadata["keywords"])
                if not valid:
                    errors["keywords"] = message

            # Validate dates
            for date_attribute in ["date_modified", "date_created", "date_published"]:
                if date_attribute in metadata:
                    valid, message = self.validate_date(metadata[date_attribute], date_attribute.replace("_", " ").capitalize())
                    if not valid:
                        errors[date_attribute] = message

            # Validate language
            if "language" in metadata:
                valid, message = self.validate_language(metadata["language"])
                if not valid:
                    errors["language"] = message

            # Validate task and modality
            for attribute in ["task", "modality"]:
                if attribute in metadata:
                    valid, message = self.validate_non_empty_string(metadata[attribute], attribute.capitalize())
                    if not valid:
                        errors[attribute] = message

            # Validate citation
            if "cite_as" in metadata:
                valid, message = self.validate_bibtex(metadata["cite_as"])
                if not valid:
                    errors["cite_as"] = message
            return errors
        except Exception as e:
            return {"error": f"An error occurred during validation: {str(e)}"}
