# validation.py
import re
from datetime import datetime
import langcodes 
import bibtexparser 
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding
import json
from constants import LICENSE_OPTIONS as LICENSES


class MetadataValidator():
    """Validate metadata attributes and values for a dataset entry."""

    def validate_year(self, year):
        """Ensure the year is a four-digit number within a reasonable range."""
        current_year = datetime.now().year
        if 1900 <= int(year) <= current_year:
            return True, "Year is valid."
        return False, "Year must be between 1900 and the current year."

    def validate_url(self, url):
        """Ensure the URL is valid."""
        regex = re.compile(r"https?://[^\s/$.?#].[^\s]*")
        if regex.match(url):
            return True, "URL is valid."
        return False, "Invalid URL format."

    def validate_license(self, license):
        """Validates a license identifier against the SPDX License List."""
        try:
            with open("licences.json") as json_file:
                licesnces_list = json.load(json_file)

            licesnces_list = licesnces_list["licenses"]

            for license_info in licesnces_list:
                if license_info.get("licenseId", "") == license or license in LICENSES:
                    return True, "License is valid."
            return False, "Invalid License"
        except Exception as e:
            return False, f"Error validating license: {str(e)}"

    def validate_non_empty_string(self, value, attribute_name):
        """Ensure the value is a non-empty string."""
        if isinstance(value, str) and value.strip():
            return True, "Value is a non-empty string."
        return False, f"{attribute_name} must be a non-empty string."

    def validate_keywords(self, keywords):
        """Ensure keywords are a comma-separated list of non-empty strings."""
        if isinstance(keywords, str) and all(keyword.strip() for keyword in keywords.split(",")):
            return True, "Keywords are valid."
        return False, "Keywords must be a comma-separated list of non-empty strings."

    def validate_date(self, date, attribute_name):
        """Ensure the date is in the format YYYY-MM-DD."""
        try:
            datetime.strptime(date, "%Y-%m-%d")
            return True, "Date is valid."
        except ValueError:
            return False, f"{attribute_name} must be in the format YYYY-MM-DD."

    def validate_language(self, language):
        """Ensure the language(s) are valid by converting names to ISO codes and validating."""
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

    def validate_bibtex(self, cite_as):
        """Ensure the citation is in valid BibTeX format."""
        try:
            parser = BibTexParser()
            parser.customization = homogenize_latex_encoding  # Optional: Normalize LaTeX encoding
            bib_database = bibtexparser.loads(cite_as, parser=parser)
            if bib_database.entries:
                return True, "Citation is valid."
            return False, "Citation must be in valid BibTeX format."
        except Exception as e:
            return False, f"Citation must be in valid BibTeX format. Error: {str(e)}"

    def validate_english_words(self, value, attribute_name):
        """Ensure the value contains only English words and no special characters or punctuation."""
        if not isinstance(value, str):
            return False, f"{attribute_name} must be a string."

        # Check for non-English words or punctuation

        # TODO Fix: title: Title contains invalid words or characters: (2025) 
        words = value.split()
        invalid_words = [word for word in words if not word.isalpha()]
        if invalid_words:
            return False, f"{attribute_name} contains invalid words or characters: {', '.join(invalid_words)}"

        return True, f"{attribute_name} contains only English words."

    def validate_all_attributes(self, metadata):
        """Check if all required attributes are valid."""
        errors = {}
        # Validate attributes for English words
        for attribute in ["name", "author", "title", "description", "publisher"]:
            if attribute in metadata:
                valid, message = self.validate_english_words(metadata[attribute], attribute.capitalize())
                if not valid:
                    errors[attribute] = message

        # Validate year
        if "year" in metadata:
            try:
                year = int(metadata["year"])  # Convert to integer for validation
                valid, message = self.validate_year(year)
                if not valid:
                    errors["year"] = message
            except ValueError:
                errors["year"] = "Year must be a valid number."

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
