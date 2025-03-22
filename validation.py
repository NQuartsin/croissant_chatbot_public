# validation.py
import re
from datetime import datetime
from constants import LICENSE_OPTIONS
from langcodes import Language
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import homogenize_latex_encoding

def validate_year(year):
    """Ensure the year is a four-digit number within a reasonable range."""
    current_year = datetime.now().year
    return 1900 <= int(year) <= current_year

def validate_url(url):
    """Ensure the URL is valid."""
    regex = re.compile(r"https?://[^\s/$.?#].[^\s]*")
    return bool(regex.match(url))

def validate_license(license_choice):
    """Ensure the license is in the predefined list."""
    return license_choice in LICENSE_OPTIONS

def validate_non_empty_string(value, field_name):
    """Ensure the value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        return f"{field_name} must be a non-empty string."
    return None

def validate_keywords(keywords):
    """Ensure keywords are a comma-separated list of non-empty strings."""
    if not isinstance(keywords, str):
        return "Keywords must be a comma-separated string."
    if not all(keyword.strip() for keyword in keywords.split(",")):
        return "Each keyword must be a non-empty string."
    return None

def validate_date(date, field_name):
    """Ensure the date is in the format YYYY-MM-DD."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"{field_name} must be in the format YYYY-MM-DD."
    return None

def validate_language(language):
    """Ensure the language is valid by converting names to ISO codes and validating."""
    try:
        # Attempt to get the language object from the input
        lang = Language.find(language)  # This handles both names (e.g., "English") and codes (e.g., "en")
        if not lang.is_valid():
            return f"Language '{language}' is not a valid ISO language code or name."
        
        # Convert to ISO language code and return None if valid
        iso_code = lang.language
        return None  # No error, the language is valid
    except Exception:
        return f"Language '{language}' is not a valid ISO language code or name."

def validate_bibtex(cite_as):
    """Ensure the citation is in valid BibTeX format."""
    try:
        parser = BibTexParser()
        parser.customization = homogenize_latex_encoding  # Optional: Normalize LaTeX encoding
        bib_database = bibtexparser.loads(cite_as, parser=parser)
        if not bib_database.entries:
            return "Citation must be in valid BibTeX format."
    except Exception as e:
        return f"Citation must be in valid BibTeX format. Error: {str(e)}"
    return None

def validate_metadata(metadata):
    """Check if all required fields are valid before finalizing metadata."""
    errors = []

    # Validate year
    if "year" in metadata:
        try:
            year = int(metadata["year"])  # Convert to integer for validation
            if not validate_year(year):
                errors.append("Invalid year format. It should be between 1900 and the current year.")
        except ValueError:
            errors.append("Year must be a valid number.")

    # Validate URL
    if "url" in metadata and not validate_url(metadata["url"]):
        errors.append("Invalid URL format.")

    # Validate license
    if "license" in metadata and not validate_license(metadata["license"]):
        errors.append("Invalid license choice.")

    # Validate non-empty string fields
    for field in ["name", "author", "title", "description", "publisher", "version"]:
        if field in metadata:
            error = validate_non_empty_string(metadata[field], field.capitalize())
            if error:
                errors.append(error)

    # Validate keywords
    if "keywords" in metadata:
        error = validate_keywords(metadata["keywords"])
        if error:
            errors.append(error)

    # Validate dates
    for date_field in ["date_modified", "date_created", "date_published"]:
        if date_field in metadata:
            error = validate_date(metadata[date_field], date_field.replace("_", " ").capitalize())
            if error:
                errors.append(error)

    # Validate language
    if "language" in metadata:
        error = validate_language(metadata["language"])
        if error:
            errors.append(error)

    # Validate task and modality
    for field in ["task", "modality"]:
        if field in metadata:
            error = validate_non_empty_string(metadata[field], field.capitalize())
            if error:
                errors.append(error)

    # Validate citation
    if "cite_as" in metadata:
        error = validate_bibtex(metadata["cite_as"])
        if error:
            errors.append(error)

    return errors
