# validation.py
import re
from datetime import datetime
from constants import LICENSE_OPTIONS  

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

def validate_metadata(metadata):
    """Check if all required fields are valid before finalizing metadata."""
    errors = []

    if "year" in metadata:
        try:
            year = int(metadata["year"])  # Convert to integer for validation
            if not validate_year(year):
                errors.append("Invalid year format. It should be between 1900 and the current year.")
        except ValueError:
            errors.append("Year must be a valid number.")

    if "url" in metadata and not validate_url(metadata["url"]):
        errors.append("Invalid URL format.")

    if "license" in metadata and not validate_license(metadata["license"]):
        errors.append("Invalid license choice.")

    return errors
