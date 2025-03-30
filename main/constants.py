# constants.py

"""Contains constant values for the project"""

# a Dictionary of metadata attributes and their descriptions
METADATA_ATTRIBUTES = {
    "name": "the name of the dataset (non-empty string).",
    "author": "the author(s) of the dataset/publication that describes the dataset (comma-separated non-empty strings) (e.g. John Doe, Jane Doe).",
    "year": "the publication year of the dataset (non-empty string) (YYYY).",
    "title": "the title of the dataset/publication that describes the dataset (non-empty string).",
    "description": "a description of the dataset (non-empty string) (2+ descriptive sentences).",
    "license": "the license of the dataset (non-empty string) (expected to be a lisence from the SPDX License List).",
    "url": "the URL of the dataset (non-empty string) (valid URL format).",
    "publisher": "the publisher of the dataset (non-emptystring).",
    "version": "the version of the dataset (non-empty string).",
    "keywords": "the keywords of the dataset (comma-separated non-empty strings) (at least 3) (e.g. computer vision, medical, biology).",
    "date_modified": "the date the dataset was last modified (non-empty string) (YYYY-MM-DD).",
    "date_created": "the date the dataset was created (non-empty string) (YYYY-MM-DD).",
    "date_published":  "the date the dataset was published (non-empty string) (YYYY-MM-DD).",
    "cite_as": "the citation for the dataset (non-empty string) (BibTeX format).",
    "language": "the language(s) of the dataset (comma-separated non-empty strings) (ISO 639-1 codes/Language names) (e.g. en, French).",
    "task": "the task(s) associated with the dataset (comma-separated non-empty strings) (e.g. text-generation, text2text-generation, question-answering).",
    "modality": "the modality(s) of the dataset (comma-separated non-empty strings) (e.g. tabular, text, timeseries, video)."
}
