import os
import json
import pandas as pd
from huggingface_hub import HfApi
from typing import Tuple, Dict

# Define the Hugging Face API
api = HfApi()

# Define the output folder
OUTPUT_FOLDER = "hf_metadata"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def fetch_hf_metadata(dataset_id_to_find: str) -> Tuple[Dict[str, str], bool]:
    """
    Fetch Hugging Face dataset metadata.

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

        hf_data = {} # dictionary to store data
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

        hf_data["name"] = getattr(found_dataset, "id", "")
        hf_data["creators"] = getattr(found_dataset, "author", "")
        hf_data["description"] = getattr(found_dataset, "description", "")
        hf_data["license"] = lisence if lisence else ""
        hf_data["url"] = "" # No URL information available on the datacard
        hf_data["publisher"] = "" # No publisher information available on the datacard
        hf_data["version"] = ""  # No way to fetch version
        hf_data["keywords"] = "" # No way to fetch keywords
        hf_data["date_modified"] = "" # No date modified information available on the datacard
        hf_data["date_created"] = "" # No date created information available on the datacard
        hf_data["date_published"] = "" # No date published information available on the datacard
        hf_data["cite_as"] = getattr(found_dataset, "citation", "")
        hf_data["task"] = ", ".join(tasks) if tasks else ""
        hf_data["modality"] = ", ".join(modalities) if modalities else ""
        hf_data["in_language"] = ", ".join(languages) if languages else ""

        return hf_data, True

    except Exception as e:
        return {'error': str(e)}, False


def save_metadata(metadata):
    """Save metadata to a JSON file."""
    dataset_name = metadata["name"].replace("/", "_")
    file_path = os.path.join(OUTPUT_FOLDER, f"{dataset_name}_hf.json")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {file_path}")

# def process_csv(csv_path):
#     """Read CSV, fetch metadata, and save JSON."""
#     df = pd.read_csv(csv_path)
    
#     if "Name" not in df.columns:
#         print("Error: 'Name' column not found in CSV.")
#         return
    
#     dataset_names = df["Name"].dropna().unique()
    
#     for dataset_name in dataset_names:
#         metadata = fetch_hf_metadata(dataset_name)
#         if metadata:
#             save_metadata(metadata)

def process_csv(csv_path):
    """Read CSV, fetch metadata for only the first dataset, and save JSON."""
    df = pd.read_csv(csv_path)
    
    if "Name" not in df.columns:
        print("Error: 'Name' column not found in CSV.")
        return
    
    dataset_names = df["Name"].dropna().unique()

    if len(dataset_names) == 0:
        print("Error: No dataset names found in CSV.")
        return
    
    first_dataset = dataset_names[99]  # Take only the first dataset
    print(f"Processing dataset: {first_dataset}")

    metadata, sucess = fetch_hf_metadata(first_dataset)
    if sucess:
        save_metadata(metadata)

# Run the script with your CSV file
csv_file_path = "dataset_selection/huggingface_datasets.csv"
process_csv(csv_file_path)
