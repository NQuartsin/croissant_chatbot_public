import os
import json
import pandas as pd
from huggingface_hub import HfApi, dataset_info

# Define the Hugging Face API
api = HfApi()

# Define the output folder
OUTPUT_FOLDER = "hf_metadata"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def fetch_hf_metadata(dataset_id):
    """Fetch metadata for a Hugging Face dataset."""
    try:
        info = dataset_info(dataset_id)
        
        metadata = {
            "name": info.id,
            "creator": ", ".join(info.cardData.get("creator", ["Unknown"])),
            "description": info.cardData.get("description", "No description available."),
            "license": info.cardData.get("license", "Unknown"),
            "url": f"https://huggingface.co/datasets/{info.id}",
            "publisher": info.cardData.get("publisher", "Unknown"),
            "version": info.cardData.get("version", "Unknown"),
            "keywords": ", ".join(info.cardData.get("keywords", ["Unknown"])),
            "date_modified": info.last_modified.strftime("%Y-%m-%d") if info.last_modified else "Unknown",
            "date_created": info.cardData.get("dataset_info", {}).get("config_name", "Unknown"),  # No direct creation date in HF
            "date_published": info.cardData.get("date_published", "Unknown"),
            "cite_as": info.cardData.get("citation", "Unknown"),
            "in_language": ", ".join(info.cardData.get("languages", ["Unknown"])),
            "task": ", ".join(info.cardData.get("task_categories", ["Unknown"])),
            "modality": ", ".join(info.cardData.get("modality", ["Unknown"]))
        }

        return metadata

    except Exception as e:
        print(f"Error fetching metadata for {dataset_id}: {e}")
        return None

def save_metadata(metadata):
    """Save metadata to a JSON file."""
    dataset_name = metadata["name"].replace("/", "_")
    file_path = os.path.join(OUTPUT_FOLDER, f"{dataset_name}.json")
    
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
    
    first_dataset = dataset_names[0]  # Take only the first dataset
    print(f"Processing first dataset: {first_dataset}")

    metadata = fetch_hf_metadata(first_dataset)
    if metadata:
        save_metadata(metadata)

# Run the script with your CSV file
csv_file_path = "dataset_selection/huggingface_datasets.csv"
process_csv(csv_file_path)
