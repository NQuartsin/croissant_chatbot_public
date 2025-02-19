import requests
from datetime import datetime
import pandas as pd
import json  

# Hugging Face API URL for datasets
url = "https://huggingface.co/api/datasets"

# Fetch dataset metadata
response = requests.get(url)

if response.status_code == 200:
    datasets = response.json()
    
    dataset_info = []

    for dataset in datasets[:100]:  # Select first 100
        dataset_id = dataset.get("id", "Unknown")
        link = f"https://huggingface.co/datasets/{dataset_id}"
        likes = dataset.get("likes", 0)
        downloads = dataset.get("downloads", 0)  

        all_tags = dataset.get("tags", [])

        tasks = []  # Store all task categories
        modality = []  # Store all modalities
        
        # Loop through the tags to find all task categories and modalities
        for tag in all_tags:
            if tag.startswith("task_categories:"):
                tasks.append(tag.split(":", 1)[1])  # Add task category value to the list
            elif tag.startswith("modality:"):
                modality.append(tag.split(":", 1)[1])  # Add modality value to the list

        # If tasks or modality lists are not empty, join them into strings
        tasks = ", ".join(tasks) if tasks else "Unknown"
        modality = ", ".join(modality) if modality else "Unknown"
        
        # Fetch "Last Modified" date (only YYYY-MM-DD)
        last_modified = dataset.get("lastModified", "Unknown")
        if last_modified != "Unknown":
            last_modified = datetime.strptime(last_modified, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")

        # Store today's date (only YYYY-MM-DD)
        last_accessed = datetime.now().strftime("%Y-%m-%d")  

        # Add dataset information to list
        dataset_info.append([dataset_id, link, likes, downloads, tasks, modality, last_modified, last_accessed])

    # Convert the dataset information to a DataFrame
    df = pd.DataFrame(dataset_info, columns=["Name", "Link", "Likes", "Downloads", "Dataset Tasks", "Modality", "Last Modified", "Date Last Accessed"])

    # Save DataFrame to CSV
    df.to_csv("huggingface_datasets.csv", index=False)
    print("Dataset information saved to huggingface_datasets.csv")

else:
    print(f"Failed to fetch datasets. Status code: {response.status_code}")
