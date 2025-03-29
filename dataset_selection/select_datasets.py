from datetime import datetime
import pandas as pd 
from huggingface_hub import HfApi

api = HfApi() # Hugging Face API
# Extract top 100 liked datasers
top_100_liked_datasets = list(api.list_datasets(sort="likes", direction=-1, limit=100))
dataset_info = []

try:
    for dataset in top_100_liked_datasets:
        # Extract dataset information
        dataset_id = dataset.id 
        link = f"https://huggingface.co/datasets/{dataset_id}"
        likes = dataset.likes if dataset.likes is not None else 0
        downloads = dataset.downloads if dataset.downloads is not None else 0
        tags = dataset.tags
        tasks = []  # Store all task categories
        modality = []  # Store all modalities
        for tag in tags:
            if tag.startswith("task_categories:"):
                tasks.append(tag.split(":", 1)[1])
            elif tag.startswith("modality:"):
                modality.append(tag.split(":", 1)[1])
        tasks = ", ".join(tasks) if tasks else "Unknown"
        modality = ", ".join(modality) if modality else "Unknown"

        # Fetch "Last Modified" date (only YYYY-MM-DD)
        last_modified = dataset.last_modified.strftime("%Y-%m-%d") if dataset.last_modified else "Unknown"

        # Store today's date (only YYYY-MM-DD)
        last_accessed = datetime.now().strftime("%Y-%m-%d")

        # Add dataset information to list
        dataset_info.append([dataset_id, link, likes, downloads, tasks, modality, last_modified, last_accessed])

    # Convert the dataset information to a DataFrame
    df = pd.DataFrame(dataset_info, columns=["Name", "Link", "Likes", "Downloads", "Dataset Tasks", "Modality", "Last Modified", "Date Last Accessed"])

    # Save DataFrame to CSV
    df.to_csv("dataset_selection/huggingface_datasets.csv", index=False)
    # confirmation message
    print("Dataset information saved to dataset_selection/huggingface_datasets.csv")

except Exception as e:
    print(f"An unexpected error occured")

