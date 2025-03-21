import gradio as gr
import requests
from datetime import datetime
import json  
import mlcroissant as mlc  # Import mlcroissant for creating Croissant format
import hashlib
import pandas as pd
from validation import validate_metadata
from constants import LICENSE_OPTIONS
from metadata_suggestions import suggest_metadata

# Metadata storage
metadata = {}
waiting_for_greeting = True  
pending_field = None  # Keeps track of which field the user is answering


metadata_fields = {
    "name": "What is the name of your dataset?",
    "author": "Who is the author of your dataset?",
    "year": "What year was your dataset published?",
    "title": "What is the title of the dataset or associated paper?",
    "description": "Please provide a brief description of your dataset.",
    "license": "Please select a license for your dataset:",
    "url": "Please provide the URL to your dataset or repository.",
}

file_object_fields = {
    "id": "unique identifier",
    "name": "name",
    "description": "description",
    "content_url": "content URL",
    "encoding_format": "encoding format",
    "sha256": "SHA-256 checksum",
}

def detect_fields(dataset_url):
    """Automatically detects field names from the dataset file."""
    response = requests.get(dataset_url)
    
    if response.status_code != 200:
        return []

    content = response.text

    # Try JSON first (assumes JSONL format)
    try:
        first_line = content.split("\n")[0].strip()
        json_data = json.loads(first_line)
        return list(json_data.keys())  # Extract JSON keys as fields
    except json.JSONDecodeError:
        pass  # If it fails, try CSV

    # Try CSV
    try:
        df = pd.read_csv(dataset_url, nrows=1)  # Read only the first row
        return df.columns.tolist()  # Extract column headers
    except Exception:
        pass  # If it fails, return empty list

    return []

# Function to download dataset and calculate checksum (MD5 or SHA256)
def download_and_get_checksum(url, checksum_type="sha256"):
    # Download dataset
    response = requests.get(url)
    if response.status_code != 200:
        return None  # Handle failed download case
    
    file_content = response.content
    if checksum_type == "sha256":
        checksum = hashlib.sha256(file_content).hexdigest()
    else:  # default to MD5
        checksum = hashlib.md5(file_content).hexdigest()
    
    return checksum

# Fetch dataset metadata from Hugging Face
def find_dataset_info(dataset_id):
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    response = requests.get(url)

    if response.status_code == 200:
        dataset = response.json()
        
        metadata["name"] = dataset.get("id", dataset_id)
        metadata["author"] = dataset.get("author", "N/A")
        metadata["year"] = datetime.strptime(dataset.get("lastModified", "N/A"), "%Y-%m-%dT%H:%M:%S.%fZ").year if dataset.get("lastModified") else "XXXX"
        metadata["title"] = dataset.get("title", "N/A")
        metadata["description"] = dataset.get("description", "N/A")
        metadata["license"] = dataset.get("license", "N/A")
        metadata["url"] = f"https://huggingface.co/datasets/{dataset_id}"

        # Fetch checksum
        checksum = download_and_get_checksum(metadata["url"], checksum_type="sha256")
        metadata["checksum"] = checksum if checksum else "Checksum not available"

        # Auto-detect fields
        detected_fields = detect_fields(metadata["url"])
        metadata["fields"] = detected_fields if detected_fields else ["text", "label"]  # Default fields if none found

        return metadata
    return None


# Generate BibTeX
def generate_bibtex(metadata):
    return f"@misc{{{metadata.get('author', 'unknown').split(' ')[0]}{metadata.get('year', 'XXXX')}," \
           f" author = {{{metadata.get('author', 'Unknown Author')}}}," \
           f" title = {{{metadata.get('title', 'Untitled Dataset')}}}," \
           f" year = {{{metadata.get('year', 'XXXX')}}}," \
           f" url = {{{metadata.get('url', 'N/A')}}} }}"

def fetch_huggingface_files(dataset_id, checksum):
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        dataset_info = response.json()
        files = dataset_info.get("siblings", [])
        
        file_objects = []
        for file in files:
            file_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{file['rfilename']}"
            file_objects.append(mlc.FileObject(
                id=file["rfilename"],
                name=file["rfilename"],
                content_size=file["size"],
                content_url=file_url,
                encoding_format="application/jsonlines" if file["rfilename"].endswith(".jsonl") else "text/csv", 
                sha256=checksum,

            ))
        return file_objects
    return []

# Finalize metadata in Croissant format
def finalise_metadata(history):
    history.append({"role": "assistant", "content": "Thanks for sharing the information! Here is your dataset metadata:"})

    # Convert into Croissant metadata format (using mlcroissant)
  

    # Ensure checksum exists, else fallback to default "Checksum not available"
    checksum_value = metadata.get("checksum", "Checksum not available")

    distribution = fetch_huggingface_files(metadata.get("name", "unknown-dataset"), checksum_value)


    # Assuming `distribution` contains a list of FileObjects

    # Create Fields linked to this DataSource
    fields = []
    file_object_id = distribution[0].id if distribution else "unknown-file"

    for field_name in metadata.get("fields", ["text", "label"]):
        fields.append(
            mlc.Field(
                id=f"{field_name}-field",
                name=field_name,
                description=f"{field_name} data field",
                data_types=[mlc.DataType.TEXT],
                source=mlc.Source(
                    extract=mlc.Extract(column=field_name),
                    file_object=file_object_id,  # Ensure it's a valid file object
                ),
            )
        )


    croissant_metadata = mlc.Metadata(
        name=metadata.get("name"),
        description=metadata.get("description"),
        cite_as=generate_bibtex(metadata),
        license=metadata.get("license"),
        url=metadata.get("url"),
        distribution=distribution,
        record_sets=[mlc.RecordSet(id="jsonl", name="jsonl", fields=fields)]
    )

    history.append({"role": "assistant", "content": f"```json\n{json.dumps(croissant_metadata.to_json(), indent=2)}\n```"})
    return history



# Handle user input through chat
def handle_user_input(prompt, history):
    global waiting_for_greeting, pending_field

    if not history:
        history = []

    history.append({"role": "user", "content": prompt})

    if waiting_for_greeting:
        history.append({"role": "assistant", "content": "Hello! I'm the Croissant Metadata Assistant. Click a button to provide metadata fields!"})
        waiting_for_greeting = False
        return history

    
    if prompt.lower() == "complete":
        errors = validate_metadata(metadata)
        if errors:
            history.append({"role": "assistant", "content": "Some metadata fields are invalid:\n" + "\n".join(errors)})
            return history
        elif all(field in metadata for field in metadata_fields):  # Ensure all fields are present
            print("Finalizing metadata...")
            return finalise_metadata(history)
        else:
            history.append({"role": "assistant", "content": "Some metadata fields are still missing. Please fill them before finalizing."})
            return history
        
    

    # Handle pending field input
    if pending_field:
        # Save the user-provided value
        metadata[pending_field] = prompt.strip()
        history.append({"role": "assistant", "content": f"Saved `{pending_field}` as: {prompt.strip()}."})

        # If the user just provided the dataset name, fetch metadata
        if pending_field == "name":
            dataset_info = find_dataset_info(prompt.strip())
            if dataset_info:
                history.append({"role": "assistant", "content": "I fetched the following metadata for your dataset:"})
                history.append({"role": "assistant", "content": f"```json\n{json.dumps(dataset_info, indent=2)}\n```"})

        # Reset pending field after processing
        pending_field = None

    # # Suggest missing fields
    # for field, question in metadata_fields.items():
    #     if not metadata.get(field):  # Check if the field is missing
    #         pending_field = field
    #         suggested_value = suggest_metadata(field, metadata.get("name", ""), metadata.get("description", ""))
    #         history.append({"role": "assistant", "content": f"The field `{field}` is missing. Suggested value: {suggested_value}. Please confirm or modify."})
    #         return history
    #     # Reset pending field after processing
    #     pending_field = None

    if all(field in metadata for field in metadata_fields) and "All metadata fields have been filled" not in [msg["content"] for msg in history]:
        history.append({"role": "assistant", "content": "All metadata fields have been filled. Click any field to update its value or type 'Complete' to finalize the metadata."})

    return history


# Handle button clicks (sets the pending field)
def ask_for_field(field, history):
    global pending_field

    if not history:
        history = []

    pending_field = field

    # Check if the field is missing or has "N/A"
    if not metadata.get(field) or metadata.get(field) == "N/A":
        # Suggest a value for the field
        suggested_value = suggest_metadata(field, metadata.get("name", ""), metadata.get("description", ""))
        history.append({"role": "assistant", "content": f"The field `{field}` is missing or has no valid value. Suggested value: {suggested_value}. Please confirm or modify."})
    else:
        # Prompt the user to update the existing value
        current_value = metadata.get(field)
        history.append({"role": "assistant", "content": f"The field `{field}` already has a value: `{current_value}`. You can update it if needed."})

    return history

# Handle license selection
def select_license(license_choice, history):
    global pending_field
    pending_field = "license"
    return handle_user_input(license_choice, history)
  

# Handle year selection
def select_year(year, history):
    global pending_field
    pending_field = "year"
    return handle_user_input(year, history)

# Undo last message
def undo_last_message(history):
    if history:
        history.pop()
    return history

# Reset chat
def reset_chat():
    global metadata, waiting_for_greeting, pending_field
    metadata = {}
    waiting_for_greeting = True
    pending_field = None
    return []  

# Generate years dynamically
current_year = datetime.now().year
YEAR_OPTIONS = [str(y) for y in range(1900, current_year + 1)]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Croissant Metadata Creator")
    
    chatbot = gr.Chatbot(label="Metadata Agent", type="messages", height=500)
    
    prompt = gr.Textbox(max_lines=1, label="Chat Message")
    
    # Chat message input
    prompt.submit(handle_user_input, [prompt, chatbot], chatbot)
    prompt.submit(lambda: "", None, [prompt])

    # Buttons for metadata fields with smaller size
    with gr.Row():
        for field in metadata_fields.keys():
            btn = gr.Button(field, elem_id=field, scale=0.5)  # Added scale argument to shrink the button size
            btn.click(ask_for_field, [gr.State(field), chatbot], chatbot)

    # Chat control buttons with smaller size
    with gr.Row():  
        retry_btn = gr.Button("🔄 Retry", scale=0.5)  # Scale set to 0.5 to reduce size
        undo_btn = gr.Button("↩️ Undo", scale=0.5)    # Scale set to 0.5 to reduce size
        refresh_btn = gr.Button("🔄 Refresh", scale=0.5)  # Scale set to 0.5 to reduce size

    retry_btn.click(lambda h: h, chatbot, chatbot)  
    undo_btn.click(undo_last_message, chatbot, chatbot)  
    refresh_btn.click(reset_chat, [], chatbot)  

    # Dropdowns for year & license selection
    year_dropdown = gr.Dropdown(choices=YEAR_OPTIONS, label="Select Publication Year", interactive=True)
    license_dropdown = gr.Dropdown(choices=LICENSE_OPTIONS, label="Select License", interactive=True)

    year_dropdown.change(select_year, [year_dropdown, chatbot], chatbot)
    license_dropdown.change(select_license, [license_dropdown, chatbot], chatbot)

# Run app
if __name__ == "__main__":
    demo.launch()
