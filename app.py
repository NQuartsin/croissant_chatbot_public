from huggingface_hub import InferenceClient
import gradio as gr
import os
import requests
from datetime import datetime
import json  


# Initialize Hugging Face Inference Client
client = InferenceClient(
    token=os.getenv("HUGGING_FACE_API_KEY")  # Ensure your Hugging Face API key is set
)

HF_datasets_url = "https://huggingface.co/api/datasets"

# Define metadata fields and prompts
metadata_fields = [
    {"field": "name", "prompt": "What is the name of your dataset?"},
    {"field": "author", "prompt": "Who is the author of your dataset? (Provide full name)"},
    {"field": "year", "prompt": "What year was your dataset published?"},
    {"field": "title", "prompt": "What is the title of the dataset or associated paper?"},
    {"field": "description", "prompt": "Please provide a brief description of your dataset."},
    {"field": "license", "prompt": "Please select a license for your dataset:"},
    {"field": "url", "prompt": "Please provide the URL to your dataset or repository."},
]

# Metadata storage
metadata = {}
current_field_idx = 0  # Start with the first metadata question
waiting_for_greeting = True  # Flag to track greeting stage

LICENSE_OPTIONS = [
    "Public Domain", "CC-0", "ODC-PDDL", "CC-BY", "ODC-BY", "CC-BY-SA", 
    "ODC-ODbL", "CC-BY-NC", "CC-BY-NC-SA", "CC BY-ND", "CC BY-NC-ND", 
    "CDLA-Permissive-1.0", "CDLA-Sharing-1.0", "MIT", "GPL", 
    "Apache License, Version 2.0", "BSD-3-Clause", "Other"
]

def find_dataset_info(dataset_id):
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    response = requests.get(url)

    if response.status_code == 200:
        dataset = response.json()
        
        metadata["name"] = dataset.get("id", "Unknown")
        metadata["author"] = dataset.get("author", "Unknown Author")
        metadata["year"] = datetime.strptime(dataset.get("lastModified", "XXXX"), "%Y-%m-%dT%H:%M:%S.%fZ").year if dataset.get("lastModified") else "XXXX"
        metadata["title"] = dataset.get("title", "Untitled Dataset")
        metadata["description"] = dataset.get("description", "No description available.")
        metadata["license"] = dataset.get("license", "Other")
        metadata["url"] = f"https://huggingface.co/datasets/{dataset_id}"
        
        return metadata
    else:
        print(f"Failed to fetch dataset info. Status code: {response.status_code}")
        return None



def generate_bibtex(metadata):
    """Generates a single-line BibTeX citation from metadata fields."""
    bibtex_entry = f"@misc{{{metadata.get('author', 'unknown').split(' ')[0]}{metadata.get('year', 'XXXX')}," \
                   f" author = {{{metadata.get('author', 'Unknown Author')}}}," \
                   f" title = {{{metadata.get('title', 'Untitled Dataset')}}}," \
                   f" year = {{{metadata.get('year', 'XXXX')}}}," \
                   f" url = {{{metadata.get('url', 'N/A')}}} }}"
    return bibtex_entry

def finalise_metadata(metadata, history):
    if not history:
        history = []
    history.append({"role": "assistant", "content": "Thanks for sharing the information! Here is your dataset metadata:"})
    metadata_json = {
        "@context": {"@language": "en", "@vocab": "https://schema.org/"},
        "@type": "sc:Dataset",
        "name": metadata.get("name"),
        "citeAs": generate_bibtex(metadata),
        "description": metadata.get("description"),
        "license": metadata.get("license"),
        "url": metadata.get("url"),
    }
    history.append({"role": "assistant", "content": f"```json\n{json.dumps(metadata_json, indent=2)}\n```"})
    return history

def respond(prompt: str, history):
    global current_field_idx, waiting_for_greeting

    if not history:
        history = []

    # Handle greeting stage
    if waiting_for_greeting:
        history.append({"role": "user", "content": prompt})  # Store user's initial message
        history.append({"role": "assistant", "content": "Hello! I will help you create metadata for your dataset, including a BibTeX citation. Let's begin!"})
        history.append({"role": "assistant", "content": metadata_fields[current_field_idx]["prompt"]})
        waiting_for_greeting = False  # Now start metadata collection
        return history

    # Save user input to metadata
    if current_field_idx < len(metadata_fields):
        field = metadata_fields[current_field_idx]["field"]
        user_input = prompt.strip()

        # Append user input to history
        history.append({"role": "user", "content": prompt})

        # If user input is not blank, update the metadata
        if user_input:
            metadata[field] = user_input

        # If the current field is "name", call find_dataset_info
        if field == "name":
            dataset_info = find_dataset_info(user_input)
            if dataset_info:
                history.append({"role": "assistant", "content": "I have fetched the following metadata for your dataset:"})
                history.append({"role": "assistant", "content": f"```json\n{json.dumps(dataset_info, indent=2)}\n```"})
            else:
                history.append({"role": "assistant", "content": "I couldn't fetch metadata for the provided dataset name. Please continue providing the information."})

        # Move to next question or finish
        current_field_idx += 1
        if current_field_idx < len(metadata_fields):
            next_prompt = metadata_fields[current_field_idx]["prompt"]
            history.append({"role": "assistant", "content": next_prompt})
        else:
            history = finalise_metadata(metadata, history)

    return history

def select_license(license_choice, history):
    global current_field_idx

    metadata["license"] = license_choice  # Store selected license

    if not history:
        history = []

    # Append selected license as a user response
    history.append({"role": "user", "content": f"Selected License: {license_choice}"})

    # Move to next question
    current_field_idx += 1
    if current_field_idx < len(metadata_fields):
        history.append({"role": "assistant", "content": metadata_fields[current_field_idx]["prompt"]})
    else:
        history = finalise_metadata(metadata, history)
    return history  # Return updated chatbot history

# Generate years dynamically (from 1900 to the current year)
current_year = datetime.now().year
YEAR_OPTIONS = [str(y) for y in range(1900, current_year + 1)]


def reset_chat():
    """Reset global state variables and clear chat."""
    global metadata, current_field_idx, waiting_for_greeting
    metadata = {}
    current_field_idx = 0
    waiting_for_greeting = True
    return []  # Clear chatbot history

with gr.Blocks() as demo:
    gr.Markdown("# Croissant Metadata Creator")
    
    chatbot = gr.Chatbot(
        label="Metadata Agent",
        type="messages",
        avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/hugging-face_1f917.png"),
        height=500  # Make chatbot bigger
    )
    
    prompt = gr.Textbox(max_lines=1, label="Chat Message")
    with gr.Row():  # Buttons in a row
        retry_btn = gr.Button("🔄 Retry")
        undo_btn = gr.Button("↩️ Undo")
        refresh_btn = gr.Button("🔄 Refresh")  # Our custom refresh button

    # Define behavior for each button
    retry_btn.click(lambda history: history, chatbot, chatbot)  # Retry does nothing for now
    undo_btn.click(lambda history: undo_last_message(history), chatbot, chatbot)  # Removes last message
    refresh_btn.click(reset_chat, [], chatbot)  # Clears chat history

    def undo_last_message(history):
        global current_field_idx
        if history:
            history.pop()  # Remove last message
            if current_field_idx > 0:
                current_field_idx -= 1  # Move back to the previous question
        return history
    
    # Dropdown for selecting the publication year
    year_dropdown = gr.Dropdown(choices=YEAR_OPTIONS, label="Select Publication Year", interactive=True, visible=False)
    
    license_dropdown = gr.Dropdown(choices=LICENSE_OPTIONS, label="Select License", interactive=True, visible=False)

    def check_ui_visibility(history):
        """Show dropdowns when necessary."""
        if current_field_idx < len(metadata_fields):
            field = metadata_fields[current_field_idx]["field"]
            if field == "year":
                return gr.update(visible=True), gr.update(visible=False)
            elif field == "license":
                return gr.update(visible=False), gr.update(visible=True)
        
        return gr.update(visible=False), gr.update(visible=False)

    chatbot.change(check_ui_visibility, chatbot, [year_dropdown, license_dropdown])

    # Handle year selection
    def select_year(year, history):
        global current_field_idx
        metadata["year"] = year
        history.append({"role": "user", "content": f"Selected Year: {year}"})
        
        current_field_idx += 1
        if current_field_idx < len(metadata_fields):
            history.append({"role": "assistant", "content": metadata_fields[current_field_idx]["prompt"]})
        else:
            history = finalise_metadata(metadata, history)

        return history

    year_dropdown.change(select_year, [year_dropdown, chatbot], chatbot)
    license_dropdown.change(select_license, [license_dropdown, chatbot], chatbot)

    prompt.submit(respond, [prompt, chatbot], chatbot)
    prompt.submit(lambda: "", None, [prompt])

if __name__ == "__main__":
    demo.launch()