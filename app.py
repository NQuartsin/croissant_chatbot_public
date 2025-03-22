from turtle import pu
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
    "publisher": "Who is the publisher of your dataset?",
    "version": "What is the version of your dataset?",
    "keywords": "Please provide keywords for your dataset.",
    "date_modified": "When was the dataset last modified?",
    "date_created": "When was the dataset created?",
    "date_publihsed": "When was the dataset published?",
    "language": "What are the languages of the dataset?",
    "cite_as": "Please provide a citation for your dataset."
}

# Generate BibTeX
def generate_bibtex(metadata):
    return f"@misc{{{metadata.get('author', 'unknown').split(' ')[0]}{metadata.get('year', 'XXXX')}," \
           f" author = {{{metadata.get('author', 'Unknown Author')}}}," \
           f" title = {{{metadata.get('title', 'Untitled Dataset')}}}," \
           f" year = {{{metadata.get('year', 'XXXX')}}}," \
           f" url = {{{metadata.get('url', 'N/A')}}} }}"

# Fetch dataset details
def find_dataset_info(dataset_id):
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    response = requests.get(url)

    if response.status_code == 200:
        dataset = response.json()
        card_data = dataset.get("cardData", {})

        
        metadata["name"] = dataset.get("id", dataset_id)
        metadata["author"] = dataset.get("author", "N/A")
        metadata["year"] = datetime.strptime(dataset.get("lastModified", "XXXX"), "%Y-%m-%dT%H:%M:%S.%fZ").year if dataset.get("lastModified") else "N/A"
        metadata["title"] = dataset.get("title", "N/A")
        metadata["description"] = dataset.get("description", "N/A")
        metadata["license"] = card_data.get("license", "N/A")
        metadata["url"] = f"https://huggingface.co/datasets/{dataset_id}"
        metadata["publisher"] = dataset.get("author", "N/A")
        metadata["version"] = dataset.get("codebase_version", "N/A")
        metadata["keywords"] = ", ".join(card_data.get("tags", [])) if card_data.get("tags") else "N/A"
        metadata["date_modified"] = datetime.strptime(dataset.get("lastModified", "N/A"), "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d") if dataset.get("lastModified") else "N/A"
        metadata["date_created"] = datetime.strptime(dataset.get("createdAt", "N/A"), "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d") if dataset.get("createdAt") else "N/A"
        metadata["date_published"] = datetime.strptime(dataset.get("createdAt", "N/A"), "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d") if dataset.get("createdAt") else "N/A"
        metadata["language"] = ", ".join(card_data.get("languages", [])) if card_data.get("languages") else "N/A"
        
        metadata["cite_as"] = dataset.get("citation", "N/A")

        return metadata
    return None  

# Finalize metadata in Croissant format
def finalise_metadata(history):
    history.append({"role": "assistant", "content": "Thanks for sharing the information! Here is your dataset metadata:"})

    croissant_metadata = mlc.Metadata(
        name=metadata.get("name"),
        creators=metadata.get("author"),
        description=metadata.get("description"),
        license=metadata.get("license"),
        url=metadata.get("url"),
        publisher=metadata.get("publisher"),
        version=metadata.get("version"),
        keywords=metadata.get("keywords"),
        last_modified=metadata.get("date_modified"),
        date_created=metadata.get("date_created"),
        date_published=metadata.get("date_published"),
        in_language=metadata.get("language"),
        cite_as=metadata.get("cite_as"),

    )
    final_metadata = croissant_metadata.to_json()
    

    history.append({"role": "assistant", "content": f"```json\n{json.dumps(croissant_metadata.to_json(), indent=2)}\n```"})
    return history


# Handle user input through chat
def handle_user_input(prompt, history):
    global waiting_for_greeting, pending_field

    if not history:
        history = []

    history.append({"role": "user", "content": prompt})

    # Handle initial greeting
    if waiting_for_greeting:
        return handle_greeting(history)

    # Handle "complete" command
    if prompt.lower() == "complete":
        return handle_complete_command(history)

    # Handle pending field input
    if pending_field:
        history = handle_pending_field_input(prompt, history)

    # Check if all fields are filled
    if all_fields_filled():
        history.append({"role": "assistant", "content": "All metadata fields have been filled. Click any field to update its value or type 'Complete' to finalize the metadata."})

    return history


def handle_greeting(history):
    """Handle the initial greeting."""
    global waiting_for_greeting
    history.append({"role": "assistant", "content": "Hello! I'm the Croissant Metadata Assistant. Click a button to provide metadata fields!"})
    waiting_for_greeting = False
    return history


def handle_complete_command(history):
    """Handle the 'complete' command."""
    errors = validate_metadata(metadata)
    if errors:
        history.append({"role": "assistant", "content": "Some metadata fields are invalid:\n" + "\n".join(errors)})
    elif all_fields_filled():
        print("Finalizing metadata...")
        return finalise_metadata(history)
    else:
        history.append({"role": "assistant", "content": "Some metadata fields are still missing. Please fill them before finalizing."})
    return history


def handle_pending_field_input(prompt, history):
    """Handle input for a pending field."""
    global pending_field
    # Save the user-provided value
    metadata[pending_field] = prompt.strip()
    history.append({"role": "assistant", "content": f"Saved `{pending_field}` as: {prompt.strip()}."})

    # Fetch metadata if the dataset name was provided
    if pending_field == "name":
        dataset_info = find_dataset_info(prompt.strip())

        if dataset_info:
            print(f"cite_as: {metadata.get('cite_as')}")
            if metadata.get("cite_as") is None or metadata.get("cite_as") == "None" or metadata.get("cite_as") == "N/A":
                print("Generating citation...")
                metadata["cite_as"] = generate_bibtex(metadata)
            history.append({"role": "assistant", "content": "I fetched the following metadata for your dataset:"})
            history.append({"role": "assistant", "content": f"```json\n{json.dumps(metadata, indent=2)}\n```"})

    # Reset pending field after processing
    pending_field = None
    return history


def all_fields_filled():
    """Check if all metadata fields are filled."""
    return all(field in metadata for field in metadata_fields)


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

def create_chatbot_ui():
    """Create the chatbot UI."""
    return gr.Chatbot(label="Metadata Agent", type="messages", height=500)

def create_prompt_input(chatbot, validation_context):
    """Create the prompt input box."""
    prompt = gr.Textbox(max_lines=1, label="Chat Message")
    prompt.submit(handle_user_input, [prompt, chatbot], chatbot)
    prompt.submit(lambda: "", None, [prompt])  # Clear the input box after submission
    return prompt

def create_metadata_buttons(chatbot):
    """Create buttons for metadata fields."""
    buttons = []
    with gr.Row():
        for field in metadata_fields.keys():
            btn = gr.Button(field, elem_id=field, scale=0.5)  # Added scale argument to shrink the button size
            btn.click(ask_for_field, [gr.State(field), chatbot], chatbot)
            buttons.append(btn)
    return buttons

def create_control_buttons(chatbot):
    """Create control buttons (Retry, Undo, Refresh)."""
    with gr.Row():
        retry_btn = gr.Button("🔄 Retry", scale=0.5)
        undo_btn = gr.Button("↩️ Undo", scale=0.5)
        refresh_btn = gr.Button("🔄 Refresh", scale=0.5)

        retry_btn.click(lambda h: h, chatbot, chatbot)
        undo_btn.click(undo_last_message, chatbot, chatbot)
        refresh_btn.click(reset_chat, [], chatbot)

def create_dropdowns(chatbot):
    """Create dropdowns for year and license selection."""
    year_dropdown = gr.Dropdown(choices=YEAR_OPTIONS, label="Select Publication Year", interactive=True)
    license_dropdown = gr.Dropdown(choices=LICENSE_OPTIONS, label="Select License", interactive=True)

    year_dropdown.change(select_year, [year_dropdown, chatbot], chatbot)
    license_dropdown.change(select_license, [license_dropdown, chatbot], chatbot)

    return year_dropdown, license_dropdown

# Main Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Croissant Metadata Creator")

    # Chatbot UI
    chatbot = create_chatbot_ui()

    # Prompt Input
    prompt = create_prompt_input(chatbot, validation_context=None)

    # Metadata Buttons
    create_metadata_buttons(chatbot)

    # Control Buttons
    create_control_buttons(chatbot)

    # Dropdowns
    create_dropdowns(chatbot)

# Run app
if __name__ == "__main__":
    demo.launch()
