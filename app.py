import gradio as gr
import requests
from datetime import datetime
import json  

# Metadata storage
metadata = {}
waiting_for_greeting = True  
pending_field = None  # Keeps track of which field the user is answering

LICENSE_OPTIONS = [
    "Public Domain", "CC-0", "ODC-PDDL", "CC-BY", "ODC-BY", "CC-BY-SA", 
    "ODC-ODbL", "CC-BY-NC", "CC-BY-NC-SA", "CC BY-ND", "CC BY-NC-ND", 
    "CDLA-Permissive-1.0", "CDLA-Sharing-1.0", "MIT", "GPL", 
    "Apache License, Version 2.0", "BSD-3-Clause", "Other"
]

metadata_fields = {
    "name": "What is the name of your dataset?",
    "author": "Who is the author of your dataset?",
    "year": "What year was your dataset published?",
    "title": "What is the title of the dataset or associated paper?",
    "description": "Please provide a brief description of your dataset.",
    "license": "Please select a license for your dataset:",
    "url": "Please provide the URL to your dataset or repository.",
}

# Fetch dataset details
def find_dataset_info(dataset_id):
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    response = requests.get(url)

    if response.status_code == 200:
        dataset = response.json()
        
        metadata["name"] = dataset.get("id", dataset_id)
        metadata["author"] = dataset.get("author", "Unknown Author")
        metadata["year"] = datetime.strptime(dataset.get("lastModified", "XXXX"), "%Y-%m-%dT%H:%M:%S.%fZ").year if dataset.get("lastModified") else "XXXX"
        metadata["title"] = dataset.get("title", "Untitled Dataset")
        metadata["description"] = dataset.get("description", "No description available.")
        metadata["license"] = dataset.get("license", "Other")
        metadata["url"] = f"https://huggingface.co/datasets/{dataset_id}"
        
        return metadata
    return None  

# Generate BibTeX
def generate_bibtex(metadata):
    return f"@misc{{{metadata.get('author', 'unknown').split(' ')[0]}{metadata.get('year', 'XXXX')}," \
           f" author = {{{metadata.get('author', 'Unknown Author')}}}," \
           f" title = {{{metadata.get('title', 'Untitled Dataset')}}}," \
           f" year = {{{metadata.get('year', 'XXXX')}}}," \
           f" url = {{{metadata.get('url', 'N/A')}}} }}"

# Finalize metadata
def finalise_metadata(history):
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
        if all(field in metadata for field in metadata_fields):  # Ensure all fields are present
            print("Finalizing metadata...")
            return finalise_metadata(history)
        else:
            history.append({"role": "assistant", "content": "Some metadata fields are still missing. Please fill them before finalizing."})
            return history

    if pending_field:
        metadata[pending_field] = prompt.strip()
        history.append({"role": "assistant", "content": f"Saved `{pending_field}` as: {prompt.strip()}."})

        # If the user just provided the dataset name, fetch metadata
        if pending_field == "name":
            dataset_info = find_dataset_info(prompt.strip())
            if dataset_info:
                history.append({"role": "assistant", "content": "I fetched the following metadata for your dataset:"})
                history.append({"role": "assistant", "content": f"```json\n{json.dumps(dataset_info, indent=2)}\n```"})
        pending_field = None  # Reset after processing

    if all(field in metadata for field in metadata_fields) and "All metadata fields have been filled" not in [msg["content"] for msg in history]:
        history.append({"role": "assistant", "content": "All metadata fields have been filled. Click any field to update its value or type 'Complete' to finalize the metadata."})

    return history


# Handle button clicks (sets the pending field)
def ask_for_field(field, history):
    global pending_field

    if not history:
        history = []

    pending_field = field
    history.append({"role": "assistant", "content": metadata_fields[field]})

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
