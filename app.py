# app.py
import keyword
import re
import gradio as gr
import requests
from datetime import datetime
import json  
import mlcroissant as mlc 
import pandas as pd
from validation import MetadataValidator
from constants import LICENSE_OPTIONS
from llm import suggest_metadata, ask_user_for_informal_description
import os
import unicodedata
from attribute_quality import AttributeQualityChecker
from huggingface_hub import list_datasets
from itertools import islice


class CroissantChatbot:
    def __init__(self):
        self.metadata = {}
        self.final_metadata = {}
        self.history = []
        self.waiting_for_greeting = True
        self.waiting_for_informal_description = False
        self.pending_attribute = None
        self.informal_description = ""
        self.confirmed_metadata = {}

        # Metadata attributes
        self.metadata_attributes = {
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
            "date_published": "When was the dataset published?",
            "cite_as": "Please provide a citation for your dataset.",
            "language": "What are the languages of the dataset?",
            "task": "What tasks can be performed with your dataset?",
            "modality": "What are the modalities of your dataset?"
        }

        # Generate years dynamically
        current_year = datetime.now().year
        self.YEAR_OPTIONS = [str(y) for y in range(1900, current_year + 1)]

    def append_to_history(self, message):
        """Append a message to the chat history."""
        self.history.append(message)

    def json_to_code_block(self, json_data, default_value=None):
        """Convert JSON data to a formatted code block."""
        return f"```json\n{json.dumps(json_data, indent=2, default=default_value)}\n```"

    def display_metadata(self):
        """Display the dataset metadata."""
        json_metadata = self.json_to_code_block(self.metadata)
        self.append_to_history({"role": "assistant", "content": f"Here is the current metadata for your dataset:\n{json_metadata}"})

    def handle_user_input(self, prompt):
        """Handle user input through chat."""
        if not self.history:
            self.history = []

        self.append_to_history({"role": "user", "content": prompt})
        if prompt.lower() == "start new dataset":
            self.reset_metadata_for_new_dataset()

        elif self.waiting_for_greeting:
            self.handle_greeting()

        elif self.waiting_for_informal_description:
            self.handle_informal_description_prompt(prompt)

        elif prompt.lower() == "complete":
            self.handle_complete_command()


        elif self.pending_attribute:
            self.handle_pending_attribute_input(prompt)

        elif self.is_all_attributes_filled():
            self.append_to_history({"role": "assistant", "content": "All metadata attributes have been filled. Click any attribute to update its value or type 'Complete' to finalise the metadata."})


        return self.history

    def handle_greeting(self):
        """Handle the initial greeting."""
        self.append_to_history({"role": "assistant", "content": "Hello! I'm the Croissant Metadata Assistant. Let's start creating metadata for your dataset."})
        self.append_to_history({"role": "assistant", "content": "Would you like to provide an informal description of your dataset? \nPlease type : your informal description/help (for guidance on providing an informal description)/no "})
        self.waiting_for_greeting = False
        self.waiting_for_informal_description = True
        return self.history

    def handle_informal_description_prompt(self, prompt):
        """Handle the user's response to the informal description prompt."""
        if prompt.lower() == "help":
            ask = ask_user_for_informal_description()
            self.append_to_history({"role": "assistant", "content": f"{ask}"})
        elif prompt.lower() == "no":
            self.append_to_history({"role": "assistant", "content": "Alright! Click any attribute to enter/update its value."})
            self.waiting_for_informal_description = False
        else:
            self.informal_description = prompt.strip()
            self.append_to_history({"role": "assistant", "content": f"Saved the informal description: {self.informal_description}"})
            self.append_to_history({"role": "assistant", "content": "Click any attribute to enter/update its value."})
            self.waiting_for_informal_description = False
        return self.history

    def handle_complete_command(self):
        """Handle the 'complete' command."""
        validator = MetadataValidator()
        errors = validator.validate_all_attributes(self.metadata)
        issues = AttributeQualityChecker().check_quality_of_all_attributes(self.metadata)

        if errors or issues:
            if errors:
                error_messages = "\n".join([f"{attribute}: {message}" for attribute, message in errors.items()])
                self.append_to_history({"role": "assistant", "content": f"Some metadata attributes are invalid:\n{error_messages}"})
            
            if issues:
                issue_messages = "\n".join([f"{attribute}: {message}" for attribute, message in issues.items()])
                self.append_to_history({"role": "assistant", "content": f"Some metadata attributes could be improved:\n{issue_messages}"})
            if self.confirmed_metadata:
                confirmed_attributes = ",".join(self.confirmed_metadata.keys())
                self.append_to_history({"role": "assistant", "content": f"You have confirmed the values for these attributes '{confirmed_attributes}' despite validation issues. Finalising metadata with these values."})
                # Merge confirmed_metadata into metadata before finalising
                self.metadata.update(self.confirmed_metadata)
                return self.finalise_metadata()
            else:
                self.append_to_history({"role": "assistant", "content": "Type 'confirm' to save the metadata despite these issues, or update the attributes to resolve the problems."})
        elif self.is_all_attributes_filled():
            # Merge confirmed_metadata into metadata before finalising
            return self.finalise_metadata()
        else:
            return self.finalise_metadata()
            # missing_attributes = self.list_missing_attributes()
            # json_missing_attributes = self.json_to_code_block(missing_attributes)
            # self.append_to_history({"role": "assistant", "content": f"Cannot finalise metadata. The following attributes are missing: \n{json_missing_attributes}"})
        
        return self.history

    def handle_pending_attribute_input(self, prompt):
        """Handle input for a pending attribute."""

        if prompt.lower() == "confirm" and self.pending_attribute:
            if self.pending_attribute in ["date_created", "date_modified", "date_published"]:
                self.append_to_history({
                    "role": "assistant",
                    "content": f"The `confirm` option is not available for the `{self.pending_attribute}` attribute. Please provide a valid date in the correct format (YYYY-MM-DD)."
                })
                return self.history
            # Save the attribute in confirmed_metadata
            self.confirmed_metadata[self.pending_attribute] = self.metadata[self.pending_attribute]
            self.append_to_history({"role": "assistant", "content": f"The value for `{self.pending_attribute}` has been saved as: {self.confirmed_metadata[self.pending_attribute]} despite validation issues."})
            self.pending_attribute = None
            return self.history
        
        if self.pending_attribute == "name":
            dataset_info = self.find_dataset_info(prompt.strip())
            if dataset_info:
                if not self.metadata.get("cite_as") or self.metadata["cite_as"] in ["None", ""]:
                    self.metadata["cite_as"] = self.generate_bibtex()
                self.append_to_history({"role": "assistant", "content": "I fetched the following metadata for your dataset:"})
                self.display_metadata()
                return self.history
            
        # Validate the input for the pending attribute
        self.metadata[self.pending_attribute] = prompt.strip()
        self.append_to_history({"role": "assistant", "content": f"Saved `{self.pending_attribute}` as: {prompt.strip()}."})

        # Perform validation checks
        validator = MetadataValidator()
        errors = validator.validate_all_attributes({f"{self.pending_attribute}": f"{prompt.strip()}"})
        issues = AttributeQualityChecker().check_quality_of_all_attributes({f"{self.pending_attribute}": f"{prompt.strip()}"})


        if errors or issues:
            error_messages = "\n".join([f"{attribute}: {message}" for attribute, message in errors.items()]) if errors else ""
            issue_messages = "\n".join([f"{attribute}: {message}" for attribute, message in issues.items()]) if issues else ""
            suggested_value = suggest_metadata(self.metadata, self.informal_description, self.pending_attribute)

            self.append_to_history({"role": "assistant", "content": f"There are issues with the value you provided for `{self.pending_attribute}`:\n{error_messages}\n{issue_messages}"})
            self.append_to_history({"role": "assistant", "content": f"{suggested_value}"})
            self.append_to_history({"Type 'confirm' to save this value anyway, or use one of these suggestions as the value or enter your own idea for the value. \n If the attribute is `date_created`, `date_modified`, or `date_published`, please provide a valid date in the format YYYY-MM-DD."})
        else:
            self.pending_attribute = None

        return self.history
    
    # Handle button clicks (sets the pending attribute)
    def handle_clicked_attribute(self, attribute):
        self.pending_attribute = attribute
        self.append_to_history({"role": "user", "content": f"Selected attribute: `{attribute}`."})

        # Check if the attribute is missing
        if not self.metadata.get(attribute) or self.metadata.get(attribute) == "":
            # Suggest a value for the attribute
            suggested_value = suggest_metadata(self.metadata, self.informal_description, attribute)
            self.append_to_history({"role": "assistant", "content": f"The attribute `{attribute}` is missing. \n {suggested_value}. \n\nYou can use one of these suggestions as the value or enter your own idea for the value."})
        else:
            # Prompt the user to update the existing value
            current_value = self.metadata.get(attribute)
            self.append_to_history({"role": "assistant", "content": f"The attribute `{attribute}` already has a value: `{current_value}`. You can update it if needed."})

        return self.history

    def undo_last_message(self):
        """Undo the last message."""
        if self.history:
            self.history.pop()
        return self.history

    def reset_chat(self):
        """Reset the chat."""
        self.metadata = {}
        self.history = []
        self.waiting_for_greeting = True
        self.pending_attribute = None
        self.waiting_for_informal_description = False
        self.informal_description = ""
        self.confirmed_metadata = {}
        return self.history

    def is_all_attributes_filled(self):
        """Check if all metadata attributes are filled."""
        return all(attribute in self.metadata for attribute in self.metadata_attributes)
    
    def list_missing_attributes(self):
        """List the missing metadata attributes."""
        return [attribute for attribute in self.metadata_attributes if attribute not in self.metadata]

    def generate_bibtex(self):
        """Generate BibTeX citation."""
        return f"@misc{{{self.metadata.get('author', 'unknown').split(' ')[0]}{self.metadata.get('year', 'XXXX')}," \
               f" author = {{{self.metadata.get('author', 'Unknown Author')}}}," \
               f" title = {{{self.metadata.get('title', 'Untitled Dataset')}}}," \
               f" year = {{{self.metadata.get('year', 'XXXX')}}}," \
               f" url = {{{self.metadata.get('url', '')}}} }}"

    def find_dataset_info(self, dataset_id):
        """Fetch dataset details."""

        datasets = list_datasets()
        dataset = next((d for d in datasets if d.id == dataset_id), None)
        if not dataset:
            return None
        lisence = ""
        tasks = []
        modalities = []
        languages = []
        
        for tag in dataset.tags:
            if tag.startswith("license:"):
                lisence = tag.split(":", 1)[1]
            elif tag.startswith("task_categories:"):
                tasks.append(tag.split(":", 1)[1])
            elif tag.startswith("modality:"):
                modalities.append(tag.split(":", 1)[1])
            elif tag.startswith("language:"):
                languages.append(tag.split(":", 1)[1])

        self.metadata["name"] = getattr(dataset, "id", "")
        self.metadata["author"] = getattr(dataset, "author", "")
        self.metadata["year"] = dataset.last_modified.year if getattr(dataset, "last_modified", None) else ""
        self.metadata["title"] = ""  # No title field in the dataset object
        self.metadata["description"] = getattr(dataset, "description", "")
        self.metadata["license"] = lisence if lisence else ""
        self.metadata["url"] = f"https://huggingface.co/datasets/{dataset_id}"
        self.metadata["publisher"] = getattr(dataset, "author", "")
        self.metadata["version"] = ""  # No version field in the dataset object
        self.metadata["keywords"] = "" # No easy way to fetch keywords
        self.metadata["date_modified"] = dataset.last_modified.strftime("%Y-%m-%d") if getattr(dataset, "last_modified", None) else ""
        self.metadata["date_created"] = dataset.created_at.strftime("%Y-%m-%d") if getattr(dataset, "created_at", None) else ""
        self.metadata["date_published"] = dataset.created_at.strftime("%Y-%m-%d") if getattr(dataset, "created_at", None) else ""
        self.metadata["cite_as"] = getattr(dataset, "citation", "")
        self.metadata["task"] = ", ".join(tasks) if tasks else ""
        self.metadata["modality"] = ", ".join(modalities) if modalities else ""
        self.metadata["language"] = ", ".join(languages) if languages else ""

        return self.metadata


    def finalise_metadata(self):
        """Finalise metadata in Croissant format."""
        self.append_to_history({"role": "assistant", "content": "Thanks for sharing the information! Here is your dataset metadata:"})

        croissant_metadata = mlc.Metadata(
            name=self.metadata.get("name"),
            creators=self.metadata.get("author"),
            description=self.metadata.get("description"),
            license=self.metadata.get("license"),
            url=self.metadata.get("url"),
            publisher=self.metadata.get("publisher"),
            version=self.metadata.get("version"),
            keywords=self.metadata.get("keywords"),
            date_modified=self.metadata.get("date_modified", "Unknown"),
            date_created=self.metadata.get("date_created", "Unknown"),
            date_published=self.metadata.get("date_published", "Unknown"),
            cite_as=self.metadata.get("cite_as", "Unknown"),
            in_language=self.metadata.get("language", "Unknown"),
        )

        self.final_metadata = croissant_metadata.to_json()

        self.final_metadata["task"] = self.metadata.get("task", "")
        self.final_metadata["modality"] = self.metadata.get("modality", "")


        
        # Convert metadata to JSON and display it
        display_metadata = self.json_to_code_block(self.final_metadata, self.json_serial)
        self.append_to_history({"role": "assistant", "content": f"\n{display_metadata}"})

        # Save metadata to a file
        filepath, filename = self.save_metadata_to_file(self.final_metadata)

        # Inform the user about the saved file
        self.append_to_history({"role": "assistant", "content": f"The metadata has been saved to a file: `{filename}`. \n Click the 'Download Metadata File' button below to download it. \n You can also start annotating a new dataset by typing 'Start new dataset'."})

        return self.history
    
    def json_serial(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d")
        raise TypeError(f"Type {type(obj)} not serialisable")    
    
    def remove_emojis(self, text):
        """Remove emojis from a string."""
        return ''.join(c for c in text if unicodedata.category(c) != 'So')

    def get_filename(self):
        """Get the filename for the metadata file."""
        if self.metadata.get("name"):
            # Remove emojis
            name_without_emojis = self.remove_emojis(self.metadata['name'])
            # Replace invalid filename characters with a hyphen (-)
            sanitised_name = re.sub(r'[<>:"/\\|?*]', '-', name_without_emojis)
            return f"{sanitised_name.replace(' ', '_').lower()}_metadata.json"
        return "metadata.json"
            
    def save_metadata_to_file(self, metadata):
        """Save the metadata to a file in the annotations folder."""
        directory = "annotations"  # Specify the folder where files should be saved
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the folder if it doesn't exist

        # Generate the filename
        filename = self.get_filename()
        filepath = os.path.join(directory, filename)  # Save in the annotations folder

        # Save the file
        with open(filepath, "w") as file:
            json.dump(metadata, file, indent=2, default=self.json_serial)

        return filepath, filename 
    
    def reset_metadata_for_new_dataset(self):
        """Reset metadata attributes for annotating a new dataset."""
        self.metadata = {}
        self.final_metadata = {}
        self.pending_attribute = None
        self.informal_description = ""
        self.confirmed_metadata = {}
        self.append_to_history({"role": "assistant", "content": "You can now start annotating a new dataset. Let's begin!"})
        return self.history

    
# Gradio UI functions
def create_chatbot_ui():
    """Create the chatbot UI."""
    return gr.Chatbot(label="Metadata Agent", type="messages", height=500)

def create_prompt_input(chatbot_instance, chatbot_ui):
    """Create the prompt input box."""
    prompt = gr.Textbox(max_lines=1, label="Chat Message")
    prompt.submit(chatbot_instance.handle_user_input, [prompt], [chatbot_ui])
    prompt.submit(lambda: "", None, [prompt])  # Clear the input box after submission
    return prompt



def create_attribute_buttons(chatbot_instance, chatbot_ui):
    """Create buttons for metadata attributes."""
    buttons = []
    with gr.Row():
        for attribute in chatbot_instance.metadata_attributes.keys():
            btn = gr.Button(attribute, elem_id=attribute, scale=1) 
            btn.click(
                lambda f=attribute: chatbot_instance.handle_clicked_attribute(f),
                [], 
                [chatbot_ui]
            )
            buttons.append(btn)
    return buttons

def create_control_buttons(chatbot_instance, chatbot_ui):
    """Create control buttons (Retry, Undo, Refresh)."""
    with gr.Row():
        retry_btn = gr.Button("🔄 Retry", scale=1)
        undo_btn = gr.Button("↩️ Undo", scale=1)
        refresh_btn = gr.Button("🔄 Refresh", scale=1)

        retry_btn.click(lambda h=chatbot_instance.history: h, [], [chatbot_ui])
        undo_btn.click(lambda: chatbot_instance.undo_last_message(), [], [chatbot_ui])
        refresh_btn.click(lambda: chatbot_instance.reset_chat(), [], [chatbot_ui])

def select_license(chatbot_instance, license_choice):
    chatbot_instance.pending_attribute = "license"
    return chatbot_instance.handle_user_input(license_choice)
  

# Handle year selection
def select_year(chatbot_instance, year_choice):
    chatbot_instance.pending_attribute = "year"
    return chatbot_instance.handle_user_input(year_choice)


def create_dropdowns(chatbot_instance, chatbot_ui):
    """Create dropdowns for year and license selection."""
    year_dropdown = gr.Dropdown(choices=chatbot_instance.YEAR_OPTIONS, label="Select Publication Year", interactive=True)
    license_dropdown = gr.Dropdown(choices=LICENSE_OPTIONS, label="Select License", interactive=True)

    year_dropdown.change(lambda y: select_year(chatbot_instance, y), [year_dropdown], [chatbot_ui])
    license_dropdown.change(lambda l: select_license(chatbot_instance, l), [license_dropdown], [chatbot_ui])

    return year_dropdown, license_dropdown

def display_metadata_wrapper(chatbot_instance):
    """Wrapper for the display_metadata method to work with Gradio."""
    chatbot_instance.display_metadata()
    return chatbot_instance.history

def create_display_metadata_button(chatbot_instance, chatbot_ui):
    """Create a button to display the current metadata."""
    with gr.Row():
        display_btn = gr.Button("📋 Display Metadata So Far", scale=1)
        # Use the wrapper function to return the correct output
        display_btn.click(lambda: display_metadata_wrapper(chatbot_instance), [], [chatbot_ui])


def create_download_metadata_button(chatbot_instance):
    """Create a button for downloading the metadata file."""
    with gr.Row():
        download_btn = gr.Button("📥 Download Metadata File", scale=1)
        download_section = gr.Group(visible=False)  # Create a hidden section
        
        with download_section:
            output_file = gr.File(label="Metadata File", interactive=False)  # File component for download
        
        def save_and_show():
            # Save the metadata to the annotations folder
            filepath, filename = chatbot_instance.save_metadata_to_file(chatbot_instance.final_metadata)
            
            # Copy the file to the current working directory for download
            download_path = os.path.join(os.getcwd(), filename)
            if os.path.exists(download_path):
                os.remove(download_path)  # Remove any existing file with the same name
            os.symlink(filepath, download_path)  # Create a symbolic link to the file
            
            # Return the download path for Gradio to serve the file
            return gr.update(visible=True), download_path


        download_btn.click(
            save_and_show,
            inputs=[],
            outputs=[download_section, output_file]
        )

# Main Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Croissant Metadata Creator")

    chatbot_instance = CroissantChatbot()

    # Chatbot UI
    chatbot_ui = create_chatbot_ui()

    # Prompt Input
    prompt = create_prompt_input(chatbot_instance, chatbot_ui)

    # Metadata Buttons
    create_attribute_buttons(chatbot_instance, chatbot_ui)

    # Display Metadata Button
    create_display_metadata_button(chatbot_instance, chatbot_ui)

    # Control Buttons
    create_control_buttons(chatbot_instance, chatbot_ui)

    # Dropdowns
    create_dropdowns(chatbot_instance, chatbot_ui)
    
    # Download Metadata Button
    create_download_metadata_button(chatbot_instance)



if __name__ == "__main__":
    demo.launch() # share=True to share the link