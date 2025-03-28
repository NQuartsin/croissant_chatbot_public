# app.py
import re
import gradio as gr
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
        self.waiting_for_HF_name = False

        # Metadata attributes
        self.metadata_attributes = {
            "name": "The name of the dataset (string).",
            "author": "The author of the dataset (string).",
            "year": "The publication year of the dataset (YYYY).",
            "title": "The title of the dataset/publication that describes the dataset (string).",
            "description": "A description of the dataset (string) (2+ sentences).",
            "license": "The license of the dataset (string) (one of the valid options).",
            "url": "The URL of the dataset (string) (valid URL format).",
            "publisher": "The publisher of the dataset (string).",
            "version": "The version of the dataset (string).",
            "keywords": "The keywords of the dataset (comma-separated string) (at least 3).",
            "date_modified": "The date the dataset was last modified (YYYY-MM-DD).",
            "date_created": "The date the dataset was created (YYYY-MM-DD).",
            "date_published":  "The date the dataset was published (YYYY-MM-DD).",
            "cite_as": "The citation for the dataset (string) (BibTeX format).",
            "language": "The language(s) of the dataset (comma-separated string) (ISO 639-1 codes/Language names).",
            "task": "The task(s) associated with the dataset (comma-separated string).",
            "modality": "The modality(s) of the dataset (comma-separated string)."
        }

        # Generate years dynamically
        current_year = datetime.now().year

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

    def display_chatbot_instructions(self):
        """Get the chatbot instructions."""
        attributes_list = "\n".join([f"- {attribute}: {description}" for attribute, description in self.metadata_attributes.items()])
        instructions = f"""
            This is a very simple chatbot that helps you create Croissant metadata for your dataset.
            When you start the chat, the chatbot will guide you through the process of entering metadata attributes.\n

            At the start of the chat, you can provide an informal description of your dataset.
            This will help the chatbot suggest more relevant metadata attributes for you. \n
            You can also provide the name of your dataset if it is from Hugging Face Datasets.
            The chatbot will try to fetch some metadata for you based on the dataset name.
            You can only do these things at the start of the chat, if you want to do them later you must click the "Refresh" button to start over.\n

            To enter metadata attributes, click on the attribute name and enter the value in the chat box.
            When you click on the attribute name, the chatbot will provide guidance on what to enter using the attribute name as a prompt.
            You must enter a value immediately after clicking the attribute name.\n
            After entering the value of an attribute, the chatbot will validate the value and suggest improvements if needed.
            If the value is valid, the chatbot will save the value and you can proceed to the next attribute.
            If the value is invalid, the chatbot will ask you to provide a new value.
            If you want to confirm the value despite validation issues, type 'confirm' in the chat box.
            If you want to update an attribute, click on the attribute name again and enter the new value.\n

            The attributes you need to provide values for are:
            {attributes_list}\n

            If you want to see the current metadata stored in the chatbot, click the 'Display Metadata So Far' button.
            If you think you have entered all the metadata attributes, type 'complete' in the chat box to finalise the metadata.
            After finalising the metadata, the chatbot will display the metadata in JSON format and save it to a file, if all the fields are valid or confirmed.
            Once the metadata is finalised, you can download the metadata file by clicking the 'Download Metadata File' button, 
            and the file will appear next to the button for download when you click the blue file size text.\n
            If you want to start annotating a new dataset, type 'start new dataset' in the chat box.\n

            This chatbot is not smart. It will not be able to process any input that is not a value for a metadata attribute
            or one of the expected commands when prompted for: 'confirm', 'complete', 'start new dataset', 'help', 'no'.
            When asked a question you must immediately answer it before the chatbot can proceed.
            Please do not press any button or enter anything unexpected when the chatbot is waiting for a response, otherwise it may not work as expected.\n
            The validation checks are basic and may not cover all possible issues with the metadata.
            They are meant to guide you in providing the best metadata possible.
            If you encounter any issues, you can refresh the chat by clicking the 'Refresh Chat' button.\n

            If you want to see these instructions again at any time, press the 'See Instructions' button.\n
        """
        formatted_instructions = f"```text\n{instructions}\n```"
        self.append_to_history({"role": "assistant", "content": formatted_instructions})

    def promt_to_click_attribute(self):
        """Prompt the user to click on an attribute to enter/update its value."""
        self.append_to_history({
            "role": "assistant",
            "content": """Click any attribute to enter/update its value. 
                If the value is empty, the chatbot will provide guidance on what to enter using the informal description.
                If the value is already filled, you can update it if needed.
                If you want to see the current metadata stored in the chatbot, click the 'Display Metadata So Far' button.
                If you think you have entered all the metadata attributes, type **complete** in the chat box to finalise the metadata."""
        })

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

        elif self.waiting_for_HF_name:
            self.handle_HF_name(prompt)

        elif prompt.lower() == "complete":
            self.handle_complete_command()

        elif self.pending_attribute:
            self.handle_pending_attribute_input(prompt)

        elif self.is_all_attributes_filled():
            self.append_to_history({
                "role": "assistant", 
                "content": """All metadata attributes have been filled."""
            })
            self.promt_to_click_attribute()

        return self.history

    def handle_greeting(self):
        """Handle the initial greeting."""
        self.append_to_history({"role": "assistant", "content": "Hello! I'm the Croissant Metadata Assistant."})
        self.display_chatbot_instructions()
        self.append_to_history({
            "role": "assistant",
            "content": """Would you like to provide an informal description of your dataset? 
                **Please type one of these options:**
                - &lt;your informal description&gt;
                - help (will provide guidance on providing an informal description) 
                - no"""
        })
        self.waiting_for_greeting = False
        self.waiting_for_informal_description = True
        return self.history

    def handle_informal_description_prompt(self, prompt):
        """Handle the user's response to the informal description prompt."""
        if prompt.lower() == "help":
            ask = ask_user_for_informal_description()
            self.append_to_history({"role": "assistant", "content": f"{ask}"})
            return self.history
        elif prompt.lower() == "no":
            self.waiting_for_informal_description = False
        else:
            self.informal_description = prompt.strip()
            self.append_to_history({"role": "assistant", "content": f"Saved the 'informal description': {self.informal_description}"})
            self.waiting_for_informal_description = False
        self.append_to_history({
            "role": "assistant", 
            "content": """Is your dataset from Hugging Face Datasets? 
                **Please type one of these options:**
                -  &lt;dataset name&gt;
                - no"""
        })
        self.waiting_for_HF_name = True
        return self.history
    
    def handle_HF_name(self, prompt):
        """Handle the user's response to the Hugging Face dataset name prompt."""
        if prompt.lower() == "no":
            self.prompt_user_to_click_attribute()
            self.waiting_for_HF_name = False
        else:
            dataset_info = self.find_dataset_info(prompt.strip())
            if dataset_info:
                self.append_to_history({"role": "assistant", "content": "I fetched the following metadata for your dataset:"})
                self.display_metadata()
                self.prompt_user_to_click_attribute()
            else:
                self.append_to_history({"role": "assistant", "content": "I couldn't find any information for the provided dataset name."})
                self.prompt_user_to_click_attribute()
            self.waiting_for_HF_name = False
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
                self.metadata.update(self.confirmed_metadata) # Merge confirmed_metadata into metadata before finalising
                return self.finalise_metadata()
            else:
                self.append_to_history({"role": "assistant", "content": "Please update the attributes to resolve the problems (for each attribute u can confirm invalid values)."})
        elif self.is_all_attributes_filled():
            return self.finalise_metadata()
        else:
            missing_attributes = self.list_missing_attributes()
            json_missing_attributes = self.json_to_code_block(missing_attributes)
            self.append_to_history({"role": "assistant", "content": f"Cannot finalise metadata. The following attributes are missing: \n{json_missing_attributes}"})
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
            self.append_to_history({"role": "assistant", "content":"Type **confirm** to save this value anyway, or use one of these suggestions as the value or enter your own idea for the value. \n If the attribute is `date_created`, `date_modified`, or `date_published`, please provide a valid date in the format YYYY-MM-DD."})
        else:
            self.pending_attribute = None

        # Regenerate citation if title, author, year, or URL is updated (but citation isnt provided by user or HF dataset)
        
        return self.history
    
    # Handle button clicks (sets the pending attribute)
    def handle_clicked_attribute(self, attribute):
        self.pending_attribute = attribute
        self.append_to_history({"role": "user", "content": f"Selected attribute: `{attribute}`."})

        attribute_description = self.metadata_attributes.get(attribute, "")
        # Check if the attribute is missing
        if not self.metadata.get(attribute) or self.metadata.get(attribute) == "":
            # Suggest a value for the attribute
            suggested_value = suggest_metadata(self.metadata, self.informal_description, attribute)
            self.append_to_history({
                "role": "assistant",
                "content": f"""The attribute `{attribute}` is missing. This is: {attribute_description}
                {suggested_value}

                You can use one of these suggestions as the value or enter your own value."""
            })
        else:
            # Prompt the user to update the existing value
            current_value = self.metadata.get(attribute)
            self.append_to_history({"role": "assistant", "content": f"The attribute `{attribute}` already has a value: `{current_value}`. You can update it if needed."})

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
        self.waiting_for_HF_name = False

        return self.history

    def is_all_attributes_filled(self):
        """Check if all metadata attributes are filled."""
        return all(attribute in self.metadata for attribute in self.metadata_attributes)
    
    def list_missing_attributes(self):
        """List the missing metadata attributes."""
        return [attribute for attribute in self.metadata_attributes if attribute not in self.metadata]

    def generate_bibtex(self):
        """Generate BibTeX citation dynamically based on available metadata."""
        # Start the BibTeX entry with the type and ID
        bibtex = f"@misc{self.metadata.get('author', 'unknown').split(' ')[0]}{self.metadata.get('year', 'XXXX')},\n"

        # Dynamically add fields if they exist
        if "author" in self.metadata and self.metadata["author"]:
            bibtex += f"  author = {{{self.metadata['author']}}},\n"
        if "title" in self.metadata and self.metadata["title"]:
            bibtex += f"  title = {{{self.metadata['title']}}},\n"
        if "year" in self.metadata and self.metadata["year"]:
            bibtex += f"  year = {{{self.metadata['year']}}},\n"
        if "url" in self.metadata and self.metadata["url"]:
            bibtex += f"  url = {{{self.metadata['url']}}},\n"

        # Remove the trailing comma and newline, then close the BibTeX entry
        if bibtex.endswith(",\n"):
            bibtex = bibtex[:-2] + "\n"
        bibtex += "}"

        return bibtex

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
        
        # Generate citation if not provided
        if not self.metadata.get("cite_as"):
            self.metadata["cite_as"] = self.generate_bibtex()

        try:
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
            self.append_to_history({
                "role": "assistant",
                "content": f"""The metadata has been saved to a file: `{filename}`. 
                    Click the 'Download Metadata File' button below to download it. 
                    You can also start annotating a new dataset by typing **start new dataset**."""
                  })
        except Exception as e:
            self.append_to_history({"role": "assistant", "content": f"An error occurred while finalising the metadata: {str(e)}"})
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
        self.waiting_for_HF_name = False
        self.append_to_history({"role": "assistant", "content": "You can now start annotating a new dataset. Let's begin!"})
        return self.history

    
# Gradio UI functions
def create_chatbot_ui():
    """Create the chatbot UI."""
    return gr.Chatbot(type="messages", height=500)

def create_prompt_input(chatbot_instance, chatbot_ui):
    """Create the prompt input box."""
    prompt = gr.Textbox(max_lines=1, label="Chat Message")
    prompt.submit(chatbot_instance.handle_user_input, [prompt], [chatbot_ui])
    prompt.submit(lambda: "", None, [prompt])  # Clear the input box after submission
    return prompt


def create_attribute_buttons(chatbot_instance, chatbot_ui):
    """Create buttons for metadata attributes with styling."""
    buttons = []
    with gr.Row():
        for attribute in chatbot_instance.metadata_attributes.keys():
            btn = gr.Button(attribute, elem_id=attribute, scale=1, elem_classes="metadata-btn")
            btn.click(
                lambda f=attribute: chatbot_instance.handle_clicked_attribute(f),
                [], 
                [chatbot_ui]
            )
            buttons.append(btn)
    return buttons

def select_license(chatbot_instance, license_choice):
    chatbot_instance.pending_attribute = "license"
    return chatbot_instance.handle_user_input(license_choice)


def create_license_dropdown(chatbot_instance, chatbot_ui):
    """Create dropdowns for license selection."""
    license_dropdown = gr.Dropdown(choices=LICENSE_OPTIONS, label="Select License", interactive=True)

    license_dropdown.change(lambda l: select_license(chatbot_instance, l), [license_dropdown], [chatbot_ui])

    return license_dropdown 

def create_metadata_group(chatbot_instance, chatbot_ui):
    """Create a grouped section for metadata attribute buttons and license dropdown."""
    with gr.Group():  # Use a box to visually group the components
        gr.Markdown("### Metadata Attributes and License Selection")  # Add a title for the group

        # Metadata Attribute Buttons
        create_attribute_buttons(chatbot_instance, chatbot_ui)

        # License Dropdown
        create_license_dropdown(chatbot_instance, chatbot_ui)

def display_metadata_wrapper(chatbot_instance):
    """Wrapper for the display_metadata method to work with Gradio."""
    chatbot_instance.display_metadata()
    return chatbot_instance.history

def display_instructions_wrapper(chatbot_instance):
    """Wrapper for the display_chatbot_instructions method to work with Gradio."""
    chatbot_instance.display_chatbot_instructions()
    return chatbot_instance.history


def create_control_buttons(chatbot_instance, chatbot_ui):
    """Create styled control buttons."""
    with gr.Row():
        display_btn = gr.Button("📋 Display Metadata So Far", scale=1, elem_classes="control-btn")
        display_btn.click(lambda: display_metadata_wrapper(chatbot_instance), inputs=[], outputs=[chatbot_ui])

        see_instructions_btn = gr.Button("📖 See Instructions", scale=1, elem_classes="control-btn")
        see_instructions_btn.click(lambda: display_instructions_wrapper(chatbot_instance), inputs=[], outputs=[chatbot_ui])

        refresh_btn = gr.Button("🔄 Refresh Chat", scale=1, elem_classes="control-btn")
        refresh_btn.click(lambda: chatbot_instance.reset_chat(), inputs=[], outputs=[chatbot_ui])

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
with gr.Blocks(css="""
    .gr-button { 
        border: 2px solid black; 
        margin: 5px; 
        padding: 8px 12px;
    }
    .metadata-btn { 
        background-color: #4CAF50; 
        color: white; 
        border-radius: 12px; /* Rounded corners */
    }
    .control-btn { 
        background-color: #008CBA; 
        color: white;
    }
""") as demo:

    gr.Markdown("# Croissant Metadata Creator")

    chatbot_instance = CroissantChatbot()

    # Chatbot UI
    chatbot_ui = create_chatbot_ui()

    # Prompt Input
    prompt = create_prompt_input(chatbot_instance, chatbot_ui)


    create_metadata_group(chatbot_instance, chatbot_ui)


    # Control Buttons
    create_control_buttons(chatbot_instance, chatbot_ui)

    # Download Metadata Button
    create_download_metadata_button(chatbot_instance)




if __name__ == "__main__":
    demo.launch() 