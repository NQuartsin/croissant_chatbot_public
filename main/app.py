# app.py
import gradio as gr
from constants import METADATA_ATTRIBUTES
import os
import tempfile
from croissant_chatbot_manager import CroissantChatbotManager


""" 
    Gradio UI functions
    All emojis used come from: https://emojidb.org/gradio-emojis
"""
def create_chatbot_ui():
    """Create the chatbot UI."""
    return gr.Chatbot(type="messages")


def create_prompt_input(chatbot_instance, chatbot_ui):
    """Create the prompt input box."""
    prompt = gr.Textbox(max_lines=1, label="Chat Message", value="Type a greeting to start the chatbot...")
    prompt.submit(chatbot_instance.handle_user_input, [prompt], [chatbot_ui])
    prompt.submit(lambda: "", None, [prompt])  # Clear the input box after submission
    return prompt


def create_metadata_attributes_dropdown(chatbot_instance):
    """Create a dropdown for metadata attributes."""
    with gr.Row():
        dropdown = gr.Dropdown(
            choices=list(METADATA_ATTRIBUTES.keys()),  
            label="Select Metadata Attribute",
            multiselect=False,
            value=None,  # No default value
            filterable=True,  # Allow user to search for attributes
        )

        def handle_dropdown_change(attribute):
            # Ignore if the dropdown value is None
            if attribute is None:
                return gr.update(value=None), chatbot_instance.history

            # Handle the selected attribute
            updated_history = chatbot_instance.handle_selected_attribute(attribute)
            # Reset the dropdown value to None after selection
            return gr.update(value=None), updated_history

        dropdown.change(
            handle_dropdown_change,
            inputs=[dropdown],
            outputs=[dropdown, chatbot_ui]
        )
        return dropdown

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

        see_instructions_btn = gr.Button("📃 See Instructions", scale=1, elem_classes="control-btn")
        see_instructions_btn.click(lambda: display_instructions_wrapper(chatbot_instance), inputs=[], outputs=[chatbot_ui])

        refresh_btn = gr.Button("🔄 Refresh Chat", scale=1, elem_classes="control-btn")
        refresh_btn.click(lambda: chatbot_instance.reset_chat(), inputs=[], outputs=[chatbot_ui])

def create_download_metadata_button(chatbot_instance):
    """Create a button for downloading the metadata file."""
    with gr.Row():
        download_btn = gr.Button("💾 Download Metadata File", scale=1, elem_classes="control-btn")
        download_section = gr.Group(visible=False)  # Create a hidden section
        
        with download_section:
            output_file = gr.File(label="Metadata File", interactive=False)  # File component for download

        def save_and_show():
            # Save the metadata to the annotations folder
            filepath, filename = chatbot_instance.metadata_manager.save_metadata_to_file(chatbot_instance.metadata_manager.final_metadata)

            # Copy the file to the system's temporary directory
            temp_dir = tempfile.gettempdir()
            temp_filepath = os.path.join(temp_dir, filename)  

            # Copy the file to the temporary directory
            with open(filepath, "r") as src, open(temp_filepath, "w") as dest:
                dest.write(src.read())

            # Return the temporary file path for Gradio to serve
            return gr.update(visible=True), temp_filepath


        download_btn.click(
            save_and_show,
            inputs=[],
            outputs=[download_section, output_file]
        )

"""
    Main Gradio UI
"""
with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink
, secondary_hue=gr.themes.colors.cyan, neutral_hue=gr.themes.colors.indigo), css="""
    .gr-button { 
        border: 2px solid black; 
        margin: 5px; 
        padding: 8px 12px;
    }
    .control-btn { 
        background-color: var(--secondary-600) !important; /* Use primary hue */
        border: 1px solid var(--secondary-700);
        color: white;
    }
""") as demo:

    gr.Markdown("# Croissant Metadata Creator")

    chatbot_instance = CroissantChatbotManager()

    # Chatbot UI
    chatbot_ui = create_chatbot_ui()

    # Prompt Input
    prompt = create_prompt_input(chatbot_instance, chatbot_ui)

    with gr.Tab("Show Metadata Dropdown"):
        create_metadata_attributes_dropdown(chatbot_instance)

    with gr.Tab("Show Control Buttons"):
        # Control Buttons
        create_control_buttons(chatbot_instance, chatbot_ui)

        # Download Metadata Button
        create_download_metadata_button(chatbot_instance)



if __name__ == "__main__":
    demo.launch() 