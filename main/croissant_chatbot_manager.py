# croissant_chatbot_manager.py
import json
from .constants import METADATA_ATTRIBUTES
from .llm import suggest_metadata, ask_user_for_informal_description
from .metadata_manager import MetadataManager
from typing import Dict


class CroissantChatbotManager:
    def __init__(self):

        self.history = []
        self.waiting_for_greeting = True
        self.waiting_for_informal_description = False
        self.pending_attribute = None
        self.informal_description = ""
        self.waiting_for_HF_name = False

        self.metadata_manager = MetadataManager()

    # Managing history
    def append_to_history(self, message: Dict[str, str]):
        """
        Append a message to the chat history.

        Args:
            message: A Dictionary containing the role and content of the message.
        """
        self.history.append(message)
    
    def reset_chat(self) -> list[Dict[str, str]]:
        """
        Reset the chat and metadata.

        Returns:
            The reset chat history as a list of messages.
        """
        self.metadata_manager.reset_metadata()
        self.history = []
        self.waiting_for_greeting = True
        self.waiting_for_informal_description = False
        self.pending_attribute = None
        self.informal_description = ""
        self.waiting_for_HF_name = False

        return self.history

    # Handle button clicks
    def handle_start_new_dataset(self) -> list[Dict[str, str]]:
        """
        Reset metadata attributes and prepare for annotating a new dataset.

        Returns:
            The updated chat history.
        """
        self.metadata_manager.reset_metadata()
        self.pending_attribute = None
        self.informal_description = ""
        self.waiting_for_HF_name = False
        self.append_to_history({"role": "assistant", "content": "You can now start annotating a new dataset."})
        self.display_informal_description_prompt()
        self.waiting_for_informal_description = True
        return self.history
    
    def handle_display_metadata(self):
        """
        Display the current dataset metadata.
        """
        json_metadata = self.json_to_code_block(self.metadata_manager.get_metadata())
        self.append_to_history({"role": "assistant", "content": f"Here is the current metadata for your dataset:\n{json_metadata}"})
    
    def handle_display_chatbot_instructions(self):
        """
        Display detailed instructions for using the chatbot.
        """
        instructions = f"""
            This is a very simple chatbot that helps you create Croissant metadata for your dataset.
            When you start the chat, the chatbot will guide you through the process of entering metadata attributes.\n

            At the start of the chat, you can provide an informal description of your dataset.
            This will help the chatbot suggest more relevant metadata attributes for you. \n
            You can also provide the name of your dataset if it is from Hugging Face Datasets.
            The chatbot will try to fetch some metadata for you based on the dataset name.
            You can only do these things at the start of the chat, if you want to do them later
            you must click the "Refresh Chat" button or type 'start new dataset' to to start over.\n

            To enter metadata attributes, select an attribute from the dropdown and enter the value in the 'Chat Message' box.
            When you select the attribute, the chatbot will provide guidance on what to enter using the attribute name and other information you have provided as a prompt.
            You must enter a value immediately after selecting the attribute name.\n
            After entering the value of an attribute, the chatbot will check the validity and quality of the value and suggest improvements if needed.
            If the value is valid, the chatbot will save the value and you can proceed to the next attribute.
            If the value is invalid, the chatbot will ask you to provide a new value.
            If you want to confirm the value despite validation issues, type 'confirm' in the 'Chat Message' box.
            If you want to update an attribute, select the attribute name and enter the new value.
            If you are unsure about a value, you can skip the attribute by not entering anything or selecting an attribute.\n

            If you want to see the current metadata stored in the chatbot, click the 'Display Metadata So Far' button.
            If you think you have entered all the metadata attributes, type 'complete' in the 'Chat Message' box to finalise the metadata.
            After finalising the metadata, the chatbot will display the metadata in JSON format and save it to a file, if all the fields are valid or confirmed.
            Once the metadata is finalised, you can download the metadata file by clicking the 'Download Metadata File' button, 
            and the file will appear next to the button for download when you click the blue file size text.\n
            If you want to start annotating a new dataset, type 'start new dataset' in the 'Chat Message' box.\n

            This chatbot is not smart. It will not be able to process any input that is not a value for a metadata attribute
            or one of the expected commands when prompted for: 'confirm', 'complete', 'start new dataset', 'help', 'no'.
            When asked a question you must immediately answer it before the chatbot can proceed.
            You cannot delete an attribute if you have already entered a value for it.
            If you want to delete an attribute, you must click the 'Refresh Chat' button or type 'start new dataset' to start over.
            Please do not press any button or enter anything unexpected when the chatbot is waiting for a response, otherwise it may not work as expected.\n

            The validation and quality checks are basic and may not cover all possible issues that could be wrong with the metadata.
            The validation checks do the following:
            - Check if the url is in the correct format of http(s)://...
            - Check if the license is from the SPDX License List.
            - Check if the date (date_published, date_created, date_modified) is in the correct format of YYYY-MM-DD.
            - Check if the language is a comma-separated list of of ISO 639-1 codes or language names.
            - Check if cite_as is in valid BibTeX format.
            - Check if creators, keywords, task, modality are a comma-separated list of non-empty strings.
            - Check if name, description, publisher, version, url, license, date_modified, date_created, date_published, cite_as are non-empty strings.

            The quality checks do the following:
            - Check if the description has high lexical diversity and sentence variety
            - Check if the keywords are at least 3 and are not repeated values

            These checks are meant to guide you in providing the best metadata possible. \n

            PLEASE NOTE: The chatbot may not always provide the accurate suggestions or guidance.

            If you encounter any issues, you can refresh the chat by clicking the 'Refresh Chat' button.
            If you want to see these instructions again at any time, press the 'See Instructions' button.\n
        """
        formatted_instructions = f"```text\n{instructions}\n```"
        self.append_to_history({"role": "assistant", "content": formatted_instructions})
    
    # Display methods
    def display_short_instructions(self):
        """
        Display short instructions for the chatbot.
        """
        self.append_to_history({
            "role": "assistant",
            "content": """Please follow the following instructions:
                - Select any attribute from the dropdown to enter/update its value. 
                - If the value is empty, I will provide guidance on what to enter using the informal description.
                - If the value is already filled, you can update it if needed.
                - If you want to see the current metadata I know about your dataset, click the 'Display Metadata So Far' button.
                - If you want to see a longer version of the instructions, click the 'See Instructions' button.
                - If you make a mistake, specifically if you add an attribute you don't want to include, press the "Refresh Chat" button to start over.
                - If you think you have entered all the metadata attributes, type **complete** in the 'Chat Message' box to finalise the metadata."""
        })

    def display_informal_description_prompt(self):
        """
        Prompt the user to provide an informal description of the dataset.
        """
        self.append_to_history({
            "role": "assistant",
            "content": """Would you like to provide an informal description of your dataset? 
                This is optional, but it will help me suggest more relevant metadata attributes for you.
                It can be as little as a few words or as long as you want.
                **Please type one of these options:**
                - &lt;your informal description&gt;
                - help (will provide guidance on providing an informal description) 
                - no"""
        })
    
    def display_hugging_face_name_prompt(self):
        """
        Prompt the user to provide the Hugging Face dataset name.
        """
        self.append_to_history({
            "role": "assistant", 
            "content": """Is your dataset from Hugging Face Datasets?
                If so, please provide the name of the dataset so I can fetch some metadata for you. 
                **Please type one of these options:**
                -  &lt;dataset name&gt;
                - no"""
        })

    
    def json_to_code_block(self, json_data: Dict[str,str], default_value=None) -> str:
        """
        Convert JSON data to a formatted code block.

        Args:
            json_data: The JSON data to format.
            default_value: A default value to use for serialization if needed.

        Returns:
            A string containing the formatted JSON code block.
        """
        try:
            return f"```json\n{json.dumps(json_data, indent=2, default=default_value)}\n```"
        except TypeError as e:
            return f"```json\n{json.dumps(json_data, indent=2)}\n```"
        except Exception as e:
            return f"```json\n{{\"error\": \"{str(e)}\"}}\n```"

    
    # Handle user input methods
    def handle_user_input(self, prompt: str) -> list[Dict[str, str]]:
        """
        Handle user input through chat.

        Args:
            prompt: The user's input message.

        Returns:
            The updated chat history.
        """
        try:
            if not self.history:
                self.history = []

            self.append_to_history({"role": "user", "content": prompt})
            if prompt.lower() == "start new dataset":
                self.handle_start_new_dataset()

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

            elif self.metadata_manager.is_all_attributes_filled():
                self.append_to_history({
                    "role": "assistant", 
                    "content": """All metadata attributes have been filled."""
                })
                self.display_short_instructions()
            
            else:
                self.append_to_history({"role": "assistant", "content": "You provided an unexpected input. This may cause issues."})
                self.display_short_instructions()

        except Exception as e:
            self.handle_errors(f"An unexpected error occurred while processing your input: {str(e)}")
        
        return self.history

    def handle_greeting(self) -> list[Dict[str, str]]:
        """
        Handle the initial greeting.

        Returns:
            The updated chat history.
        """
        self.append_to_history({"role": "assistant", "content": "Hello! I am a Chatbot designed to help you create Croissant metadata for your dataset."})
        self.display_informal_description_prompt()
        self.waiting_for_greeting = False
        self.waiting_for_informal_description = True
        return self.history

    def handle_informal_description_prompt(self, prompt: str) -> list[Dict[str, str]]:
        """
        Handle the user's response to the informal description prompt.

        Args:
            prompt: The user's input for the informal description.

        Returns:
            The updated chat history.
        """
        if prompt.lower() == "help":
            try:
                chatbot_response = ask_user_for_informal_description()
                self.append_to_history({"role": "assistant", "content": f"{chatbot_response}"})
            except Exception as e:
                # Handle errors from the llm module
                self.handle_errors(f"An error occurred while fetching guidance for the informal description: {str(e)}")
            return self.history
        elif prompt.lower() == "no":
            self.waiting_for_informal_description = False
        else:
            self.informal_description = prompt.strip()
            self.append_to_history({"role": "assistant", "content": f"Saved the 'informal description': {self.informal_description}"})
            self.waiting_for_informal_description = False
        self.display_hugging_face_name_prompt()
        self.waiting_for_HF_name = True
        return self.history
    
    def handle_HF_name(self, prompt: str) -> list[Dict[str, str]]:
        """
        Handle the user's response to the Hugging Face dataset name prompt.

        Args:
            prompt: The user's input for the Hugging Face dataset name.

        Returns:
            The updated chat history.
        """
        try:
            if prompt.lower() == "no":
                self.display_short_instructions()
                self.waiting_for_HF_name = False
            else:
                try:
                    dataset_info, success = self.metadata_manager.find_dataset_info(prompt.strip())
                    if success:
                        self.append_to_history({"role": "assistant", "content": "I fetched the following metadata for your dataset:"})
                        self.handle_display_metadata()
                    elif dataset_info is None:
                        self.append_to_history({"role": "assistant", "content": "I couldn't find any information for the provided dataset name."})
                    elif "error" in dataset_info:
                        self.append_to_history({"role": "assistant", "content": f"An error occurred while trying to fetch metadata information: {dataset_info['error']}"})
                    else:
                        self.append_to_history({"role": "assistant", "content": "I couldn't find any information for the provided dataset name."})
                except Exception as e:
                    self.handle_errors(f"An unexpected error occurred while fetching dataset information: {str(e)}")
                self.waiting_for_HF_name = False
                self.display_short_instructions()
        except Exception as e:
            self.handle_errors(f"An unexpected error occurred while fetching the dataset information: {str(e)}")
            self.waiting_for_HF_name = False
            self.display_short_instructions()
        return self.history

    def handle_complete_command(self) -> list[Dict[str, str]]:
        """
        Handle the 'complete' command to finalize metadata.

        Returns:
            The updated chat history.
        """
        try:
            is_valid, error_messages, issue_messages = self.metadata_manager.validate_and_check_quality_all_attributes()
            if not is_valid:
                if self.metadata_manager.get_confirmed_metadata():
                    confirmed_attributes = ",".join(self.metadata_manager.get_confirmed_metadata().keys())
                    self.append_to_history({"role": "assistant", "content": f"You have confirmed the values for these attributes '{confirmed_attributes}' despite validation issues. Finalising metadata with these values."})
                    self.metadata_manager.merge_confirmed_metadata()  # Merge confirmed metadata into main metadata
                elif error_messages or issue_messages:
                    self.append_to_history({"role": "assistant", "content": f"Here are the issues with the metadata:\n{error_messages}\n{issue_messages}"})
                    self.append_to_history({"role": "assistant", "content": "Please resolve the issues before finalising the metadata."})
                    self.append_to_history({"role": "assistant", "content": "You can select any attribute from the dropdown to update them."})
                    self.append_to_history({"role": "assistant", "content": "If you want to confirm the values despite validation issues, type **confirm** in the 'Chat Message' box after selecting each attribute."})
                    return self.history
            success, final_metadata = self.metadata_manager.finalise_metadata()
            if success:
                self.append_to_history({"role": "assistant", "content": "Here is your final Croissant metadata:"})
                self.append_to_history({"role": "assistant", "content": self.json_to_code_block(final_metadata, self.metadata_manager.json_serial)})
                self.append_to_history({"role": "assistant", "content": "You can download the metadata file by clicking the 'Download Metadata File' button, then clicking the blue file size text."})
                self.append_to_history({"role": "assistant", "content": "You can still update the metadata if needed by selecting an attribute from the dropdown."})
                self.append_to_history({"role": "assistant", "content": "If you want to start annotating a new dataset, type **start new dataset** in the 'Chat Message' box."})
            else:
                self.append_to_history({"role": "assistant", "content": f"An error occurred: {final_metadata}"})
        except Exception as e:
            self.handle_errors(f"An unexpected error occurred while finalising the metadata: {str(e)}")
        return self.history

    def handle_pending_attribute_input(self, prompt: str) -> list[Dict[str, str]]:
        """
        Handle input for a pending metadata attribute.

        Args:
            prompt: The user's input for the pending attribute.

        Returns:
            The updated chat history.
        """
        try:
            # Check if the user wants to confirm the value despite issues
            if prompt.lower() == "confirm" and self.pending_attribute:
                if self.pending_attribute in ["date_created", "date_modified", "date_published"]:
                    self.append_to_history({
                        "role": "assistant",
                        "content": f"The **confirm** option is not available for the `{self.pending_attribute}` attribute. Please provide a valid date in the correct format (YYYY-MM-DD)."
                    })
                    return self.history

                # Confirm the attribute value
                value = self.metadata_manager.get_temporary_metadata_value(self.pending_attribute)
                self.metadata_manager.confirm_metadata_value(self.pending_attribute, value)
                self.append_to_history({"role": "assistant", "content": f"Despite validation issues, the value for `{self.pending_attribute}` has been saved as: {value}"})
                self.pending_attribute = None
                return self.history

            # Otherwise, handle the input for the pending attribute
            elif self.pending_attribute:
                # Update and validate the attribute value
                self.metadata_manager.update_temporary_metadata({self.pending_attribute: prompt.strip()})
                is_valid, error_messages, issue_messages = self.metadata_manager.validate_and_check_quality(self.pending_attribute, prompt.strip())

                if not is_valid:
                    # Handle validation errors
                    self.append_to_history({"role": "assistant", "content": f"The value you provided for `{self.pending_attribute}` is invalid."})
                    self.append_to_history({"role": "assistant", "content": f"Here are the issues with the value you provided:\n{error_messages}\n{issue_messages}"})
                    suggested_value = suggest_metadata(self.metadata_manager.get_metadata(), self.informal_description, self.pending_attribute)
                    self.append_to_history({"role": "assistant", "content": f"{suggested_value}"})
                    self.append_to_history({
                        "role": "assistant", 
                        "content": f"""**Please type one of these options:**
                            - &lt;a new value for this attribute&gt;
                            - &lt;one of the suggestions for this attribute&gt;
                            - **confirm** (to confirm the value despite validation issues)
                    """})
                else:
                    self.metadata_manager.clear_temporary_metadata()
                    self.metadata_manager.set_metadata_value(self.pending_attribute, prompt.strip())
                    self.append_to_history({"role": "assistant", "content": f"Saved `{self.pending_attribute}` as: {prompt.strip()}."})
                    self.pending_attribute = None
        except Exception as e:
            self.handle_errors(f"An unexpected error occurred while processing your input for `{self.pending_attribute}`: {str(e)}")
            self.pending_attribute = None
            self.metadata_manager.clear_temporary_metadata()
        return self.history
    
    def handle_selected_attribute(self, attribute: str) -> list[Dict[str, str]]:
        """
        Handle the selection of a metadata attribute.

        Args:
            attribute: The name of the selected metadata attribute.

        Returns:
            The updated chat history.
        """
        try:
            self.pending_attribute = attribute
            self.append_to_history({"role": "user", "content": f"Selected attribute: `{attribute}`."})

            attribute_description = METADATA_ATTRIBUTES.get(attribute, "")
            current_value = self.metadata_manager.get_metadata_value(attribute)

            if not current_value:
                # Suggest a value for the attribute
                try:
                    suggested_value = suggest_metadata(self.metadata_manager.get_metadata(), self.informal_description, attribute)
                    self.append_to_history({
                        "role": "assistant",
                        "content": f"The attribute `{attribute}` is missing. This attribute should be: {attribute_description}"})
                    self.append_to_history({"role": "assistant","content": f"{suggested_value}"})
                    self.append_to_history({
                        "role": "assistant",
                        "content": f"""You can do one of the following: 
                        - Enter a new value for `{attribute}`.
                        - Use one of the suggestions provided.
                        - Select a different attribute from the dropdown if you want to skip this one.
                        """
                    })               
                except Exception as e:
                    # Handle errors from the llm module
                    self.handle_errors(f"An error occurred while suggesting metadata for `{attribute}`: {str(e)}")
            else:
                # Prompt the user to update the existing value
                self.append_to_history({"role": "assistant", "content": f"The attribute `{attribute}` already has a value: `{current_value}`"})
                self.append_to_history({"role": "assistant", "content": f"This attribute should be: {attribute_description}"})
                self.append_to_history({
                    "role": "assistant",
                    "content": f"""You can do one of the following:
                    - Update the value by entering a new one.
                    - Keep the current value by doing nothing.
                    - Re-enter the current value to undergo validation and quality checks.
                    - Enter any input to trigger suggestions for this attribute.
                    """
                })
        except Exception as e:
            self.handle_errors(f"An unexpected error occurred while processing the selected attribute `{attribute}`: {str(e)}")
            self.pending_attribute = None
            self.metadata_manager.clear_temporary_metadata()
        return self.history
    
    def handle_errors(self, error_message: str) -> list[Dict[str, str]]:
        """
        Handle errors in the chatbot.

        Args:
            error_message: The error message to display.

        Returns:
            The updated chat history.
        """
        self.append_to_history({"role": "assistant", "content": f"Error: {error_message}"})
        self.display_short_instructions()
        return self.history



