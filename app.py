from huggingface_hub import InferenceClient
import gradio as gr
import os

# Initialize Hugging Face Inference Client
client = InferenceClient(
    token=os.getenv("HUGGING_FACE_API_KEY")  # Make sure you have your Hugging Face API key set
)

# Define metadata fields and associated prompts
metadata_fields = [
    {"field": "name", "prompt": "What is the name of your dataset?"},
    {"field": "citation", "prompt": "Can you provide a citation for your dataset (title, authors, year, etc.)?"},
    {"field": "description", "prompt": "Please provide a brief description of your dataset."},
    {"field": "license", "prompt": "What license is your dataset under?"},
    {"field": "url", "prompt": "Please provide the URL to your dataset or repository."},
    {"field": "distribution", "prompt": "How is your dataset distributed (e.g., GitHub repo, file formats)?"},
    {"field": "structure", "prompt": "Please specify the structure of your dataset (e.g., fields like context and completion)."},
]

# Initialize metadata storage
metadata = {}
current_field_idx = 0

def respond(prompt: str, history):
    global current_field_idx

    if not history:
        history = [{"role": "system", "content": "You are a chatbot that helps users create machine-readable metadata for datasets."}]
    
    # Save user input to the metadata dictionary
    if current_field_idx < len(metadata_fields):
        field = metadata_fields[current_field_idx]["field"]
        metadata[field] = prompt

        # Append user input to history
        history.append({"role": "user", "content": prompt})

        # Move to next question or finish
        current_field_idx += 1
        if current_field_idx < len(metadata_fields):
            next_prompt = metadata_fields[current_field_idx]["prompt"]
        else:
            next_prompt = "Thanks for sharing the information! Here is your dataset metadata:"

        # Append assistant response to history
        history.append({"role": "assistant", "content": next_prompt})

        yield history  # Display user input + next question

    # If all fields are filled, display the final metadata
    if current_field_idx >= len(metadata_fields):
        metadata_json = {
            "@type": "sc:Dataset",
            "name": metadata.get("name"),
            "citeAs": metadata.get("citation"),
            "description": metadata.get("description"),
            "license": metadata.get("license"),
            "url": metadata.get("url"),
            "distribution": metadata.get("distribution"),
            "recordSet": metadata.get("structure"),
            "@context": {
                "@language": "en",
                "@vocab": "https://schema.org/",
                "cr": "http://mlcommons.org/croissant/",
                "dct": "http://purl.org/dc/terms/",
            },
        }

        history.append({"role": "assistant", "content": f"```json\n{metadata_json}\n```"})
        yield history  # Display the final metadata JSON

with gr.Blocks() as demo:
    gr.Markdown("# Dataset Metadata Creator")
    chatbot = gr.Chatbot(
        label="Metadata Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/376/hugging-face_1f917.png",
        ),
    )
    prompt = gr.Textbox(max_lines=1, label="Chat Message")
    prompt.submit(respond, [prompt, chatbot], [chatbot])
    prompt.submit(lambda: "", None, [prompt])

if __name__ == "__main__":
    demo.launch()
