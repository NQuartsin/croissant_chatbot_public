from huggingface_hub import InferenceClient
import gradio as gr
import os

# Initialize Hugging Face Inference Client
client = InferenceClient(
    token=os.getenv("HUGGING_FACE_API_KEY")  # Ensure your Hugging Face API key is set
)

# Define metadata fields and prompts
metadata_fields = [
    {"field": "name", "prompt": "What is the name of your dataset?"},
    {"field": "author", "prompt": "Who is the author of your dataset? (Provide full name)"},
    {"field": "year", "prompt": "What year was your dataset published?"},
    {"field": "title", "prompt": "What is the title of the dataset or associated paper?"},
    {"field": "description", "prompt": "Please provide a brief description of your dataset."},
    {"field": "license", "prompt": "What license is your dataset under?"},
    {"field": "url", "prompt": "Please provide the URL to your dataset or repository."},
    {"field": "distribution", "prompt": "How is your dataset distributed (e.g., GitHub repo, file formats)?"},
    {"field": "structure", "prompt": "Please specify the structure of your dataset (e.g., fields like context and completion)."},
]

# Metadata storage
metadata = {}
current_field_idx = 0

def generate_bibtex(metadata):
    """Generates a single-line BibTeX citation from metadata fields."""
    bibtex_entry = f"@misc{{{metadata.get('author', 'unknown').split(' ')[0]}{metadata.get('year', 'XXXX')}," \
                   f" author = {{{metadata.get('author', 'Unknown Author')}}}," \
                   f" title = {{{metadata.get('title', 'Untitled Dataset')}}}," \
                   f" year = {{{metadata.get('year', 'XXXX')}}}," \
                   f" url = {{{metadata.get('url', 'N/A')}}} }}"
    return bibtex_entry

def respond(prompt: str, history):
    global current_field_idx

    if not history:
        history = [{"role": "system", "content": "I will guide you through generating metadata for your dataset, including a BibTeX citation."}]

    # Save user input to metadata
    if current_field_idx < len(metadata_fields):
        field = metadata_fields[current_field_idx]["field"]
        metadata[field] = prompt.strip()

        # Append user input to history
        history.append({"role": "user", "content": prompt})

        # Move to next question or finish
        current_field_idx += 1
        if current_field_idx < len(metadata_fields):
            next_prompt = metadata_fields[current_field_idx]["prompt"]
        else:
            next_prompt = "Thanks for sharing the information! Here is your dataset metadata:"

        # Append assistant response
        history.append({"role": "assistant", "content": next_prompt})
        yield history  # Show response immediately

    # If all fields are filled, display final metadata including BibTeX
    if current_field_idx >= len(metadata_fields):
        metadata_json = {
            "@type": "sc:Dataset",
            "name": metadata.get("name"),
            "citeAs": generate_bibtex(metadata),  # Store BibTeX as a single-line string
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
        yield history  # Show metadata JSON

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
