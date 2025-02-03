from huggingface_hub import InferenceClient
import gradio as gr
import os

# Initialize Hugging Face Inference Client
client = InferenceClient(
    token=os.getenv("HUGGING_FACE_API_KEY")  # Make sure you have your Hugging Face API key set
)

# Store metadata fields in a dictionary
metadata = {}

def respond(prompt: str, history):
    # Initialize history if empty
    if not history:
        history = [{"role": "system", "content": "You are a friendly chatbot. I will guide you through creating metadata for your dataset."}]
    
    # Process user input
    history.append({"role": "user", "content": prompt})

    # Check if the first question is about the dataset name
    if "name" not in metadata:
        metadata["name"] = prompt
        response = {"role": "assistant", "content": "Great! Now, can you provide a citation for your dataset (title, authors, year, etc.)?"}
    # Ask for citation if dataset name is collected
    elif "citation" not in metadata:
        metadata["citation"] = prompt
        response = {"role": "assistant", "content": "Thanks! Now, can you provide a brief description of your dataset?"}
    # Ask for description if citation is collected
    elif "description" not in metadata:
        metadata["description"] = prompt
        response = {"role": "assistant", "content": "Awesome! What license is your dataset under?"}
    # Ask for license if description is collected
    elif "license" not in metadata:
        metadata["license"] = prompt
        response = {"role": "assistant", "content": "Got it! Please provide the URL to your dataset or repository."}
    # Ask for URL if license is collected
    elif "url" not in metadata:
        metadata["url"] = prompt
        response = {"role": "assistant", "content": "Great! Could you specify how the dataset is distributed (e.g., GitHub repo, file formats)?"}
    # Ask for distribution info if URL is collected
    elif "distribution" not in metadata:
        metadata["distribution"] = prompt
        response = {"role": "assistant", "content": "Perfect! Please specify the structure of your dataset (e.g., fields like context and completion)."}
    # Ask for dataset structure (fields) if distribution info is collected
    elif "structure" not in metadata:
        metadata["structure"] = prompt
        response = {"role": "assistant", "content": "Thanks for sharing the dataset structure! Your metadata is now ready."}

    # Yield the updated history and response
    yield history + [response]

    # After all fields are collected, show metadata as JSON
    if "structure" in metadata:
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
        response["content"] = f"Here is your dataset metadata:\n\n{metadata_json}"
        yield history + [response]

with gr.Blocks() as demo:
    gr.Markdown("# Chat with Hugging Face Zephyr 7b 🤗")
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
