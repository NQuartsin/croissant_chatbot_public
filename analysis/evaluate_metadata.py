import json
import os
from textblob import TextBlob
import spacy
import csv

# Load the English NLP model
nlp = spacy.load("en_core_web_md")

# Mapping of attributes from Hugging Face metadata to chatbot metadata
def get_attribute_mapping():
    return {
        "name": "name",
        "creators": "creators",
        "description": "description",
        "license": "license",
        "url": "url",
        "publisher": "publisher",
        "version": "version",
        "keywords": "keywords",
        "date_created": "dateCreated",
        "date_modified": "dateModified",
        "date_published": "datePublished",
        "cite_as": "citeAs",
        "task": "task",
        "modality": "modality",
        "in_language": "inLanguage"
    }

def load_metadata(file_path):
    """Load metadata from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def clean_metadata(metadata):
    """Remove unwanted fields like @context and @type from the metadata."""
    keys_to_remove = ["@context", "@type"]
    return {key: value for key, value in metadata.items() if key not in keys_to_remove}

def match_metadata_files(hf_metadata_folder, chatbot_metadata_folder):
    """
    Match Hugging Face metadata files with chatbot metadata files based on the 'name' key in the JSON.
    
    Args:
        hf_metadata_folder (str): Folder containing Hugging Face metadata files.
        chatbot_metadata_folder (str): Folder containing chatbot-generated metadata files.
        
    Returns:
        dict: A dictionary mapping Hugging Face metadata filenames to chatbot metadata filenames.
    """
    hf_files = {}
    chatbot_files = {}
    # Load Hugging Face files and store them by their 'name' key
    for filename in os.listdir(hf_metadata_folder):
        if filename.endswith('_hf.json'):
            file_path = os.path.join(hf_metadata_folder, filename)
            hf_metadata = load_metadata(file_path)
            name = hf_metadata.get('name')
            if name:
                hf_files[name] = file_path
    
    # Load chatbot metadata files, clean unwanted fields, and store them by their 'name' key
    for filename in os.listdir(chatbot_metadata_folder):
        if filename.endswith('_metadata.json'):
            file_path = os.path.join(chatbot_metadata_folder, filename)
            chatbot_metadata = load_metadata(file_path)
            cleaned_metadata = clean_metadata(chatbot_metadata)  # Clean the metadata
            name = cleaned_metadata.get('name')
            if name:
                chatbot_files[name] = file_path
    
    # Match files based on the 'name' key
    matched_files = {}
    
    for name, hf_file in hf_files.items():
        if name in chatbot_files:
            matched_files[hf_file] = chatbot_files[name]
    
    return matched_files


def get_char_length(text):
    """Returns the length of the text in characters."""
    return len(text)

def get_word_count(text):
    """Returns the number of meaningful words in the text."""
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]  # Ignore punctuation, count words
    return len(words)


def evaluate_description_subjectivity(description):
    """Evaluate description subjectivity using TextBlob."""
    if description == "":
        return 0.0
    blob = TextBlob(description)
    subjectivity_score = blob.sentiment.subjectivity  # 0.0 is very objective and 1.0 is very subjective.

    return round(subjectivity_score, 2)

def evaluate_task_quality(task):
    """Evaluate the task description based on actionability and specificity, with a score between 0 and 1."""
    if task == "":
        return 0.0, 0.0
    
    doc = nlp(task)

    # Extract action verbs (more verbs indicate more actionability)
    action_verbs = [token for token in doc if token.pos_ == "VERB"]
    # Extract named entities (more entities indicate more specificity)
    entities = [ent.text for ent in doc.ents]
    
    # Get total number of words (tokens) in the task description
    total_tokens = len([token for token in doc if not token.is_stop and not token.is_punct])

    # Normalize the counts by total_tokens to get values between 0 and 1
    # Actionability score (based on verbs)
    actionability_score = len(action_verbs) / total_tokens if total_tokens > 0 else 0.0
    
    # Specificity score (based on entities)
    specificity_score = len(entities) / total_tokens if total_tokens > 0 else 0.0

    return actionability_score, specificity_score

def evaluate_keyword_similarity(keywords):
    """Check if keywords are too similar (not diverse)."""
    if keywords == "":
        return 0.0
    
    keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
    
    similarities = []
    for i in range(len(keyword_list)):
        for j in range(i + 1, len(keyword_list)):
            sim = nlp(keyword_list[i]).similarity(nlp(keyword_list[j]))
            similarities.append(sim)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    return round(avg_similarity, 2)

def evaluate_metadata_quality(hf_metadata, chatbot_metadata):
    hf_quality_results = {}
    chatbot_quality_results = {}

    # Task Quality Evaluation
    hf_task = hf_metadata.get("task", "")
    chatbot_task = chatbot_metadata.get("task", "")
    hf_actionability_score, hf_specificity_score = evaluate_task_quality(hf_task)
    chatbot_actionability_score, chatbot_specificity_score = evaluate_task_quality(chatbot_task)

    hf_quality_results["task_quality"] = {
        "task_length": get_char_length(hf_task),
        "task_actionability_score": hf_actionability_score,
        "task_specificity_score": hf_specificity_score,
        "task_count": len(hf_task.split(',')),  # Number of tasks listed
    }
    
    chatbot_quality_results["task_quality"] = {
        "task_length": get_char_length(chatbot_task),
        "task_actionability_score": chatbot_actionability_score,
        "task_specificity_score": chatbot_specificity_score,
        "task_count": len(chatbot_task.split(',')),  # Number of tasks listed
    }

    # Modality Quality Evaluation
    hf_modality = hf_metadata.get("modality", "")
    chatbot_modality = chatbot_metadata.get("modality", "")

    hf_quality_results["modality_quality"] = {
        "modality_count": len(hf_modality.split(',')),  # Number of modalities listed
    }
    
    chatbot_quality_results["modality_quality"] = {
        "modality_count": len(chatbot_modality.split(',')),  # Number of modalities listed
    }

    # Keyword Quality Evaluation
    hf_keywords = hf_metadata.get("keywords", "")
    chatbot_keywords = chatbot_metadata.get("keywords", "")

    hf_quality_results["keyword_quality"] = {
        "keywords_length": get_char_length(hf_keywords),
        "keyword_similarity_score": evaluate_keyword_similarity(hf_keywords),
        "keywords_count": len(hf_keywords.split(',')), # Number of keywords listed
    }
    
    chatbot_quality_results["keyword_quality"] = {
        "keywords_length": get_char_length(chatbot_keywords),
        "keyword_similarity_score": evaluate_keyword_similarity(chatbot_keywords),
        "keywords_count": len(chatbot_keywords.split(',')), # Number of keywords listed
    }

    # Description Quality Evaluation
    hf_description = hf_metadata.get("description", "")
    chatbot_description = chatbot_metadata.get("description", "")

    hf_quality_results["description_quality"] = {
        "description_word_count": get_word_count(hf_description),
        "description_subjectivity": evaluate_description_subjectivity(hf_description),
    }
    
    chatbot_quality_results["description_quality"] = {
        "description_word_count": get_word_count(chatbot_description),
        "description_subjectivity": evaluate_description_subjectivity(chatbot_description),
    }

    return hf_quality_results, chatbot_quality_results

def compare_metadata(hf_metadata, chatbot_metadata):

    comparison_results = {
        "dataset_name": hf_metadata.get('name'),
        "completeness": {"hf": 0, "chatbot": 0},  # Track which source has more filled attributes
        "value_match": 0,
        "total_attributes": 0
    }
    attribute_mapping = get_attribute_mapping()

    for hf_attribute, hf_value in hf_metadata.items():
        
        # Use the mapping to find the equivalent chatbot attribute
        chatbot_attribute = attribute_mapping.get(hf_attribute)

        if chatbot_attribute and chatbot_attribute in chatbot_metadata:
            chatbot_value = chatbot_metadata[chatbot_attribute]
            comparison_results["total_attributes"] += 1

            # Check completeness for both Hugging Face and chatbot
            if hf_value:  # Hugging Face has a filled attribute
                comparison_results["completeness"]["hf"] += 1
            if chatbot_value:  # Chatbot has a filled attribute
                comparison_results["completeness"]["chatbot"] += 1

            # Check if values match (value_match)
            if hf_value == chatbot_value:
                comparison_results["value_match"] += 1

    # Calculate completeness and value_match rates
    if comparison_results["total_attributes"] > 0:
        comparison_results["completeness"]["hf"] /= comparison_results["total_attributes"]
        comparison_results["completeness"]["chatbot"] /= comparison_results["total_attributes"]
        comparison_results["value_match"] /= comparison_results["total_attributes"]
    
    return comparison_results

# Function to evaluate datasets
def evaluate_datasets(hf_metadata_folder, chatbot_metadata_folder):

    matched_files = match_metadata_files(hf_metadata_folder, chatbot_metadata_folder)
    evaluation_results = []
    for hf_file, chatbot_file in matched_files.items():

        # Load and compare the first matched dataset
        hf_metadata = load_metadata(hf_file)
        chatbot_metadata = load_metadata(chatbot_file)
        
        # Clean the chatbot metadata to remove unwanted fields
        cleaned_chatbot_metadata = clean_metadata(chatbot_metadata)
        
        comparison_result = compare_metadata(hf_metadata, cleaned_chatbot_metadata)
        
        # Evaluate quality metrics for task, modality, description, and keywords
        hf_quality, chatbot_quality = evaluate_metadata_quality(hf_metadata, cleaned_chatbot_metadata)
        
        # Combine all results into a single dictionary
        result = {
            "comparison": comparison_result,
            "hf_quality": hf_quality,
            "chatbot_quality": chatbot_quality
        }
        evaluation_results.append(result)
    

    return evaluation_results

hf_metadata_folder = "hf_metadata"
chatbot_metadata_folder = "annotations"

results = evaluate_datasets(hf_metadata_folder, chatbot_metadata_folder)

# Save results to a JSON file
output_file = "analysis/metadata_evaluation.json"
with open(output_file, "w") as file:
    json.dump(results, file, indent=4)

print(f"Results saved to {output_file}")

def calculate_averages(results):
    """Calculate percentage averages for HF, Chatbot, and Comparison metrics."""
    hf_totals = {
        "task_quality": 0,
        "modality_quality": 0,
        "keyword_quality": 0,
        "description_quality": 0
    }
    chatbot_totals = {
        "task_quality": 0,
        "modality_quality": 0,
        "keyword_quality": 0,
        "description_quality": 0
    }
    comparison_totals = {
        "completeness_hf": 0,
        "completeness_chatbot": 0,
        "value_match": 0
    }

    # Track maximum values for normalization
    max_values = {
        "task_length": 0,
        "keywords_length": 0,
        "description_word_count": 0
    }

    num_datasets = len(results)

    # First pass: Calculate totals and find maximum values
    for result in results:
        # HF Quality
        hf_totals["task_quality"] += result["hf_quality"]["task_quality"]["task_length"]
        hf_totals["modality_quality"] += result["hf_quality"]["modality_quality"]["modality_count"]
        hf_totals["keyword_quality"] += result["hf_quality"]["keyword_quality"]["keywords_length"]
        hf_totals["description_quality"] += result["hf_quality"]["description_quality"]["description_word_count"]

        # Chatbot Quality
        chatbot_totals["task_quality"] += result["chatbot_quality"]["task_quality"]["task_length"]
        chatbot_totals["modality_quality"] += result["chatbot_quality"]["modality_quality"]["modality_count"]
        chatbot_totals["keyword_quality"] += result["chatbot_quality"]["keyword_quality"]["keywords_length"]
        chatbot_totals["description_quality"] += result["chatbot_quality"]["description_quality"]["description_word_count"]

        # Comparison Metrics
        comparison_totals["completeness_hf"] += result["comparison"]["completeness"]["hf"]
        comparison_totals["completeness_chatbot"] += result["comparison"]["completeness"]["chatbot"]
        comparison_totals["value_match"] += result["comparison"]["value_match"]

        # Update maximum values
        max_values["task_length"] = max(max_values["task_length"], result["hf_quality"]["task_quality"]["task_length"], result["chatbot_quality"]["task_quality"]["task_length"])
        max_values["keywords_length"] = max(max_values["keywords_length"], result["hf_quality"]["keyword_quality"]["keywords_length"], result["chatbot_quality"]["keyword_quality"]["keywords_length"])
        max_values["description_word_count"] = max(max_values["description_word_count"], result["hf_quality"]["description_quality"]["description_word_count"], result["chatbot_quality"]["description_quality"]["description_word_count"])

    # Second pass: Normalize and calculate averages
    hf_averages = {
        "task_quality": round((hf_totals["task_quality"] / (num_datasets * max_values["task_length"])) * 100, 2) if max_values["task_length"] > 0 else 0,
        "modality_quality": round((hf_totals["modality_quality"] / num_datasets) * 100, 2),
        "keyword_quality": round((hf_totals["keyword_quality"] / (num_datasets * max_values["keywords_length"])) * 100, 2) if max_values["keywords_length"] > 0 else 0,
        "description_quality": round((hf_totals["description_quality"] / (num_datasets * max_values["description_word_count"])) * 100, 2) if max_values["description_word_count"] > 0 else 0
    }
    chatbot_averages = {
        "task_quality": round((chatbot_totals["task_quality"] / (num_datasets * max_values["task_length"])) * 100, 2) if max_values["task_length"] > 0 else 0,
        "modality_quality": round((chatbot_totals["modality_quality"] / num_datasets) * 100, 2),
        "keyword_quality": round((chatbot_totals["keyword_quality"] / (num_datasets * max_values["keywords_length"])) * 100, 2) if max_values["keywords_length"] > 0 else 0,
        "description_quality": round((chatbot_totals["description_quality"] / (num_datasets * max_values["description_word_count"])) * 100, 2) if max_values["description_word_count"] > 0 else 0
    }
    comparison_averages = {
        "completeness_hf": round((comparison_totals["completeness_hf"] / num_datasets) * 100, 2),
        "completeness_chatbot": round((comparison_totals["completeness_chatbot"] / num_datasets) * 100, 2),
        "value_match": round((comparison_totals["value_match"] / num_datasets) * 100, 2)
    }

    return hf_averages, chatbot_averages, comparison_averages

def save_to_csv(hf_averages, chatbot_averages, comparison_averages, output_file):
    """Save the averages to a CSV file."""
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write headers
        writer.writerow(["Metric", "HF Average (%)", "Chatbot Average (%)", "Comparison Average (%)"])

        # Write HF and Chatbot Quality Metrics
        writer.writerow(["Task Quality", hf_averages["task_quality"], chatbot_averages["task_quality"], "N/A"])
        writer.writerow(["Modality Quality", hf_averages["modality_quality"], chatbot_averages["modality_quality"], "N/A"])
        writer.writerow(["Keyword Quality", hf_averages["keyword_quality"], chatbot_averages["keyword_quality"], "N/A"])
        writer.writerow(["Description Quality", hf_averages["description_quality"], chatbot_averages["description_quality"], "N/A"])

        # Write Comparison Metrics
        writer.writerow(["Completeness (HF)", "N/A", "N/A", comparison_averages["completeness_hf"]])
        writer.writerow(["Completeness (Chatbot)", "N/A", "N/A", comparison_averages["completeness_chatbot"]])
        writer.writerow(["Value Match", "N/A", "N/A", comparison_averages["value_match"]])

    print(f"Averages saved to {output_file}")

# Calculate averages
hf_averages, chatbot_averages, comparison_averages = calculate_averages(results)

# Save averages to a CSV file
csv_output_file = "analysis/evaluation_averages.csv"
save_to_csv(hf_averages, chatbot_averages, comparison_averages, csv_output_file)