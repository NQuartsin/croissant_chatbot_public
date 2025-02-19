# plot_datasets.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("huggingface_datasets.csv")

# Count occurrences of each modality
modality_counts = df["Modality"].value_counts()

# Plot bar chart for Modality
plt.figure(figsize=(10, 6))
sns.barplot(x=modality_counts.index, y=modality_counts.values, hue=None, palette="viridis", legend=False)
plt.xlabel("Modality")
plt.ylabel("Number of Datasets")
plt.title("Distribution of Datasets by Modality")
plt.xticks(rotation=45)
plt.show(block=False)
input("Press Enter to exit...")

# Count occurrences of each task
task_counts = df["Dataset Tasks"].value_counts()

# Plot bar chart for Task Categories
plt.figure(figsize=(12, 6))
sns.barplot(x=task_counts.index, y=task_counts.values, hue=None, palette="coolwarm", legend=False)
plt.xlabel("Task Categories")
plt.ylabel("Number of Datasets")
plt.title("Distribution of Datasets by Task")
plt.xticks(rotation=90)
plt.show(block=False)
input("Press Enter to exit...")
