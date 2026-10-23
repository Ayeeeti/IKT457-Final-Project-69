from sklearn.datasets import fetch_20newsgroups
import re


# Remove headers, footers, and quotes from the training and testing datasets
train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

# Normalize the text by converting to lowercase and removing non-alphanumeric characters
def normalize_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"^http\S+|www\.\s+", " ", text) # Remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Remove non-alphanumeric characters
    text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces with a single space
    return text

# Convert text into minary features