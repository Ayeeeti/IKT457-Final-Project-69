from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import CountVectorizer


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

X_train_texts = [normalize_text(text) for text in train_data.data]
X_test_texts = [normalize_text(text) for text in test_data.data]

# Convert text into binary features

vectorizer = CountVectorizer(
    binary=True,            # 1 if a word is present, 0 otherwise
    max_features=1000,      # Keep at most 1 000 vocabulary words
    stop_words="english",   # Remove common english words
    min_df=5                # Ignore words appearing fewer than 5 training documents
    )

X_train_docs = vectorizer.fit_transform(X_train_texts)
X_test_docs = vectorizer.transform(X_test_texts)

print(X_train_docs.shape)
print(X_test_docs.shape)