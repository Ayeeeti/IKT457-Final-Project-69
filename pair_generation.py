import re

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Normalize the text by converting to lowercase and removing non-alphanumeric characters
def normalize_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"(https?://|www\.)\S+", " ", text)  # Remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)        # Remove non-alphanumeric characters
    text = re.sub(r"\s+", " ", text).strip()        # Replace multiple spaces with a single space
    return text

# Indices of documents with at least one vocabulary word.
def non_empty_docs(x):
    return np.flatnonzero(np.asarray(x.sum(axis=1)).ravel() > 0)


# Convert text into binary features
vectorizer = CountVectorizer(
    binary=True,            # 1 if a word is present, 0 otherwise
    max_features=1000,      # Keep at most 1 000 vocabulary words
    stop_words="english",   # Remove common english words
    min_df=5                # Ignore words appearing fewer than 5 training documents
    )


# Remove headers, footers, and quotes from the training and testing datasets
train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)


X_train_texts = [normalize_text(text) for text in train_data.data]
X_test_texts = [normalize_text(text) for text in test_data.data]

# Split documents before creating pairs or fitting the vocabulary.
train_document_indices, val_document_indices = train_test_split(
    np.arange(len(X_train_texts)),
    test_size=0.2,
    random_state=42,
    stratify=train_data.target,
)

# The document splits must not share source-document indices
assert len(set(train_document_indices).intersection(val_document_indices)) == 0

X_fit_texts = [X_train_texts[i] for i in train_document_indices]
X_val_texts = [X_train_texts[i] for i in val_document_indices]

# Fit the vocabulary only on training documents.
X_train_docs = vectorizer.fit_transform(X_fit_texts)
X_val_docs = vectorizer.transform(X_val_texts)
X_test_docs = vectorizer.transform(X_test_texts)

print("Training binary document matrix:", X_train_docs.shape)
print("Validation binary document matrix:", X_val_docs.shape)
print("Test binary document matrix:", X_test_docs.shape)


# Generate balanced document pairs
# Want to create a balanced dataset with 50% positive and 50% negative pairs
# A positive pair consists of two documents from the same category, 
# while a negative pair consists of two documents from different categories

def generate_pairs(labels, n_pairs=5000, seed=42, valid_indices=None):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    if valid_indices is None:
        valid_indices = np.arange(len(labels))
    else:
        valid_indices = np.asarray(valid_indices, dtype=int)

    if n_pairs <= 0 or n_pairs % 2 != 0:
        raise ValueError("n_pairs must be a positive, even number.")
    if len(labels) == 0:
        raise ValueError("labels must not be empty.")
    if np.any(valid_indices < 0) or np.any(valid_indices >= len(labels)):
        raise ValueError("valid_indices contains an out-of-range document index.")

    pairs = []
    labels_list = []
    seen_pairs = set()

    # Group document indices by category
    category_indices = {}
    for idx in valid_indices:
        label = labels[idx]
        if label not in category_indices:
            category_indices[label] = []
        category_indices[label].append(idx)

    categories = list(category_indices.keys())
    positive_categories = [
        category for category in categories
        if len(category_indices[category]) >= 2
    ]

    if len(positive_categories) == 0:
        raise ValueError("At least one category must contain two valid documents.")
    if len(categories) < 2:
        raise ValueError("At least two categories are required to generate negative pairs.")

    # Generate positive pairs
    attempts = 0
    max_attempts = max(1000, n_pairs * 1000)
    while len(pairs) < n_pairs // 2:
        category = rng.choice(positive_categories)
        indx1, indx2 = rng.choice(category_indices[category], size=2, replace=False)
        pair = tuple(sorted((int(indx1), int(indx2))))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            pairs.append(pair)
            labels_list.append(1)
        attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("Could not generate enough unique positive pairs.")


    # Generate negative pairs
    while len(pairs) < n_pairs:
        cat1, cat2 = rng.choice(categories, size=2, replace=False)
        indx1 = rng.choice(category_indices[cat1])
        indx2 = rng.choice(category_indices[cat2])
        pair = tuple(sorted((int(indx1), int(indx2))))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            pairs.append(pair)
            labels_list.append(0)
        attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("Could not generate enough unique negative pairs.")

    # Shuffle and return 
    permutation = rng.permutation(len(pairs))
    pairs = np.array(pairs)[permutation]
    labels_list = np.array(labels_list)[permutation]
    return pairs, labels_list


train_valid_idx = non_empty_docs(X_train_docs)
val_valid_idx = non_empty_docs(X_val_docs)
test_valid_idx = non_empty_docs(X_test_docs)

train_labels = train_data.target[train_document_indices]
val_labels = train_data.target[val_document_indices]

train_pairs, y_train_pairs = generate_pairs(
    train_labels,
    n_pairs=20000,
    seed=42,
    valid_indices=train_valid_idx,
)

# Each row = [words of document A | words of document B]
def pair_to_vectors(pair, X):
    return sp.hstack([X[pair[0]], X[pair[1]]], format="csr")

# Convert each generated pair into a feature vector
X_train_pairs = sp.vstack([
    pair_to_vectors(pair, X_train_docs)
    for pair in train_pairs
])

val_pairs, y_val_pairs = generate_pairs(
    val_labels,
    n_pairs=1000,
    seed=43,
    valid_indices=val_valid_idx,
)

X_val_pairs = sp.vstack([
    pair_to_vectors(pair, X_val_docs)
    for pair in val_pairs
])

# Generate test pairs separately from the test documents and their targets.
test_pairs, test_pair_labels = generate_pairs(
    test_data.target,
    n_pairs=1000,
    seed=44,
    valid_indices=test_valid_idx
)

X_test_pairs = sp.vstack([
    pair_to_vectors(pair, X_test_docs)
    for pair in test_pairs
    ])

# Convert pair-local document indices back to indices in the original datasets
# This lets us verify that every document used by the pairs belongs to one split only
train_pair_documents = set(
    train_document_indices[np.asarray(train_pairs, dtype=int).ravel()]
)
val_pair_documents = set(
    val_document_indices[np.asarray(val_pairs, dtype=int).ravel()]
)
test_pair_documents = set(np.asarray(test_pairs, dtype=int).ravel())

assert train_pair_documents.isdisjoint(val_pair_documents)

print("Training documents used in pairs:", len(train_pair_documents))
print("Validation documents used in pairs:", len(val_pair_documents))
print("Test documents used in pairs:", len(test_pair_documents))


# Check that labels and feature matrix align
print("Training pair feature matrix:", X_train_pairs.shape)
print("Training pair label shape:", y_train_pairs.shape)
print("Validation pair feature matrix:", X_val_pairs.shape)
print("Validation pair label shape:", y_val_pairs.shape)
print("Test pair feature matrix:", X_test_pairs.shape)
print("Test pair label shape:", test_pair_labels.shape)

print("Training pair balance [negative, positive]:", np.bincount(y_train_pairs))
print("Validation pair balance [negative, positive]:", np.bincount(y_val_pairs))
print("Test pair balance [negative, positive]:", np.bincount(test_pair_labels))
print("Number of unique test pairs:", len(set(map(tuple, test_pairs))))
print("Total number of generated test pairs:", len(test_pairs))


'''
Preprocessing stegene som er blitt gjort:
- Lowercasing text
- Removing URLs, punctuation, and extra whitespace, headers, footers, and quotes
- Removing non-alphanumeric characters
- Converting text into binary features using CountVectorizer
- Filtering documents with no vocabulary words
- Concentrating each pair's two document vectors into one 2k-feature vector
- Building a 1k-wordk training vocabulary (CountVectorizer)
'''

'''
Feature representation:
- Each document is represented as a binary vector of length 1000
- A document pair is represented as a concatenation of two document vectors,
resulting in a feature vector of length 2000
- Vocab is fitted only on the training data to avoid data leakage,
and the same training vocab and feature positions are then used to
transform the test documents

'''