from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

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
    text = re.sub(r"^http\S+|www\.\s+", " ", text)  # Remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)        # Remove non-alphanumeric characters
    text = re.sub(r"\s+", " ", text).strip()        # Replace multiple spaces with a single space
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



# Fit.transform on training data and transform on testing data ensures that the test does not influence the learning vocab
X_train_docs = vectorizer.fit_transform(X_train_texts)
X_test_docs = vectorizer.transform(X_test_texts)

print(X_train_docs.shape)
print(X_test_docs.shape)


# Generate balanced document pairs
# We want to create a balanced dataset with 50% positive and 50% negative pairs
# A positive pair consists of two documents from the same category, 
# while a negative pair consists of two documents from different categories

def generate_pairs(labels, n_pairs=5000, seed=42):
    rng = np.random.default_rng(seed)

    pairs = []
    labels_list = []
    seen_pairs = set()

    # Group document indices by category
    category_indices = {}

    for idx, label in enumerate(labels):
        if label not in category_indices:
            category_indices[label] = []
        category_indices[label].append(idx)

    categories = list(category_indices.keys())

    while len(pairs) < n_pairs // 2:
        category = rng.choice(categories)

        indx1, indx2 = rng.choice(
            category_indices[category],
            size=2,
            replace=False
        )

        pair = tuple(sorted((int(indx1), int(indx2))))
        
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            pairs.append(pair)
            labels_list.append(1)  # Positive pair

        # Generate negative pairs
        while len(pairs) < n_pairs:
            cat1, cat2 = rng.choice(categories, size=2, replace=False)

            indx1 = rng.choice(category_indices[cat1])
            indx2 = rng.choice(category_indices[cat2])

            pair = tuple(sorted((int(indx1), int(indx2))))

            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pairs.append(pair)
                labels_list.append(0)  # Negative pair

        # Shuffle the pairs and labels together
        permutation = rng.permutation(len(pairs))

        pairs = np.array(pairs)[permutation]
        labels_list = np.array(labels_list)[permutation]

        return pairs, labels_list



    '''
    TESTING FUNCTION FOR PAIR GENERATION
    '''