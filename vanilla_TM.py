import numpy as np


from pair_generation import (
    X_train_pairs,
    y_train_pairs,
    X_val_pairs,
    y_val_pairs,
    X_test_pairs,
    test_pair_labels,
    vectorizer
)

# Convert sparse matrices to dense arrays
X_train = X_train_pairs.astype(np.uint8).toarray()
X_val = X_val_pairs.astype(np.uint8).toarray()
X_test = X_test_pairs.astype(np.uint8).toarray()