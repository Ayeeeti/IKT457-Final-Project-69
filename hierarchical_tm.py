import numpy as np

from pair_generation import (
    X_train_pairs,
    y_train_pairs,
    X_val_pairs,
    y_val_pairs,
    X_test_pairs,
    test_pair_labels,
)

# Convert the prepared sparse matrices into dense arrays
X_train = X_train_pairs.astype(np.uint32).toarray()
X_val = X_val_pairs.astype(np.uint32).toarray()
X_test = X_test_pairs.astype(np.uint32).toarray()

# Convert labels as well
y_train = y_train_pairs.astype(np.uint32)
y_val = y_val_pairs.astype(np.uint32)
y_test = test_pair_labels.astype(np.uint32)

print("Hierarchical TM data loaded successfully")

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)