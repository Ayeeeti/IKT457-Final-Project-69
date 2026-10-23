from sklearn.datasets import fetch_20newsgroups

# Load training and testing datasets
train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')

# Extract documents and category labels
X_train = train_data.data
y_train = train_data.target

X_test = test_data.data
y_test = test_data.target

print("Training documents:", len(X_train))
print("Testing documents:", len(X_test))
print("Number of categories:", len(train_data.target_names))

print("Document:")
print(X_train[0])

print("\nCategory:")
print(train_data.target_names[y_train[0]])