import numpy as np
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


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

class VanillaTM:
    # Binary, unweighted tsetlin machine with standard feedback at s=1

    def __init__(self, n_features, clauses=100, threshold=15, states=100, seed=42):
        # Error checking
        if n_features < 1 or clauses < 2 or clauses % 2 or threshold < 1:
            raise ValueError("Use positive features and threshold, and an even clause count.")
        if not 1 <= states <= np.iinfo(np.int16).max // 2:
            raise ValueError("states is outside the supported range.")
        
        self.states = states
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        self.automata = np.full((clauses, 2 * n_features), states, dtype=np.int16)
        self.polarity = np.tile([1, -1], clauses // 2)

    def _literals(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] * 2 != self.automata.shape[1]:
            raise ValueError("X must be a matrix with the configured feature width.")
        if not np.all((X == 0) | (X == 1)):
            raise ValueError("X must contain binary values.")
        return np.concatenate((X == 1, X == 0), axis=1)

    @staticmethod
    def _fires(included, literals):
              return ~np.array(included & ~literals, axis=1)

    def fit_epoch(self, x, y):
        literals = self._literals(x)
        y = np.asarray(y)
        if y.shape != (len(literals),) or not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain binary label per row.")

        for index in self.rng.permutation(len(y)):
            row = literals[index]
            included = self.automata > self.states
            fires = self._fires(included, row)
            vote = np.clip(fires @ self.polarity, -self.threshold, self.threshold)
            target = 1 if y[index] == 1 else -1
            probability = (self.threshold - target * vote) / (2 * self.threshold)
            feedback = self.rng.random(np.sum(fires)) < probability

            # At s=1, standard Type I never increments true literals
            type_i = feedback & (self.polarity == target)
            decrement = type_i[:, None] & ~(fires[:, None] & row)
            type_ii = feedback & (self.polarity != target) & fires
            increment = type_ii[:, None] & included & row
            self.automata -= decrement
            self.automata += increment
            np.clip(self.automata, 1, 2 * self.states, out=self.automata)
        return self

    def predict(self, x):
         included = self.automata > self.states
         nonempty = included.any(axis=1)
         return np.fromiter(
              ((self._fires(included, row) & nonempty) @ self.polarity >= 0
               for row in self._literals(x)),
               dtype=np.uint8,
               count=len(x)
         )


def main():
    parser = argparse.ArgumentParser(description="Vanilla Tsetlin Machine at s=1.0")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--clauses", type=int, default=100, help="Number of clauses")
    parser.add_argument("--threshold", type=int, default=15, help="Voting threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Number of epochs must be positive.")


    from pair_generation import (
    X_train_pairs,
    y_train_pairs,
    X_val_pairs,
    y_val_pairs,
    X_test_pairs,
    test_pair_labels,
    vectorizer
    )

    X_train = X_train_pairs.toarray().astype(np.uint8)
    X_val = X_val_pairs.toarray().astype(np.uint8)
    X_test = X_test_pairs.toarray().astype(np.uint8)
    model = VanillaTM(
        X_train.shape[1], args.clauses, args.thershold, seed=args.seed)
    best_f1 = -1
    print(f"s = 1.0, clauses = {args.clauses}, threshold = {args.threshold}, seed={args.seed}")
    for epoch in range (1, args.epochs + 1):
        model.fit_epoch(X_train, y_train_pairs)
        predictions = model.predict(X_val)
        score = f1_score(y_val_pairs, predictions, average="macro")
        print(f"epoch {epoch}: Validation_macro_f1= {score:.4f}")

        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_automata = model.automata.copy()

    model.automata = best_automata
    predictions = model.predict(X_test_pairs.toarray().astype(np.uint8))
    print(f"best_epoch={best_epoch}")
    print(f"test_accuracy={accuracy_score(test_pair_labels, predictions):.4f}")
    print(f"test_macro_f1={f1_score(test_pair_labels, predictions, average='macro'):.4f}")
    print(confusion_matrix(test_pair_labels, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()